"""IQM worker: runs inside the separate IQM interpreter, never inside the MCP server.

iqm-client pins packaging<25 and pydantic<2.13, which conflicts with the deltakit and
pydantic versions the main tool environment needs, so it lives in its own venv
(scripts/setup_iqm_env.sh) and this script is the only code that imports it. It must not
import chatqec_mcp_tools: it is run by path (``python -I _worker.py``), with the IQM venv's
interpreter.

Protocol: one JSON request on stdin, one JSON reply on stdout.
  request  {"op": "snapshot" | "submit" | "job", ...op fields}
  reply    {"ok": true, "result": {...}} or {"ok": false, "kind": "auth"|"error", "error": "..."}
Credentials come from the environment only (IQM_TOKEN, IQM_SERVER_URL, optional
IQM_QUANTUM_COMPUTER) and are never echoed back.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

# Quality-metric and calibration fields read from the server, verified live 2026-11-12
# against the ORNL 20-qubit IQM system (dut M194_F0W1388_P08_Q12).
_QUBIT_FIELDS = {
    "characterization.model.{q}.t1_time": "t1_s",
    "characterization.model.{q}.t2_time": "t2_s",
    "characterization.model.{q}.t2_echo_time": "t2_echo_s",
    "metrics.ssro.measure.constant.{q}.fidelity": "readout_fidelity",
    "metrics.ssro.measure.constant.{q}.error_0_to_1": "readout_error_0_to_1",
    "metrics.ssro.measure.constant.{q}.error_1_to_0": "readout_error_1_to_0",
    "metrics.rb.prx.drag_crf_sx.{q}.fidelity:par=d2": "prx_fidelity",
}
_QUBIT_CAL_FIELDS = {
    "gates.prx.drag_crf_sx.{q}.duration": "prx_duration_s",
    "gates.measure.constant.{q}.duration": "measure_duration_s",
}
_PAIR_METRIC = re.compile(r"^metrics\.irb\.cz\.[^.]+\.(QB\d+)__(QB\d+)\.fidelity")
_PAIR_CLIFFORD = re.compile(r"^metrics\.rb\.clifford\.uz_cz\.(QB\d+)__(QB\d+)\.fidelity")
_PAIR_DURATION = re.compile(r"^gates\.cz\.[^.]+\.(QB\d+)__(QB\d+)\.duration$")


class AuthError(RuntimeError):
    pass


_TOKEN = [""]  # the token, once popped from the environment by _settings()


def _settings() -> tuple[str, str, str | None]:
    # iqm-client also reads IQM_TOKEN from the environment itself and refuses a token
    # given both ways ("Parameter sources must not be mixed"), so take it out of the env.
    token = os.environ.pop("IQM_TOKEN", "").strip() or _TOKEN[0]
    _TOKEN[0] = token
    url = os.environ.get("IQM_SERVER_URL", "").strip()
    if not token:
        raise AuthError("IQM_TOKEN is not set.")
    if not url:
        raise AuthError("IQM_SERVER_URL is not set.")
    return token, url, os.environ.get("IQM_QUANTUM_COMPUTER") or None


def _client() -> Any:
    from iqm.iqm_client import IQMClient  # type: ignore[import-not-found]  # noqa: PLC0415

    token, url, qc = _settings()
    return IQMClient.from_url(url, quantum_computer=qc, token=token)


def _backend() -> Any:
    from iqm.qiskit_iqm import IQMProvider  # type: ignore[import-not-found]  # noqa: PLC0415

    token, url, qc = _settings()
    return IQMProvider(url, quantum_computer=qc, token=token).get_backend()


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if int(a[2:]) < int(b[2:]) else (b, a)


def _snapshot(_req: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    static = client.get_static_quantum_architecture()
    dynamic = client.get_dynamic_quantum_architecture()
    metrics = {o.dut_field: o.value for o in client.get_quality_metric_set().observations}
    calibration = {o.dut_field: o.value for o in client.get_calibration_set().observations}
    try:
        healthy = bool(client._executor.client.get_health().get("healthy", False))
    except Exception:
        healthy = None

    qubits = []
    for name in static.qubits:
        entry: dict[str, Any] = {"name": name}
        for template, key in _QUBIT_FIELDS.items():
            entry[key] = metrics.get(template.format(q=name))
        for template, key in _QUBIT_CAL_FIELDS.items():
            entry[key] = calibration.get(template.format(q=name))
        qubits.append(entry)

    couplers: dict[tuple[str, str], dict[str, Any]] = {
        _pair_key(a, b): {"qubits": list(_pair_key(a, b))} for a, b in static.connectivity
    }
    for source, pattern, key in (
        (metrics, _PAIR_METRIC, "cz_fidelity"),
        (metrics, _PAIR_CLIFFORD, "clifford_cz_fidelity"),
        (calibration, _PAIR_DURATION, "cz_duration_s"),
    ):
        for field, value in source.items():
            m = pattern.match(field)
            if m and _pair_key(*m.groups()) in couplers:
                couplers[_pair_key(*m.groups())][key] = value

    return {
        "quantum_computer": os.environ.get("IQM_QUANTUM_COMPUTER") or "default",
        "dut_label": static.dut_label,
        "healthy": healthy,
        "calibration_set_id": str(dynamic.calibration_set_id),
        "native_gates": sorted(dynamic.gates),
        "qubits": qubits,
        "couplers": list(couplers.values()),
    }


def _submit(req: dict[str, Any]) -> dict[str, Any]:
    from qiskit import QuantumCircuit, transpile  # noqa: PLC0415

    backend = _backend()
    circuit = QuantumCircuit.from_qasm_str(req["qasm"])
    layout = [backend.qubit_name_to_index(name) for name in req["initial_layout"]]
    isa = transpile(circuit, backend=backend, initial_layout=layout, optimization_level=1)
    job = backend.run(isa, shots=int(req["shots"]), use_timeslot=bool(req.get("use_timeslot")))
    try:
        status = str(job.status().name)
    except Exception as exc:
        status = f"unknown ({exc})"
    return {
        "job_id": str(job.job_id()),
        "status": status,
        "backend": backend.name,
        "transpiled_two_qubit_gate_count": int(isa.count_ops().get("cz", 0)),
        "transpiled_depth": int(isa.depth()),
    }


def _job(req: dict[str, Any]) -> dict[str, Any]:
    backend = _backend()
    job = backend.retrieve_job(req["job_id"])
    status = str(job.status().name)
    out: dict[str, Any] = {"job_id": req["job_id"], "status": status, "counts": {}}
    if status == "DONE":
        out["counts"] = dict(job.result().get_counts())
    return out


_OPS = {"snapshot": _snapshot, "submit": _submit, "job": _job}


def _looks_like_auth(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("unauthorized", "401", "403", "forbidden", "authentic"))


def main() -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # library chatter must not corrupt the JSON reply
    token = os.environ.get("IQM_TOKEN", "").strip()
    try:
        req = json.loads(sys.stdin.read())
        reply: dict[str, Any] = {"ok": True, "result": _OPS[req["op"]](req)}
    except AuthError as exc:
        reply = {"ok": False, "kind": "auth", "error": str(exc)}
    except Exception as exc:
        kind = "auth" if _looks_like_auth(exc) else "error"
        message = f"{type(exc).__name__}: {exc}"
        if token:
            message = message.replace(token, "<redacted>")
        reply = {"ok": False, "kind": kind, "error": message[:2000]}
    real_stdout.write(json.dumps(reply))
    real_stdout.flush()


if __name__ == "__main__":
    main()
