"""Run the bloqade-decoders CPU decoders in a child process (``[decoders]`` extra).

Why: ldpc's native BpOsdDecoder segfaults on some detector error models (any
model with zero error mechanisms, e.g. a noiseless LightStim circuit, and some
small degenerate ones; bloqade-decoders 0.6.1 / ldpc 2.4.1). In-process, that
kills the whole MCP server and every other tool with it. In a child, it becomes
an ``ok=False`` result for that one decoder.

A ``forkserver`` context preloads ``bloqade.decoders`` once (~1 s import), so
each decode only pays a cheap fork; unlike plain ``fork`` it is safe in the MCP
server's multithreaded process. Only this file imports ``bloqade.decoders``.
GPU decoders stay in-process (forking a CUDA context is unsafe) and fail cleanly.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import time
from multiprocessing.connection import Connection
from multiprocessing.context import ForkServerContext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_BP_MAX_ITER = 30
_OSD_ORDER = 5
_MS_SCALING = 0.625  # standard min-sum scaling; bloqade's default of 1.0 is unscaled
# OSD-CS order 5 costs ~0.6 s/shot on the gross code (14.6k error mechanisms);
# above this size BP+OSD falls back to OSD-0 (~45 ms/shot, same LER there).
OSD_CS_MAX_ERRORS = 5_000
# The lookup table stores every syndrome seen in training, so it only makes sense for
# codes with few detectors (a d=3 repetition memory has ~8, a d=3 surface memory ~24).
TABLE_MAX_DETECTORS = 30
_TABLE_TRAIN_SHOTS = 200_000


@functools.lru_cache(maxsize=1)
def _context() -> ForkServerContext:
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["numpy", "stim", "bloqade.decoders"])
    return ctx


def _build_bp(bd: Any, name: str, dem: Any) -> Any:
    bp = {"max_iter": _BP_MAX_ITER, "ms_scaling_factor": _MS_SCALING}
    if name == "bposd":
        if dem.num_errors > OSD_CS_MAX_ERRORS:
            return bd.BpOsdDecoder(dem, osd_method="OSD_0", **bp)
        return bd.BpOsdDecoder(dem, osd_method="OSD_CS", osd_order=_OSD_ORDER, **bp)
    # bloqade-decoders 0.6.1 defaults bits_per_step=0, which breaks cluster growth:
    # BP+LSD gave 6% logical error on the noisy gross code vs 0% with bits_per_step=1.
    if name == "bplsd":
        return bd.BpLsdDecoder(dem, bits_per_step=1, **bp)
    return bd.BeliefFindDecoder(dem, bits_per_step=1, **bp)


def _build(name: str, dem: Any) -> Any:
    from bloqade import decoders as bd  # noqa: PLC0415

    if name == "tesseract":
        return bd.TesseractDecoder(dem)
    if name == "mwpf":
        return bd.MWPFDecoder(dem)
    if name == "mle":
        return bd.GurobiDecoder(dem)  # imports gurobipy: ImportError -> [mle] extra hint
    if name == "table":
        return _build_table(bd, dem)
    return _build_bp(bd, name, dem)


def _build_table(bd: Any, dem: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    if dem.num_detectors > TABLE_MAX_DETECTORS:
        raise RuntimeError(
            f"error model too large for the table decoder ({dem.num_detectors} detectors, "
            f"max {TABLE_MAX_DETECTORS})"
        )
    det, obs, _ = dem.compile_sampler(seed=0).sample(shots=_TABLE_TRAIN_SHOTS)
    return bd.TableDecoder.from_det_obs_shots(dem, np.concatenate([det, obs], axis=1))


def _child(name: str, dem_text: str, det: Any, conn: Connection) -> None:
    try:
        import stim  # noqa: PLC0415

        decoder = _build(name, stim.DetectorErrorModel(dem_text))
        start = time.perf_counter()
        pred = decoder.decode(det)
        conn.send(("ok", pred, time.perf_counter() - start))
    except ImportError as e:
        conn.send(("import", str(e), 0.0))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}", 0.0))
    finally:
        conn.close()


def predict_isolated(
    name: str, dem_text: str, det: NDArray[np.bool_], timeout: float
) -> tuple[NDArray[np.bool_], float]:
    """Decode in a child process; return (observable predictions, decode seconds).

    Raises:
        ImportError: the ``[decoders]`` extra is missing.
        RuntimeError: the decoder failed, crashed in native code, or timed out.
    """
    ctx = _context()
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(name, dem_text, det, send), daemon=True)
    proc.start()
    send.close()  # parent keeps only the read end, so a dead child reads as EOF
    try:
        if not recv.poll(timeout):
            proc.kill()
            raise RuntimeError(f"timed out after {timeout:.0f} s")
        try:
            status, payload, seconds = recv.recv()
        except EOFError:
            proc.join()
            raise RuntimeError(
                f"decoder crashed in native code (exit code {proc.exitcode}); "
                "ldpc's BP+OSD is known to do this on empty or tiny error models"
            ) from None
    finally:
        recv.close()
        proc.join(timeout=5)
    if status == "import":
        raise ImportError(payload)
    if status == "error":
        raise RuntimeError(payload)
    return payload, seconds
