"""Sources page — searchable, filterable registry table."""

from pathlib import Path

import streamlit as st

from chatqec.ingestion.registry import Registry, SourceStatus

st.title("Sources")

reg_path = Path("sources/registry.jsonl")
if not reg_path.exists():
    st.info("No registry yet — run `chatqec ingest` to add sources.")
    st.stop()

reg = Registry(reg_path)
records = list(reg.all())
if not records:
    st.info("Registry is empty.")
    st.stop()

# Build a plain list-of-dicts (no pandas dep) for Streamlit's native dataframe widget
rows: list[dict[str, object]] = [
    {
        "id": r.id,
        "type": r.type.value,
        "title": r.title,
        "tier": r.authority_tier,
        "status": r.status.value,
        "year": r.year,
        "verified": r.verified,
        "chunks": r.chunk_count,
    }
    for r in records
]

all_types = sorted({str(row["type"]) for row in rows})
all_tiers = sorted({str(row["tier"]) for row in rows})
all_statuses = [s.value for s in SourceStatus]

c1, c2, c3 = st.columns(3)
with c1:
    types = st.multiselect("Type", all_types, default=all_types)
with c2:
    statuses = st.multiselect("Status", all_statuses, default=["active"])
with c3:
    tiers = st.multiselect("Authority tier", all_tiers, default=all_tiers)

filtered = [
    row
    for row in rows
    if row["type"] in types and row["status"] in statuses and row["tier"] in tiers
]
st.caption(f"{len(filtered)} / {len(rows)} records")
st.dataframe(filtered, use_container_width=True, hide_index=True)

picked = st.selectbox("Inspect a record", [""] + [str(row["id"]) for row in filtered])
if picked:
    record = reg.get(picked)
    st.json(record.model_dump_json(indent=2))
