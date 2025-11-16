#!/usr/bin/env python3
"""
Precompute enhanced ontology JSON for bundled sample PDFs.

Outputs:
  precomputed_samples/<stable_document_id>/
    - <stable_document_id>_enhanced.json
    - <stable_document_id>_index.json
    - metadata.json  (includes original filename and stable_document_id)
"""
import json
import os
from pathlib import Path
from datetime import datetime

from ehrx.vlm.pipeline import DocumentPipeline
from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = (ROOT / "SampleEHR_docs").resolve()
OUT_DIR = (ROOT / "precomputed_samples").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slugify_filename(filename: str) -> str:
    base = Path(filename).stem.lower()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in base)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-_") or "sample"


def get_paths(document_id: str) -> dict:
    doc_dir = OUT_DIR / document_id
    return {
        "dir": doc_dir,
        "enhanced": doc_dir / f"{document_id}_enhanced.json",
        "index": doc_dir / f"{document_id}_index.json",
        "metadata": doc_dir / "metadata.json",
    }


def main():
    if not SAMPLES_DIR.exists():
        raise SystemExit(f"Sample docs directory not found: {SAMPLES_DIR}")

    vlm_config = VLMConfig.from_env()
    pipeline = DocumentPipeline(vlm_config=vlm_config, checkpoint_interval=50, dpi=200)

    for pdf in sorted(SAMPLES_DIR.glob("*.pdf")):
        stable_id = f"sample_{slugify_filename(pdf.name)}"
        paths = get_paths(stable_id)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        if paths["enhanced"].exists() and paths["metadata"].exists():
            print(f"[skip] Already precomputed: {pdf.name} -> {stable_id}")
            continue

        print(f"[run] Processing: {pdf} -> {stable_id}")
        document = pipeline.process_document(
            pdf_path=str(pdf),
            output_dir=str(paths["dir"]),
            page_range="all",
            document_context={"document_type": "Clinical EHR"},
        )

        grouper = SubDocumentGrouper(confidence_threshold=0.80)
        enhanced_doc = grouper.group_document(document)

        with open(paths["enhanced"], "w") as f:
            json.dump(enhanced_doc, f, indent=2)

        index = generate_hierarchical_index(enhanced_doc)
        with open(paths["index"], "w") as f:
            json.dump(index, f, indent=2)

        metadata = {
            "stable_document_id": stable_id,
            "original_filename": pdf.name,
            "created_at": datetime.utcnow().isoformat(),
        }
        with open(paths["metadata"], "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[ok] Wrote: {paths['enhanced']} and {paths['metadata']}")


if __name__ == "__main__":
    main()


