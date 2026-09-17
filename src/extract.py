"""Unpack EDRM Public Download.zip and nested text archives for RAG.

Run from the repo root:
    python src/extract.py
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

WIN_BAD = re.compile(r'[<>:"|?*]')

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "EDRM Public Download.zip"
EXTRACTED = ROOT / "data" / "extracted"
INVENTORY = ROOT / "data" / "processed" / "inventory.json"

# Nested containers that usually hold readable text for a first RAG pass.
NESTED_ZIP = {".zip"}
NESTED_GZ = {".gz"}
NESTED_TAR = {".tar", ".gtar", ".tgz", ".tar.gz"}

# Forensic / media containers: keep on disk, do not unpack for RAG v1.
SKIP_UNPACK = {".dmg", ".snoop", ".e01", ".mp3", ".jpg", ".jpeg", ".tif", ".tiff"}

TEXTISH = {
    ".txt",
    ".csv",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".md",
    ".rtf",
    ".pdf",
    ".doc",
    ".docx",
    ".docm",
    ".dotm",
    ".odt",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xltx",
    ".xla",
    ".ppt",
    ".pptx",
    ".pps",
    ".pot",
    ".mbox",
}


def sanitize_member(member: str) -> str:
    parts = []
    for part in member.replace("\\", "/").split("/"):
        if part in {"", ".", ".."}:
            continue
        cleaned = WIN_BAD.sub("_", part).rstrip(" .")
        parts.append(cleaned or "_")
    return "/".join(parts)


def safe_join(base: Path, member: str) -> Path:
    dest = (base / sanitize_member(member)).resolve()
    if not dest.is_relative_to(base.resolve()):
        raise ValueError(f"unsafe archive path: {member}")
    return dest


def unzip(src: Path, dest: Path) -> None:
    dest.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            target = safe_join(dest, info.filename)
            if info.is_dir() or info.filename.endswith("/"):
                target.mkdir(parents = True, exist_ok = True)
                continue
            target.parent.mkdir(parents = True, exist_ok = True)
            with zf.open(info) as src_f, target.open("wb") as out_f:
                shutil.copyfileobj(src_f, out_f)


def gunzip_file(src: Path) -> Path | None:
    if src.suffix.lower() != ".gz":
        return None
    stem = src.with_suffix("")
    if stem.suffix.lower() in {".tar", ".gtar"}:
        return None
    with gzip.open(src, "rb") as gz, stem.open("wb") as out:
        shutil.copyfileobj(gz, out)
    return stem


def untar(src: Path, dest: Path) -> None:
    dest.mkdir(parents = True, exist_ok = True)
    with tarfile.open(src, "r:*") as tf:
        for member in tf.getmembers():
            target = safe_join(dest, member.name)
            if member.isdir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents = True, exist_ok = True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            try:
                with extracted, target.open("wb") as out:
                    shutil.copyfileobj(extracted, out)
            except OSError as exc:
                print(f"skip tar member {member.name}: {exc}")


def classify(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    ext = path.suffix.lower()
    if ext in SKIP_UNPACK or suffixes.endswith((".jpg", ".jpeg", ".tif", ".tiff", ".mp3")):
        return "binary_skip"
    if ext in TEXTISH or path.suffixes[-2:] in ([".csv", ".gz"],):
        return "text_candidate"
    if ext in NESTED_ZIP or suffixes.endswith((".tar.gz", ".gtar.gz", ".tgz")) or ext == ".gz":
        return "nested_archive"
    return "other"


def unpack_nested(extracted: Path) -> None:
    """Unpack zip/gz/tar that typically contain CSV, HTML, or web text."""
    archives = list(extracted.rglob("*"))
    for path in archives:
        if not path.is_file():
            continue
        suffixes = "".join(path.suffixes).lower()
        ext = path.suffix.lower()
        if ext in SKIP_UNPACK:
            continue
        out_dir = path.with_name(path.stem + "_unpacked")
        try:
            if ext == ".zip":
                unzip(path, out_dir)
            elif suffixes.endswith((".tar.gz", ".gtar.gz", ".tgz", ".tar")):
                untar(path, out_dir)
            elif ext == ".gz":
                gunzip_file(path)
        except (OSError, zipfile.BadZipFile, tarfile.TarError, gzip.BadGzipFile, ValueError) as exc:
            print(f"skip nested {path.relative_to(extracted)}: {exc}")


def inventory(extracted: Path) -> dict:
    files = [p for p in extracted.rglob("*") if p.is_file()]
    by_ext = Counter()
    by_class = Counter()
    rows = []
    for path in files:
        ext = path.suffix.lower() or "(none)"
        kind = classify(path)
        by_ext[ext] += 1
        by_class[kind] += 1
        rows.append(
            {
                "path": str(path.relative_to(extracted)),
                "bytes": path.stat().st_size,
                "ext": ext,
                "class": kind,
            }
        )
    return {
        "file_count": len(files),
        "by_class": dict(by_class),
        "by_ext": dict(by_ext),
        "files": rows,
    }


def main() -> None:
    if not RAW_ZIP.exists():
        raise SystemExit(f"missing zip: {RAW_ZIP}")

    if EXTRACTED.exists():
        shutil.rmtree(EXTRACTED)
    EXTRACTED.mkdir(parents = True, exist_ok = True)

    print(f"extracting {RAW_ZIP.name} -> {EXTRACTED}")
    unzip(RAW_ZIP, EXTRACTED)
    unpack_nested(EXTRACTED)

    report = inventory(EXTRACTED)
    INVENTORY.parent.mkdir(parents = True, exist_ok = True)
    INVENTORY.write_text(json.dumps(report, indent = 2), encoding = "utf-8")

    print(f"files: {report['file_count']}")
    print("by class:", json.dumps(report["by_class"], indent = 2))
    print("top extensions:")
    for ext, count in sorted(report["by_ext"].items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:4} {ext}")
    print(f"inventory -> {INVENTORY}")
    print(
        "next: parse text_candidate files into data/processed "
        "(PDF/Office/mbox/CSV), then chunk and embed into data/index."
    )


if __name__ == "__main__":
    main()
