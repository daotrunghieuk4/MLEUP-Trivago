"""Pack the two original CSVs into Git-sized pieces and restore them byte for byte."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


CSV_NAMES = ("train.csv", "item_metadata.csv")
MIB = 1024 * 1024


def pack(raw_dir, parts_dir, part_mib):
    if not 1 <= part_mib <= 99:
        raise ValueError("Part size must be between 1 and 99 MiB for regular GitHub Git.")
    parts_dir.mkdir(parents=True, exist_ok=False)
    hashes = []
    with TemporaryDirectory(dir=parts_dir.parent) as temp:
        archive = Path(temp) / "trivago.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=3) as zipped:
            for name in CSV_NAMES:
                zipped.write(raw_dir / name, name)
        with archive.open("rb") as source:
            while chunk := source.read(part_mib * MIB):
                (parts_dir / f"trivago.part{len(hashes):03d}").write_bytes(chunk)
                hashes.append(hashlib.sha256(chunk).hexdigest())
    (parts_dir / "manifest.json").write_text(json.dumps({"parts_sha256": hashes}, indent=2) + "\n",
                                            encoding="utf-8")


def unpack(parts_dir, raw_dir):
    hashes = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))["parts_sha256"]
    raw_dir.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(dir=raw_dir.parent) as temp:
        archive = Path(temp) / "trivago.zip"
        with archive.open("wb") as output:
            for index, expected in enumerate(hashes):
                part = parts_dir / f"trivago.part{index:03d}"
                chunk = part.read_bytes()
                if hashlib.sha256(chunk).hexdigest() != expected:
                    raise ValueError(f"File không khớp SHA-256: {part}")
                output.write(chunk)
        with ZipFile(archive) as zipped:
            for name in CSV_NAMES:
                with zipped.open(name) as source, (raw_dir / name).open("wb") as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("--part-mib", type=int, default=90)
    args = parser.parse_args()
    pack(args.raw_dir, args.parts_dir, args.part_mib)


if __name__ == "__main__":
    main()
