"""Convert a DomainBed / WILDS installation into the layout ``sadl.data.real`` expects.

    <root>/<environment>/<class>/<image files>

DomainBed's PACS and TerraIncognita already use this layout, so those only need to
be pointed at or symlinked.  WILDS ships flat image directories plus a metadata
CSV, so Camelyon17 and iWildCam are materialised as symlinks grouped by
(environment, class).

Examples
--------
    python scripts/prepare_real_data.py link --src ~/data/PACS --name PACS
    python scripts/prepare_real_data.py wilds \
        --src ~/data/camelyon17_v1.0 --name camelyon17 \
        --env-col center --label-col tumor --file-col filename --image-dir patches
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SADL_DATA", ROOT / "data"))


def cmd_link(args: argparse.Namespace) -> None:
    dest = DATA / args.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest} already exists")
        return
    dest.symlink_to(Path(args.src).expanduser().resolve(), target_is_directory=True)
    print(f"linked {dest} -> {args.src}")


def cmd_wilds(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser().resolve()
    meta = src / args.metadata
    if not meta.exists():
        raise SystemExit(f"metadata not found: {meta}")
    dest = DATA / args.name
    n = 0
    with open(meta, newline="") as fh:
        for row in csv.DictReader(fh):
            env = f"{args.env_col}_{row[args.env_col]}"
            label = str(row[args.label_col])
            rel = row[args.file_col]
            target = src / args.image_dir / rel if args.image_dir else src / rel
            if not target.exists():
                continue
            out_dir = dest / env / label
            out_dir.mkdir(parents=True, exist_ok=True)
            link = out_dir / Path(rel).name
            if not link.exists():
                link.symlink_to(target)
                n += 1
            if args.limit and n >= args.limit:
                break
    print(f"created {n} links under {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("link", help="symlink an already-grouped directory")
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True, help="PACS | terra_incognita | camelyon17 | iwildcam")
    p.set_defaults(func=cmd_link)

    p = sub.add_parser("wilds", help="group a flat WILDS directory by (environment, class)")
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--metadata", default="metadata.csv")
    p.add_argument("--env-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--file-col", default="filename")
    p.add_argument("--image-dir", default="")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_wilds)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
