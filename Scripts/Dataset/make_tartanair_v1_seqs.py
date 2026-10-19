"""Write a TartanAir_NoIMU sequence list for TartanAir v1 read in place from its zips.

    python Scripts/Dataset/make_tartanair_v1_seqs.py --data D:/Datasets/TartanAirV1 \
        --out Config/Sequence/Training_Dataset/TartanAirV1_Zip.yaml

Scans <data>/<env>/<Easy|Hard>/ for finished zips (from download_tartanair_v1.py) and lists every
P### present in all required archives. Each entry's root is <data>/<env>/<level>/P###, which does not
exist on disk: TartanAir.resolve_dir reads it from the zips next to it. An environment/level whose
zips are not all finished yet is skipped (re-run once the download completes).
"""
import argparse
import zipfile
from collections import Counter
from pathlib import Path

REQUIRED = ["image_left", "image_right", "flow_flow", "flow_mask"]


def sequences(archive: Path) -> Counter:
    """P### -> number of files in its modality directory."""
    counts = Counter()
    for name in zipfile.ZipFile(archive).namelist():
        parts = name.split("/")
        if len(parts) == 5 and parts[4]:            # env/level/P###/<subdir>/<file>
            counts[parts[2]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-depth", action="store_true", help="omit depth_left (gtDepth: false)")
    ap.add_argument("--envs", nargs="+", help="subset of environments (default: all found)")
    args = ap.parse_args()

    required = REQUIRED + ([] if args.no_depth else ["depth_left"])
    entries, frames, skipped = [], 0, []
    for level_dir in sorted(p for p in args.data.glob("*/*") if p.is_dir()):
        env, level = level_dir.parent.name, level_dir.name
        if args.envs and env not in args.envs:
            continue
        zips = [level_dir / f"{m}.zip" for m in required]
        if not all(z.exists() for z in zips):
            skipped.append(f"{env}/{level}")
            continue
        counts = [sequences(z) for z in zips]
        for seq in sorted(set.intersection(*(set(c) for c in counts))):
            entries.append((f"{env}_{level}_{seq}", (level_dir / seq).as_posix()))
            frames += counts[0][seq]

    lines = []
    for name, root in entries:
        lines += ["-   type: TartanAir_NoIMU",
                  f"    name: {name}",
                  "    args:",
                  f"        root: {root}",
                  "        compressed: false",
                  f"        gtDepth: {'false' if args.no_depth else 'true'}",
                  "        gtPose: true",
                  "        gtFlow: true"]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(entries)} sequences, {frames} frames -> {args.out}")
    if skipped:
        print(f"skipped (zips missing or still downloading): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
