"""Download the TartanAir v1 training zips (kept zipped) with resume and checksum verification.

    python Scripts/Dataset/download_tartanair_v1.py --out D:/Datasets/TartanAirV1
    python Scripts/Dataset/download_tartanair_v1.py --out D:/Datasets/TartanAirV1 --dry-run

Files land at <out>/<env>/<Easy|Hard>/<modality>.zip (the bucket layout; members inside are
<env>/<Easy|Hard>/P###/...). Each file is fetched into <name>.zip.part with HTTP Range resume,
checked against the index size and the file's SHA-256 (from the HF mirror), then renamed. Re-running
skips finished files and resumes partial ones. Before each file starts, the run stops if the free
space left after all pending downloads would drop below --min-free-gib.

tartanair_v1_zipfiles.txt is download_training_zipfiles.txt from castacks/tartanair_tools
(sizes in GB with 9 decimals, i.e. exact bytes). The mirrors serve byte-identical files.
"""
import argparse
import hashlib
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRRORS = {
    "airlab": "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair/",
    "hf":     "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/",
}
# flow_mask first: small, and enough to enumerate every sequence early.
MODALITIES = ["flow_mask", "image_left", "image_right", "depth_left", "flow_flow"]
CHUNK = 8 << 20
GIB = 1 << 30


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None


def read_index(path: Path) -> list[tuple[str, int]]:
    entries = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].endswith(".zip"):
            entries.append((parts[0], round(float(parts[1]) * 1e9)))
    return entries


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Downloader:
    def __init__(self, out: Path, jobs: list[tuple[str, int]], mirrors: list[str], min_free: int):
        self.out, self.mirrors, self.min_free = out, mirrors, min_free
        self.lock = threading.Lock()
        self.remaining = {key: 0 if self.is_done(key, size) else size - self.part_size(key) for key, size in jobs}
        self.total = sum(size for _, size in jobs)
        self.fetched = 0
        self.stop = threading.Event()

    def final(self, key: str) -> Path: return self.out / key
    def part(self, key: str) -> Path:  return self.out / (key + ".part")

    def part_size(self, key: str) -> int:
        p = self.part(key)
        return p.stat().st_size if p.exists() else 0

    def is_done(self, key: str, size: int) -> bool:
        f = self.final(key)
        return f.exists() and f.stat().st_size == size

    def add_bytes(self, key: str, n: int):
        with self.lock:
            self.remaining[key] -= n
            self.fetched += n

    def space_ok(self) -> tuple[bool, int, int]:
        free = shutil.disk_usage(self.out).free
        with self.lock:
            pending = sum(max(r, 0) for r in self.remaining.values())
        return free - pending >= self.min_free, free, pending

    @staticmethod
    def expected_digest(key: str) -> tuple[str | None, str | None, int | None]:
        """(algorithm, hexdigest, length). HF's LFS oid is the file's SHA-256 for every object. AirLab's
        ETag is an MD5 only for plain objects; static large objects (> ~5 GB) carry a manifest ETag."""
        try:
            urllib.request.build_opener(_NoRedirect).open(
                urllib.request.Request(MIRRORS["hf"] + key, method="HEAD"), timeout=60)
        except urllib.error.HTTPError as e:     # the 302 carries the LFS metadata
            sha, length = (e.headers.get("X-Linked-ETag") or "").strip('"'), e.headers.get("X-Linked-Size")
            if len(sha) == 64 and length:
                return "sha256", sha, int(length)
        except OSError:
            pass
        try:
            with urllib.request.urlopen(urllib.request.Request(MIRRORS["airlab"] + key, method="HEAD"), timeout=60) as r:
                etag, length = (r.headers.get("ETag") or "").strip('"'), int(r.headers.get("Content-Length") or -1)
                if len(etag) == 32 and not r.headers.get("X-Static-Large-Object"):
                    return "md5", etag, length
                return None, None, length
        except OSError:
            return None, None, None

    def fetch_into_part(self, key: str, url: str, h) -> None:
        part = self.part(key)
        start = self.part_size(key)
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            if start and r.status != 206:           # no resume from this server: start over next attempt
                part.unlink()
                with self.lock:
                    self.remaining[key] += start
                raise OSError(f"server ignored Range (HTTP {r.status}), discarded .part")
            with open(part, "ab") as f:
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    h.update(chunk)
                    self.add_bytes(key, len(chunk))
                    if self.stop.is_set():
                        raise InterruptedError("stopping")

    def run_one(self, key: str, size: int) -> str:
        final, part = self.final(key), self.part(key)
        if self.is_done(key, size):
            return "skip"
        if self.stop.is_set():
            return "stopped"
        ok, free, pending = self.space_ok()
        if not ok:
            log(f"STOP space guard: free {free / GIB:.0f} GiB, pending {pending / GIB:.0f} GiB, "
                f"floor {self.min_free / GIB:.0f} GiB")
            self.stop.set()
            return "stopped"

        final.parent.mkdir(parents=True, exist_ok=True)
        algo, want, length = self.expected_digest(key)
        if length not in (None, size):
            log(f"FAIL {key}: server has {length} B, index says {size} B")
            return "fail"
        t0 = time.time()
        for attempt in range(30):
            mirror = self.mirrors[attempt % len(self.mirrors)]
            h = hashlib.new(algo or "sha256")
            if part.exists():                       # resume: hash what is already on disk
                with open(part, "rb") as f:
                    while chunk := f.read(CHUNK):
                        h.update(chunk)
            try:
                if self.part_size(key) < size:
                    self.fetch_into_part(key, MIRRORS[mirror] + key, h)
                got = self.part_size(key)
                if got != size or (want and h.hexdigest() != want):
                    log(f"BAD  {key}: {got} B, {h.name} {h.hexdigest()} (want {size} B, {want}); restarting")
                    with self.lock:
                        self.remaining[key] = size
                    part.unlink()
                    continue
                part.replace(final)
                dt = time.time() - t0
                with self.lock, open(self.out / "_manifest.jsonl", "a") as f:
                    f.write(json.dumps({"key": key, "bytes": size, h.name: h.hexdigest(),
                                        "verified": want is not None, "mirror": mirror,
                                        "seconds": round(dt, 1)}) + "\n")
                log(f"ok   {key} {size / 1e9:.2f} GB in {dt / 60:.1f} min ({mirror}, "
                    f"{h.name + ' verified' if want else 'size-only check'})")
                return "ok"
            except InterruptedError:
                return "stopped"
            except Exception as e:  # network errors: back off and resume from the .part
                wait = min(60, 2 ** min(attempt, 6))
                log(f"retry {key} via {mirror} after {type(e).__name__}: {e} (sleep {wait}s)")
                time.sleep(wait)
        log(f"FAIL {key} after 30 attempts")
        return "fail"

    def report(self, every: float):
        t0, last = time.time(), 0
        while not self.stop.wait(every):
            with self.lock:
                fetched = self.fetched
                pending = sum(max(r, 0) for r in self.remaining.values())
            rate = (fetched - last) / every
            last = fetched
            free = shutil.disk_usage(self.out).free
            eta = pending / rate / 3600 if rate > 0 else float("inf")
            log(f"---- {(self.total - pending) / 1e9:.0f}/{self.total / 1e9:.0f} GB, "
                f"{rate / 1e6:.0f} MB/s, ETA {eta:.1f} h, free {free / GIB:.0f} GiB, "
                f"elapsed {(time.time() - t0) / 3600:.1f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=HERE / "tartanair_v1_zipfiles.txt")
    ap.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES + ["depth_right", "seg_left", "seg_right"])
    ap.add_argument("--envs", nargs="+", help="subset of environments (default: all)")
    ap.add_argument("--levels", nargs="+", default=["Easy", "Hard"])
    ap.add_argument("--mirrors", nargs="+", default=["airlab", "hf"], choices=list(MIRRORS),
                    help="tried in turn on each retry")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gib", type=float, default=800)
    ap.add_argument("--report-every", type=float, default=300, help="seconds between progress lines")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    order = {m: i for i, m in enumerate(args.modalities)}
    jobs = []
    for key, size in read_index(args.index):
        env, level, name = key.split("/")
        mod = name[:-len(".zip")]
        if mod in order and level in args.levels and (args.envs is None or env in args.envs):
            jobs.append((key, size))
    envs = sorted({k.split("/")[0] for k, _ in jobs})
    # all masks first, then environment by environment so finished environments are usable early
    jobs.sort(key=lambda j: (j[0].endswith("flow_mask.zip") is False, j[0].split("/")[0],
                             j[0].split("/")[1], order[j[0].split("/")[2][:-4]]))

    args.out.mkdir(parents=True, exist_ok=True)
    dl = Downloader(args.out, jobs, args.mirrors, int(args.min_free_gib * GIB))
    ok, free, pending = dl.space_ok()
    log(f"{len(jobs)} files, {len(envs)} envs, {dl.total / 1e9:.1f} GB total, {pending / 1e9:.1f} GB to fetch; "
        f"free {free / GIB:.0f} GiB -> {(free - pending) / GIB:.0f} GiB after (floor {args.min_free_gib:.0f} GiB)")
    if not ok:
        log("not enough space for the selection; aborting")
        return 1
    if args.dry_run:
        for key, size in jobs:
            log(f"  {key} {size / 1e9:.2f} GB")
        return 0

    reporter = threading.Thread(target=dl.report, args=(args.report_every,), daemon=True)
    reporter.start()
    with ThreadPoolExecutor(args.workers) as pool:
        try:
            results = list(pool.map(lambda j: dl.run_one(*j), jobs))
        except KeyboardInterrupt:              # workers notice the event within one chunk; .part files resume
            dl.stop.set()
            pool.shutdown(cancel_futures=True)
            raise
    dl.stop.set()
    counts = {s: results.count(s) for s in sorted(set(results))}
    log(f"done: {counts}; free {shutil.disk_usage(args.out).free / GIB:.0f} GiB")
    return 0 if all(r in ("ok", "skip") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
