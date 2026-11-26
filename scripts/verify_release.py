"""Verify every public byte against the locally built release and its Git revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/"
    local = json.loads(Path("dist/release.json").read_text())
    with urlopen(base+"release.json", timeout=30) as response:
        remote = json.load(response)
    sha = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if local != remote or sha != remote["commit"]:
        raise ValueError("Deployed manifest differs from this checked-out release")
    def verify(item):
        path, expected = item
        with urlopen(base+path, timeout=30) as response:
            if hashlib.file_digest(response,"sha256").hexdigest() != expected:
                raise ValueError(f"Deployed bytes differ for {path}")
        return path
    with ThreadPoolExecutor(max_workers=5) as pool:
        paths = list(pool.map(verify, local["files"].items()))
    print(json.dumps(dict(commit=sha, url=base, verified_files=len(paths), sha256_match=True)))


if __name__ == "__main__":
    main()
