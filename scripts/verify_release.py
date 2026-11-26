"""Verify the public deployment and optionally save its exact bytes for browser QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def read_public(url):
    return subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                                    "--retry", "2", "--max-time", "45", url])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--copy", type=Path, help="Save verified public responses to a new directory")
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/"
    local = json.loads(Path("dist/release.json").read_text())
    manifest = read_public(base + "release.json")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if local != json.loads(manifest) or sha != local["commit"]:
        raise ValueError("Deployed manifest differs from this checked-out release")
    if args.copy:
        args.copy.mkdir(parents=True, exist_ok=False)

    def verify(item):
        path, expected = item
        received = read_public(base + path)
        if hashlib.sha256(received).hexdigest() != expected:
            raise ValueError(f"Deployed bytes differ for {path}")
        if args.copy:
            output = args.copy / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(received)
        return path

    with ThreadPoolExecutor(max_workers=5) as pool:
        paths = list(pool.map(verify, local["files"].items()))
    if args.copy:
        (args.copy / "release.json").write_bytes(manifest)
    print(json.dumps(dict(commit=sha, url=base, verified_files=len(paths), sha256_match=True)))


if __name__ == "__main__":
    main()
