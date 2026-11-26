"""Build a static replay, its evidence, and an exact-source download from a clean commit."""
from base64 import b64encode
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import markdown


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/sokldjs554/roadlens-int8"
STYLE = """
body{max-width:74rem;margin:0 auto;padding:2rem clamp(1rem,4vw,4rem);background:#10172e;color:#edf0ff;
font:1rem/1.65 system-ui,sans-serif;word-break:keep-all;overflow-wrap:anywhere}a{color:#a9bfff}
h1,h2{line-height:1.3}h2{margin-top:2.4rem}table{display:block;overflow:auto;border-collapse:collapse}
th,td{border:1px solid #40517f;padding:.6rem .8rem;text-align:left;min-width:6rem}
pre{overflow:auto;padding:1rem;background:#202e52}nav{display:flex;gap:1rem;flex-wrap:wrap}
"""


def generate_page(source, dest, sha):
    body = markdown.markdown(source.read_text(), extensions=["tables", "fenced_code"])
    def file_link(match):
        href = match[1]
        if urlsplit(href).scheme or href.startswith('#'):
            return match[0]
        path = posixpath.normpath(str(source.parent.relative_to(ROOT)) + '/' + href)
        return f'href="{REPO}/blob/{sha}/{path.lstrip("/")}"'
    body = re.sub(r'href="([^"]+)"', file_link, body)
    title = source.read_text().splitlines()[0].lstrip("# ")
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · RoadLens</title><style>{STYLE}</style><nav><a href="../">RoadLens 장면 비교</a><a href="index.html">실험 근거</a><a href="runtime.html">연산 배치</a><a href="../downloads/roadlens-source.zip">소스 ZIP</a></nav><main>{body}</main></html>"""
    dest.write_text(page)


def build():
    if subprocess.check_output(["git","status","--porcelain"], cwd=ROOT).strip():
        raise SystemExit("Build from a clean Git checkout so the published SHA identifies the inputs.")
    sha = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory() as temporary:
        dest = Path(temporary) / "dist"
        shutil.copytree(ROOT / "web", dest)
        # Some static hosts return the full file for byte-range requests. An inline
        # media source keeps native seeking available without JavaScript on those hosts.
        player = dest / "replay.html"
        video = b64encode((dest / "roadlens-replay.mp4").read_bytes()).decode("ascii")
        player.write_text(player.read_text().replace('src="roadlens-replay.mp4"',
                                                    f'src="data:video/mp4;base64,{video}"', 1))
        evidence = dest / "evidence"
        evidence.mkdir()
        for name, target in (("runtime", "runtime"), ("evidence", "index"), ("training", "training"),
                             ("evaluation", "evaluation"), ("references","references"), ("brief","brief")):
            generate_page(ROOT / "docs" / f"{name}.md", evidence / f"{target}.html", sha)
        for name, target in (("benchmark.json","benchmark.json"),("scene_metrics.csv","results.csv"),
                             ("latency.csv","latency.csv")):
            shutil.copy2(ROOT / "artifacts" / name, evidence / target)
        downloads = dest / "downloads"
        downloads.mkdir()
        subprocess.run(["git", "archive", "--format=zip", "--prefix=roadlens-int8/",
                        "-o", str(downloads / "roadlens-source.zip"), sha], cwd=ROOT, check=True)
        files = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(dest.rglob("*")) if p.is_file()}
        (dest/"release.json").write_text(json.dumps(dict(commit=sha, files=files),indent=2)+"\n")
        shutil.copytree(dest, ROOT / "dist", dirs_exist_ok=True)
    print(f"{sha} · {len(files)} hashed assets")


if __name__ == "__main__":
    build()
