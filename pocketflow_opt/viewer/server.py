"""Zero-dependency molecule viewer for generation evolution.

Scans a results root for *.sdf tracks (one track = one generation run, AL
round, or PPO epoch), serves them over HTTP, and renders molecules in 3D
with 3Dmol.js (loaded from CDN). Pure stdlib -- no new dependencies.

    python scripts/mol_viewer.py --root gen_results --port 8765
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pocketflow-opt viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body { margin:0; font:13px monospace; background:#18181c; color:#ddd;
         display:grid; grid-template-columns: 260px 300px 1fr; height:100vh; }
  #tracks, #mols { overflow-y:auto; border-right:1px solid #333; padding:6px; }
  .track, .mol { padding:3px 6px; cursor:pointer; border-radius:3px;
                 white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .track:hover, .mol:hover { background:#2a2a33; }
  .sel { background:#3b4a6b; }
  #right { display:flex; flex-direction:column; }
  #view { flex:1; position:relative; }
  #bar { padding:6px; background:#202027; display:flex; gap:8px;
         align-items:center; }
  button { background:#3b4a6b; color:#ddd; border:0; padding:4px 10px;
           cursor:pointer; border-radius:3px; font:inherit; }
  #slider { flex:1; }
  #info { padding:4px 6px; color:#888; max-height:70px; overflow-y:auto;
          white-space:pre-wrap; }
</style></head>
<body>
  <div id="tracks"></div>
  <div id="mols"></div>
  <div id="right">
    <div id="view"></div>
    <div id="bar">
      <button onclick="show(i-1)">&#9664;</button>
      <button id="play" onclick="play()">play</button>
      <button onclick="show(i+1)">&#9654;</button>
      <input id="slider" type="range" min="0" max="0" value="0"
             oninput="show(this.value)">
      <span id="counter"></span>
    </div>
    <div id="info"></div>
  </div>
<script>
let tracks = [], track = null, mols = [], i = 0, viewer = null, timer = null;
let pocketModel = null, ligandModel = null;

async function api(u) { const r = await fetch(u); return r.json(); }

async function init() {
  viewer = $3Dmol.createViewer("view", {backgroundColor: "#18181c"});
  tracks = await api("/api/tracks");
  const el = document.getElementById("tracks");
  tracks.forEach((t, k) => {
    const d = document.createElement("div");
    d.className = "track"; d.title = t.id;
    d.textContent = t.label + "  (" + t.n_mols + (t.has_pocket ? ", P" : "") + ")";
    d.onclick = () => select(k);
    el.appendChild(d);
  });
  if (tracks.length) select(0);
}

async function select(k) {
  track = tracks[k];
  document.querySelectorAll(".track").forEach((d, j) =>
    d.classList.toggle("sel", j === k));
  mols = await api("/api/mols?track=" + encodeURIComponent(track.id));
  document.getElementById("info").textContent = track.info || "";
  const el = document.getElementById("mols");
  el.innerHTML = "";
  mols.forEach((m, j) => {
    const d = document.createElement("div");
    d.className = "mol"; d.textContent = m.label;
    d.onclick = () => show(j);
    el.appendChild(d);
  });
  document.getElementById("slider").max = Math.max(0, mols.length - 1);
  viewer.clear(); pocketModel = null; ligandModel = null;
  const r = await fetch("/api/pocket?track=" + encodeURIComponent(track.id));
  if (r.ok) {
    pocketModel = viewer.addModel(await r.text(), "pdb");
    pocketModel.setStyle({cartoon: {color: "spectrum", opacity: 0.85}});
  }
  show(0);
  viewer.zoomTo();           // camera set once per track; stable while stepping
}

async function show(k) {
  if (!track || !mols.length) return;
  i = (k + mols.length) % mols.length;
  const m = await api("/api/mol?track=" + encodeURIComponent(track.id)
                      + "&i=" + i);
  if (ligandModel) viewer.removeModel(ligandModel);
  ligandModel = viewer.addModel(m.sdf, "sdf");
  ligandModel.setStyle({stick: {radius: 0.15}, sphere: {scale: 0.25}});
  viewer.render();
  document.querySelectorAll(".mol").forEach((d, j) =>
    d.classList.toggle("sel", j === i));
  document.getElementById("slider").value = i;
  document.getElementById("counter").textContent =
    (i + 1) + "/" + mols.length + "  " + m.label;
}

function play() {
  if (timer) { clearInterval(timer); timer = null;
               document.getElementById("play").textContent = "play"; }
  else { timer = setInterval(() => show(i + 1), 800);
         document.getElementById("play").textContent = "pause"; }
}

document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") show(i + 1);
  if (e.key === "ArrowLeft") show(i - 1);
});
init();
</script>
</body></html>
"""


def split_sdf(path: Path) -> list[str]:
    """SDF file -> list of mol blocks (without the $$$$ separators)."""
    text = path.read_text(errors="replace")
    return [b for b in (blk.strip() for blk in text.split("$$$$")) if b]


def _smiles_for(sdf: Path) -> list[str]:
    smi = sdf.with_suffix(".smi")
    if not smi.exists():
        return []
    return [ln.split()[0] for ln in smi.read_text(errors="replace").splitlines()
            if ln.strip()]


def discover(root: Path) -> list[dict]:
    """One track per *.sdf under root, oldest first (evolution order)."""
    root = Path(root)
    tracks = []
    for sdf in sorted(root.rglob("*.sdf"), key=lambda p: p.stat().st_mtime):
        info = ""
        readme = sdf.parent / "readme.txt"
        if readme.exists():
            info = readme.read_text(errors="replace")[:2000]
        tracks.append({
            "id": sdf.relative_to(root).as_posix(),
            "label": sdf.relative_to(root).with_suffix("").as_posix(),
            "n_mols": len(split_sdf(sdf)),
            "info": info,
        })
    return tracks


def _resolve(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if p.suffix != ".sdf" or root.resolve() not in p.parents:
        raise ValueError(f"bad track path: {rel}")
    return p


def _pocket_path(root: Path, rel: str, pocket_root: Path | None):
    """Pocket PDB for a track: <run dir>/pocket.pdb wins; otherwise the
    'pocket:' path from readme.txt, resolved absolute, then under
    pocket_root, then under the results root."""
    sdf = _resolve(root, rel)
    local = sdf.parent / "pocket.pdb"
    if local.exists():
        return local
    readme = sdf.parent / "readme.txt"
    if readme.exists():
        for line in readme.read_text(errors="replace").splitlines():
            if line.startswith("pocket:"):
                raw = line.split(":", 1)[1].strip()
                cand = Path(raw)
                if cand.is_absolute():
                    return cand if cand.exists() else None
                for base in (pocket_root, root):
                    if base is not None and (base / raw).exists():
                        return base / raw
    return None


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8765,
                pocket_root: Path | None = None):
    root = Path(root).resolve()
    cache: dict[str, tuple[float, list[str], list[str]]] = {}

    def blocks_and_labels(sdf: Path):
        mt = sdf.stat().st_mtime
        key = str(sdf)
        if key not in cache or cache[key][0] != mt:
            blocks = split_sdf(sdf)
            smiles = _smiles_for(sdf)
            labels = [smiles[k] if k < len(smiles) else f"mol {k}"
                      for k in range(len(blocks))]
            cache[key] = (mt, blocks, labels)
        return cache[key][1], cache[key][2]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, ctype: str = "application/json",
                  code: int = 200):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if url.path == "/":
                    self._send(_PAGE, "text/html")
                elif url.path == "/api/tracks":
                    tracks = discover(root)
                    for t in tracks:
                        t["has_pocket"] = (
                            _pocket_path(root, t["id"], pocket_root)
                            is not None)
                    self._send(json.dumps(tracks))
                elif url.path == "/api/pocket":
                    p = _pocket_path(root, q["track"][0], pocket_root)
                    if p is None:
                        raise ValueError("no pocket for track")
                    self._send(p.read_text(errors="replace"),
                               "chemical/x-pdb")
                elif url.path == "/api/mols":
                    sdf = _resolve(root, q["track"][0])
                    _, labels = blocks_and_labels(sdf)
                    self._send(json.dumps(
                        [{"i": k, "label": v} for k, v in enumerate(labels)]))
                elif url.path == "/api/mol":
                    sdf = _resolve(root, q["track"][0])
                    blocks, labels = blocks_and_labels(sdf)
                    k = int(q["i"][0]) % len(blocks)
                    self._send(json.dumps({"sdf": blocks[k],
                                           "label": labels[k]}))
                else:
                    self._send("not found", "text/plain", 404)
            except (KeyError, IndexError, ValueError, OSError) as e:
                self._send(str(e), "text/plain", 400)

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765,
          pocket_root: Path | None = None) -> None:
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"results root not found: {root}")
    srv = make_server(root, host, port,
                      pocket_root=Path(pocket_root) if pocket_root else None)
    n = len(discover(root))
    print(f"serving {n} track(s) under {root} at http://{host}:{srv.server_address[1]}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
