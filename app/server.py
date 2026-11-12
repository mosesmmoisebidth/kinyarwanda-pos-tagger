"""Kinyarwanda POS tagger UI: switch between the trained models, compare a model pair on your own
text, and browse the test-set results.

    python app/server.py                 # then open http://127.0.0.1:8790
    python app/server.py --max_loaded 2  # keep two models in RAM (faster pair comparison)

Checkpoints are read from models/<recipe>_<model>/model.pt (the fp16 files written by training);
results/summary.json, pairs.json and disagreements.json come from src/compare_pairs.py.
"""
import argparse
import gc
import glob
import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch  # noqa: E402

from compare_pairs import MODELS, PAIRS, RECIPES  # noqa: E402
from kinpos.batching import encode_sentences  # noqa: E402
from kinpos.pretok import pretokenize  # noqa: E402
from tag_text import load_model  # noqa: E402
from train_tagger import run_eval  # noqa: E402

STATIC = os.path.join(ROOT, "app", "static")
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“«(])")
MAX_CHARS = 20000
DEVICE = torch.device("cpu")   # set from --device in main()


def free_ram_gb():
    """Available physical memory in GB (Windows and Linux), or None if unknown."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
            m = MS()
            m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.avail / 2**30
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return None


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def discover(models_dir, results_dir):
    """Every models/<recipe>_<model>/model.pt becomes a selectable model."""
    summary = read_json(os.path.join(results_dir, "summary.json"), {})
    out = OrderedDict()
    order = [k for a, b, _ in PAIRS for k in (a, b)]      # each pair side by side: general, Afro-centric
    for recipe in RECIPES:
        for key in order:
            d = os.path.join(models_dir, f"{recipe}_{key}")
            if not os.path.isfile(os.path.join(d, "model.pt")):
                continue
            m = read_json(os.path.join(d, "metrics.json"), {})
            s = summary.get(f"{recipe}/{key}", {})
            out[f"{recipe}_{key}"] = {
                "id": f"{recipe}_{key}", "recipe": recipe, "model": key, **MODELS[key],
                "path": os.path.join(d, "model.pt"),
                "ckpt_dev": m.get("dev", {}).get("acc"), "ckpt_test": m.get("test", {}).get("acc"),
                "mean_test": (s.get("test_acc") or [None])[0], "std_test": (s.get("test_acc") or [None, None])[1],
                "mean_oov": (s.get("test_oov") or [None])[0], "seeds": len(s.get("seeds", [])),
                "size_mb": round(os.path.getsize(os.path.join(d, "model.pt")) / 2**20),
            }
    for e in out.values():  # the other model of its pair, same recipe
        for a, b, _ in PAIRS:
            other = b if e["model"] == a else a if e["model"] == b else None
            if other and f"{e['recipe']}_{other}" in out:
                e["partner"] = f"{e['recipe']}_{other}"
    return out


class Manager:
    """Loads checkpoints on demand and keeps at most max_loaded in RAM (least recently used goes first)."""

    def __init__(self, entries, max_loaded, precision):
        self.entries, self.max_loaded, self.precision = entries, max_loaded, precision
        self.loaded = OrderedDict()
        self.loading = None
        self.lock = threading.Lock()

    def _get(self, mid):
        if mid in self.loaded:
            self.loaded.move_to_end(mid)
            return self.loaded[mid]
        while len(self.loaded) >= self.max_loaded:
            self.loaded.popitem(last=False)
            gc.collect()
        self.loading = mid
        try:
            e = self.entries[mid]
            t0 = time.time()
            # fp32 is about 3x faster on most CPUs; bf16 halves the RAM. auto = fp32 when the model fits easily.
            need_gb = 2.2 * e["size_mb"] / 1024 + 0.8
            free = free_ram_gb()
            fp32 = self.precision == "fp32" or (self.precision == "auto" and (free is None or free > need_gb))
            if DEVICE.type == "cuda":
                fp32 = True      # GPU: plenty of speed, keep full precision
            self.loaded[mid] = load_model(e["path"], DEVICE, local_files_only=False,
                                          encoder_dtype=None if fp32 else torch.bfloat16)
            print(f"loaded {mid} ({'fp32' if fp32 else 'bf16'}, free RAM {free or 0:.1f} GB) in {time.time() - t0:.1f}s",
                  flush=True)
        finally:
            self.loading = None
        return self.loaded[mid]

    def tag(self, mid, sents):
        with self.lock, torch.inference_mode():
            model, tok, chars, max_len = self._get(mid)
            chunks = encode_sentences(sents, tok, chars, max_len)
            preds = run_eval(model, chunks, sents, tok.pad_token_id, DEVICE, 16, False)
            model.eval()
        return [[[w, t] for w, t in zip(s["words"], p)] for s, p in zip(sents, preds)]


def split_text(text):
    sents = []
    for para in text[:MAX_CHARS].splitlines():
        for chunk in SENT_END.split(para.strip()):
            words = pretokenize(chunk)
            if words:
                sents.append({"words": words, "tags": None, "lang": "kin"})
    return sents


def make_handler(mgr, results_dir):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False))

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                models = [{k: v for k, v in e.items() if k != "path"} | {"loaded": e["id"] in mgr.loaded}
                          for e in mgr.entries.values()]
                return self._json({"models": models, "loading": mgr.loading, "max_loaded": mgr.max_loaded,
                                   "recipes": RECIPES, "pairs": [list(p) for p in PAIRS], "labels": MODELS})
            if path == "/api/results":
                return self._json({"summary": read_json(os.path.join(results_dir, "summary.json"), {}),
                                   "pairs": read_json(os.path.join(results_dir, "pairs.json"), [])})
            if path == "/api/disagreements":
                return self._json(read_json(os.path.join(results_dir, "disagreements.json"), []))
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.normpath(os.path.join(STATIC, rel))
            if not full.startswith(STATIC) or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype)

        def do_POST(self):
            if self.path != "/api/tag":
                return self._json({"error": "not found"}, 404)
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    body = json.loads(raw.decode("utf-8"))
                except UnicodeDecodeError:
                    body = json.loads(raw.decode("cp1252", errors="replace"))
                mid = body.get("model")
                if mid not in mgr.entries:
                    return self._json({"error": "Unknown model."}, 400)
                sents = split_text(body.get("text", ""))
                if not sents:
                    return self._json({"error": "Type or paste some Kinyarwanda text first."}, 400)
                t0 = time.time()
                out = {"model": mid, "sentences": mgr.tag(mid, sents)}
                partner = mgr.entries[mid].get("partner")
                if body.get("compare") and partner:
                    out["partner"] = partner
                    out["partner_sentences"] = mgr.tag(partner, sents)
                out["ms"] = round(1000 * (time.time() - t0))
                self._json(out)
            except Exception as e:  # report instead of dropping the connection
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--max_loaded", type=int, default=1, help="models kept in RAM at once")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto",
                    help="fp32 is faster on CPU, bf16 needs half the RAM; auto picks fp32 when RAM allows")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto (GPU when available, e.g. on Colab)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    global DEVICE
    DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    entries = discover(args.models_dir, args.results_dir)
    print(f"{len(entries)} model(s): {', '.join(entries) or 'none found in ' + args.models_dir}", flush=True)
    mgr = Manager(entries, max(1, args.max_loaded), args.precision)
    print(f"SERVING http://{args.host}:{args.port} on {DEVICE}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(mgr, args.results_dir)).serve_forever()


if __name__ == "__main__":
    main()
