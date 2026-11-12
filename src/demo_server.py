"""Local demo: paste Kinyarwanda text, get a POS tag for every word.

python src/demo_server.py --models models/demo/model_half.pt --port 8765
then open http://127.0.0.1:8765
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinpos.batching import encode_sentences  # noqa: E402
from kinpos.pretok import pretokenize  # noqa: E402
from tag_text import load_model  # noqa: E402
from train_tagger import run_eval  # noqa: E402

SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“«(])")
MAX_CHARS = 20000


class Engine:
    """Loads the model(s) in a background thread, waiting for the files to appear if needed,
    so the page is reachable immediately and starts tagging as soon as the model is ready."""

    def __init__(self, patterns, base_model):
        self.patterns, self.base_model = patterns, base_model
        self.models = []
        self.ready = False
        self.status = "The model is still being prepared. This page will start working by itself."
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        while True:
            paths = [p for m in self.patterns for p in sorted(glob.glob(m))]
            if paths and all(os.path.isfile(p) and not os.path.exists(p + ".part") for p in paths):
                break
            time.sleep(5)
        self.status = "Loading the model into memory..."
        try:
            for p in paths:
                model, tok, chars, max_len = load_model(p, torch.device("cpu"), local_files_only=False,
                                                        base_model=self.base_model)
                self.models.append((model, tok, chars, max_len))
                print(f"loaded {p}", flush=True)
            self.ready = True
            self.tag("Muraho.")  # warm-up
            self.status = "Model ready."
            print("MODEL READY", flush=True)
        except Exception as e:
            self.status = f"Model failed to load: {e}"
            print(self.status, flush=True)

    def tag(self, text):
        if not self.ready:
            raise RuntimeError(self.status)
        text = text[:MAX_CHARS]
        sents = []
        for para in text.splitlines():
            for chunk in SENT_END.split(para.strip()):
                words = pretokenize(chunk)
                if words:
                    sents.append({"words": words, "tags": None, "lang": "kin"})
        if not sents:
            return []
        with self.lock, torch.inference_mode():
            votes = []
            for model, tok, chars, max_len in self.models:
                chunks = encode_sentences(sents, tok, chars, max_len)
                votes.append(run_eval(model, chunks, sents, tok.pad_token_id, torch.device("cpu"), 16, False))
        out = []
        for i, s in enumerate(sents):
            cols = zip(*[v[i] for v in votes])
            tags = [Counter(c).most_common(1)[0][0] for c in cols]
            out.append([{"w": w, "t": t} for w, t in zip(s["words"], tags)])
        return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kinyarwanda POS Tagger</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--accent:#0f6e5f;--accent-ink:#fff;
--n:#2563eb;--n-bg:#e8efff;--v:#c2410c;--v-bg:#fff0e6;--m:#15803d;--m-bg:#e7f6ec;--p:#7c3aed;--p-bg:#f1eaff;
--f:#0e7490;--f-bg:#e3f5f8;--o:#57534e;--o-bg:#efedea;}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--panel:#1b1b1f;--ink:#ececf1;--muted:#9a9aa3;--line:#2c2c33;
--accent:#3fb59c;--accent-ink:#06120f;--n:#8fb0ff;--n-bg:#1d2740;--v:#ffab7a;--v-bg:#3a2316;--m:#7fd39b;--m-bg:#16301f;
--p:#c4a6ff;--p-bg:#2a2140;--f:#7fd4e4;--f-bg:#132f35;--o:#c9c4bd;--o-bg:#2a2826;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
textarea{width:100%;min-height:140px;resize:vertical;border:1px solid var(--line);border-radius:10px;padding:12px;
font:inherit;background:var(--bg);color:var(--ink)}textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
button{font:inherit;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:9px 14px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600}
button:disabled{opacity:.6;cursor:wait}.hint{color:var(--muted);font-size:13px}
.samples{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.samples button{font-size:13px;padding:5px 10px}
#status{margin:16px 2px 8px;color:var(--muted);font-size:14px;min-height:20px}
.sent{display:flex;flex-wrap:wrap;gap:6px 6px;padding:14px 0;border-bottom:1px dashed var(--line)}.sent:last-child{border-bottom:0}
.tok{display:flex;flex-direction:column;align-items:center;border-radius:9px;padding:5px 8px 4px;background:var(--c-bg);min-width:34px}
.tok .w{font-size:16px;color:var(--ink)}.tok .t{font-size:11px;font-weight:700;letter-spacing:.04em;color:var(--c)}
.g-n{--c:var(--n);--c-bg:var(--n-bg)}.g-v{--c:var(--v);--c-bg:var(--v-bg)}.g-m{--c:var(--m);--c-bg:var(--m-bg)}
.g-p{--c:var(--p);--c-bg:var(--p-bg)}.g-f{--c:var(--f);--c-bg:var(--f-bg)}.g-o{--c:var(--o);--c-bg:var(--o-bg)}
.counts{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}.counts span{font-size:13px;border-radius:999px;padding:3px 10px;background:var(--c-bg);color:var(--c);font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px;font-size:13px;margin:8px 0 0}
details{margin-top:16px}summary{cursor:pointer;font-weight:600}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px;margin-top:10px}
.legend div{display:flex;gap:8px;align-items:baseline;font-size:14px}.legend b{font-size:12px;border-radius:6px;padding:1px 6px;background:var(--c-bg);color:var(--c);min-width:52px;text-align:center}
.section{margin-top:16px}h2{font-size:16px;margin:0 0 6px}
</style></head><body><div class="wrap">
<h1>Kinyarwanda POS Tagger</h1>
<p class="sub">Paste Kinyarwanda text. Every word gets its part of speech (Universal Dependencies tags).</p>
<div class="card">
<textarea id="txt" spellcheck="false" placeholder="Andika cyangwa ushyiremo inyandiko y'Ikinyarwanda..."></textarea>
<div class="samples" id="samples"></div>
<div class="row"><button class="primary" id="go">Tag text</button><button id="clear">Clear</button>
<span class="hint">Ctrl + Enter also works</span></div>
</div>
<div id="status"></div>
<div class="card" id="out" hidden>
<div id="sents"></div>
<div class="counts" id="counts"></div>
<div class="section"><div class="row" style="margin-top:0;justify-content:space-between"><h2>Plain output</h2><button id="copy">Copy</button></div>
<pre id="plain"></pre></div>
</div>
<details><summary>What the tags mean</summary><div class="legend" id="legend"></div></details>
</div>
<script>
const TAGS={NOUN:["n","Noun","umuturage, igihugu"],PROPN:["n","Proper noun","Kigali, Rwanda"],VERB:["v","Verb","abarirwa, kubaho"],
AUX:["v","Auxiliary","marks tense or mood next to a verb"],ADJ:["m","Adjective","benshi, mwiza, kinini"],ADV:["m","Adverb","hafi, nyuma, ubu"],
PRON:["p","Pronoun","aho, uyu, cyo"],DET:["p","Determiner","u (u Rwanda)"],NUM:["m","Number","31, miliyoni"],
ADP:["f","Adposition","ku, mu, kuri"],CCONJ:["f","Coordinating conjunction","n, na, kandi"],SCONJ:["f","Subordinating conjunction","ko, ngo, ariko"],
PART:["f","Particle","negation and similar markers"],INTJ:["o","Interjection","yego, oya"],PUNCT:["o","Punctuation",". , ' ( )"],
SYM:["o","Symbol","%, $, Frw"],X:["o","Other","words with no real category"]};
const SAMPLES=["Mu barwayi bashya abenshi babonetse mu mujyi wa Kigali 59 naho 14 babonetse muri Rubavu. Abakirwaye bose ni 953 barimo icyenda barembye.",
"Byarangiye Apple ivanye uwo mukino muri App Store nubwo rukigeretse mu nkiko hagati y’izo sosiyete zombi. Urubanza rwazo ruzasubukurwa mu cyumweru gitaha.",
"Ni igitabo Padiri Amerika avuga ko yanditse nyuma yo kubona ko buri munsi amadini yiyongera ariko umusaruro uyavamo ntugaragarire mu mbuto abayarimo bera.",
"Rugwizangoga yavuze ko Ubald kuva icyo gihe ariwe wahindutse nk’umugabo mu rugo, akaba umujyanama wa nyina ndetse uko yakuraga akagenda arushaho kubyitwaramo neza."];
const $=id=>document.getElementById(id);
const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
SAMPLES.forEach((s,i)=>{const b=document.createElement("button");b.textContent="Sample "+(i+1);b.onclick=()=>{$("txt").value=s;run()};$("samples").appendChild(b)});
$("legend").innerHTML=Object.entries(TAGS).map(([k,[g,n,e]])=>`<div class="g-${g}"><b>${k}</b><span>${n}<span class="hint"> · ${esc(e)}</span></span></div>`).join("");
async function run(){const text=$("txt").value.trim();if(!text){$("status").textContent="Type or paste some Kinyarwanda text first.";return}
$("go").disabled=true;$("status").textContent="Tagging...";const t0=performance.now();
try{const r=await fetch("/api/tag",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);render(d.sentences);
const n=d.sentences.reduce((a,s)=>a+s.length,0);$("status").textContent=`${n} words in ${d.sentences.length} sentence${d.sentences.length>1?"s":""}, ${Math.round(performance.now()-t0)} ms`;}
catch(e){$("status").textContent="Something went wrong: "+e.message}finally{$("go").disabled=false}}
function render(sents){const cnt={};
$("sents").innerHTML=sents.map(s=>`<div class="sent">${s.map(({w,t})=>{cnt[t]=(cnt[t]||0)+1;const g=(TAGS[t]||["o"])[0];
return `<div class="tok g-${g}" title="${t}: ${esc((TAGS[t]||["","Other"])[1])}"><span class="w">${esc(w)}</span><span class="t">${t}</span></div>`}).join("")}</div>`).join("");
$("counts").innerHTML=Object.entries(cnt).sort((a,b)=>b[1]-a[1]).map(([t,n])=>`<span class="g-${(TAGS[t]||["o"])[0]}">${t} ${n}</span>`).join("");
$("plain").textContent=sents.map(s=>s.map(({w,t})=>`${w}_${t}`).join(" ")).join("\n");$("out").hidden=false}
async function poll(){try{const d=await (await fetch("/health")).json();$("go").disabled=!d.ready;
if(!d.ready){$("status").textContent=d.status;setTimeout(poll,4000)}else if($("status").textContent.startsWith("The model")||$("status").textContent.startsWith("Loading")){$("status").textContent="Model ready. Paste text and press Tag text."}}
catch(e){$("status").textContent="Server not reachable.";setTimeout(poll,4000)}}poll();
$("go").onclick=run;$("clear").onclick=()=>{$("txt").value="";$("out").hidden=true;$("status").textContent="";$("txt").focus()};
$("txt").addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.ctrlKey||e.metaKey)){e.preventDefault();run()}});
$("copy").onclick=async()=>{try{await navigator.clipboard.writeText($("plain").textContent);$("copy").textContent="Copied";setTimeout(()=>$("copy").textContent="Copy",1200)}catch(e){}};
</script></body></html>"""


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, json.dumps({"ready": engine.ready, "status": engine.status}), "application/json")
            else:
                self._send(404, json.dumps({"error": "not found"}), "application/json")

        def do_POST(self):
            if self.path != "/api/tag":
                return self._send(404, json.dumps({"error": "not found"}), "application/json")
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                try:
                    body = raw.decode("utf-8")
                except UnicodeDecodeError:  # e.g. Windows clients sending cp1252
                    body = raw.decode("cp1252", errors="replace")
                text = json.loads(body).get("text", "")
                t0 = time.time()
                sents = engine.tag(text)
                self._send(200, json.dumps({"sentences": sents, "ms": round(1000 * (time.time() - t0))},
                                           ensure_ascii=False), "application/json; charset=utf-8")
            except Exception as e:  # report instead of dropping the connection
                self._send(500, json.dumps({"error": str(e)}), "application/json")

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base_model", default="Davlan/afro-xlmr-large-76L",
                    help="backbone name for config + tokenizer (weights come from the checkpoint)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    engine = Engine(args.models, args.base_model)
    print(f"SERVING http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
