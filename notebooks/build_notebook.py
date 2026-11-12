"""Generate notebooks/Kinyarwanda_POS_Tagger.ipynb from models.json (model list + accuracy).

python notebooks/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "mosesmmoisebidth/kinyarwanda-pos-tagger"
man = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
models = man["models"]
by_key = {(m["recipe"], m["model"]): m for m in models}
best = man["best"]
RECIPE_TXT = {"full": "full recipe", "plain": "plain fine-tune"}

SAMPLE = ("Mu barwayi bashya abenshi babonetse mu mujyi wa Kigali 59 naho 14 babonetse muri Rubavu. "
          "Abakirwaye bose ni 953 barimo icyenda barembye.")
SAMPLE2 = ("Rugwizangoga yavuze ko Ubald kuva icyo gihe ariwe wahindutse nk'umugabo mu rugo, akaba umujyanama "
           "wa nyina ndetse uko yakuraga akagenda arushaho kubyitwaramo neza.")


def option(m):
    s = f"{m['id']} | {m['label']}, {m['family']}, {RECIPE_TXT[m['recipe']]}, {m['mean_test']:.2f}% test"
    return s + " (best)" if m["id"] == best else s


options = [option(m) for m in sorted(models, key=lambda m: (m["recipe"] != "full", -m["mean_test"]))]
pairs = []
for g, a, title in man["pairs"]:
    if any(m["model"] == g for m in models) and any(m["model"] == a for m in models):
        pairs.append((f"{title.split(' (')[0]}: {by_key.get(('full', g), by_key.get(('plain', g)))['label']} vs "
                      f"{by_key.get(('full', a), by_key.get(('plain', a)))['label']}", g, a))
recipes = [r for r in ("full", "plain") if any(m["recipe"] == r for m in models)]


def table():
    rows = ["| Model | Type | Size | Recipe | Test accuracy (mean of 5 seeds) | Unseen words |", "|---|---|---|---|---|---|"]
    for m in sorted(models, key=lambda m: (m["recipe"] != "full", -m["mean_test"])):
        name = f"**{m['label']}**" if m["id"] == best else m["label"]
        rows.append(f"| {name} | {m['family']} | {m['params']} | {RECIPE_TXT[m['recipe']]} | "
                    f"{m['mean_test']:.2f} ± {m['std_test']:.2f} | {m['mean_oov']:.2f} |")
    return "\n".join(rows)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


q = lambda xs: "[" + ", ".join(json.dumps(x) for x in xs) + "]"
cells = [
    md(f"""
# Kinyarwanda Part-of-Speech Tagger

Give it Kinyarwanda text and every word gets its part of speech (the 17 [Universal Dependencies](https://universaldependencies.org/u/pos/) tags: NOUN, VERB, PRON, ADP...).
The models were trained on [MasakhaPOS](https://github.com/masakhane-io/masakhane-pos). This notebook also compares a **general**
multilingual model with an **Afro-centric** one, so you can see which is more accurate and where.

### How to use
1. Optional: **Runtime > Change runtime type > T4 GPU** (faster, but CPU works too).
2. **Runtime > Run all**, or run the cells one by one with **Shift + Enter**.
3. Change the text and the model in the forms on the right of each cell, then run the cell again.

Code, results and details: [github.com/{REPO}](https://github.com/{REPO})

### Models (test set: 604 sentences, about 17k words)
{table()}
"""),
    code(f"""
#@title 1. Setup (gets the code, about 30 seconds)
import os, sys, subprocess, importlib
REPO_URL = "https://github.com/{REPO}"
REPO = "/content/kinyarwanda-pos-tagger"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "sentencepiece"], check=True)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import kinpos_infer as K
importlib.reload(K)
print("Ready. Running on", "GPU" if K.DEVICE.type == "cuda" else "CPU", "|", len(K.MANIFEST["models"]), "trained models available")
"""),
    code(f"""
#@title 2. Choose a model (downloads it the first time, 0.3 to 1.1 GB)
MODEL = {json.dumps(options[0])} #@param {q(options)}
MODEL_ID = MODEL.split("|")[0].strip()
m = K.BY_ID[MODEL_ID]
K.get(MODEL_ID)
print(f"Loaded {{m['label']}} ({{m['family']}}, {{m['params']}} parameters, {{m['recipe']}} recipe).")
print(f"Test accuracy: {{m['mean_test']:.2f}}% on average over 5 training runs; this checkpoint {{m['test']:.2f}}%.")
"""),
    code(f"""
#@title 3. Tag Kinyarwanda text
TEXT = {json.dumps(SAMPLE)} #@param {{type:"string"}}
result = K.tag(MODEL_ID, TEXT)
K.show(result, title=K.BY_ID[MODEL_ID]["label"])
print(K.as_text(result))
"""),
    code(f"""
#@title 4. Compare the two models of a pair on your text
PAIR = {json.dumps(pairs[0][0])} #@param {q([p[0] for p in pairs])}
RECIPE = {json.dumps(recipes[0])} #@param {q(recipes)}
TEXT = {json.dumps(SAMPLE2)} #@param {{type:"string"}}
general, afro = {json.dumps({p[0]: [p[1], p[2]] for p in pairs})}[PAIR]
K.show_compare(f"{{RECIPE}}_{{general}}", f"{{RECIPE}}_{{afro}}", TEXT);
"""),
    code("""
#@title 5. Tag a whole text file (upload a .txt, get the tagged file back)
FORMAT = "conll" #@param ["conll", "inline"]
from google.colab import files
uploaded = files.upload()
for name in uploaded:
    out = os.path.splitext(name)[0] + ("_pos.conll" if FORMAT == "conll" else "_pos.txt")
    n = K.tag_file(MODEL_ID, name, out, FORMAT)
    print(f"{name}: {n} sentences tagged with {K.BY_ID[MODEL_ID]['label']} -> {out}")
    files.download(out)
"""),
    code("""
#@title 6. Check the accuracy yourself on the MasakhaPOS test set (1 to 3 minutes)
K.evaluate_test(MODEL_ID);
"""),
    code("""
#@title 7. Open the full web app: switch models, compare pairs, browse test-set errors
WHICH = "full recipe (4 models)" #@param ["full recipe (4 models)", "all 8 models", "only the selected model"]
import time, urllib.request, torch
ids = {"full recipe (4 models)": [x["id"] for x in K.MANIFEST["models"] if x["recipe"] == "full"],
       "all 8 models": [x["id"] for x in K.MANIFEST["models"]],
       "only the selected model": [MODEL_ID]}[WHICH]
for mid in ids:
    K.download(mid)
K._cache.clear()                     # free memory for the app
if torch.cuda.is_available():
    torch.cuda.empty_cache()
subprocess.run("pkill -f app/server.py", shell=True)
subprocess.Popen([sys.executable, "app/server.py", "--device", "auto", "--max_loaded", "2", "--port", "8790"],
                 stdout=open("/content/app.log", "w"), stderr=subprocess.STDOUT, cwd=REPO)
for _ in range(90):
    try:
        urllib.request.urlopen("http://127.0.0.1:8790/api/state", timeout=2); break
    except Exception:
        time.sleep(1)
from google.colab import output
output.serve_kernel_port_as_window(8790, anchor_text="Open the Kinyarwanda POS Tagger app in a new tab")
output.serve_kernel_port_as_iframe(8790, height=900)
"""),
    md(f"""
### Train the models yourself
Everything is reproducible on a free Colab T4 (about 4 hours for all 44 runs; results are saved to your Drive and the
queue resumes after a disconnect). See **Reproduce** in the [README](https://github.com/{REPO}#reproduce).

### Credits
Data: MasakhaPOS (Dione et al., 2023), CC BY-NC 4.0, non-commercial use. Pretrained models: XLM-RoBERTa (Meta AI),
AfroXLMR (Alabi et al., 2022), AfriBERTa (Ogueji et al., 2021), mBERT (Google).
"""),
]
nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "name": "Kinyarwanda_POS_Tagger.ipynb"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "notebooks", "Kinyarwanda_POS_Tagger.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out, "with", len(options), "model options")
