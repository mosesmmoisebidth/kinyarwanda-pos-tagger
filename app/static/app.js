"use strict";
const TAGS = {
  NOUN: ["n", "Noun", "umuturage, igihugu"], PROPN: ["n", "Proper noun", "Kigali, Rwanda"],
  VERB: ["v", "Verb", "abarirwa, kubaho"], AUX: ["v", "Auxiliary", "marks tense or mood next to a verb"],
  ADJ: ["m", "Adjective", "benshi, mwiza, kinini"], ADV: ["m", "Adverb", "hafi, nyuma, ubu"],
  NUM: ["m", "Number", "31, miliyoni"], PRON: ["p", "Pronoun", "aho, uyu, cyo"], DET: ["p", "Determiner", "u (u Rwanda)"],
  ADP: ["f", "Adposition", "ku, mu, kuri"], CCONJ: ["f", "Coordinating conjunction", "n, na, kandi"],
  SCONJ: ["f", "Subordinating conjunction", "ko, ngo, ariko"], PART: ["f", "Particle", "negation and similar markers"],
  INTJ: ["o", "Interjection", "yego, oya"], PUNCT: ["o", "Punctuation", ". , ' ( )"], SYM: ["o", "Symbol", "%, $, Frw"],
  X: ["o", "Other", "words with no real category"],
};
const SAMPLES = [
  "Mu barwayi bashya abenshi babonetse mu mujyi wa Kigali 59 naho 14 babonetse muri Rubavu. Abakirwaye bose ni 953 barimo icyenda barembye.",
  "Byarangiye Apple ivanye uwo mukino muri App Store nubwo rukigeretse mu nkiko hagati y'izo sosiyete zombi. Urubanza rwazo ruzasubukurwa mu cyumweru gitaha.",
  "Ni igitabo Padiri Amerika avuga ko yanditse nyuma yo kubona ko buri munsi amadini yiyongera ariko umusaruro uyavamo ntugaragarire mu mbuto abayarimo bera.",
  "Rugwizangoga yavuze ko Ubald kuva icyo gihe ariwe wahindutse nk'umugabo mu rugo, akaba umujyanama wa nyina ndetse uko yakuraga akagenda arushaho kubyitwaramo neza.",
];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const grp = (t) => (TAGS[t] || ["o"])[0];
const fmt = (x, d = 2) => (x == null ? "n/a" : Number(x).toFixed(d));
const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } }, set(k, v) { try { localStorage.setItem(k, v); } catch {} } };

let state = null, results = null, disagreements = null;
let recipe = store.get("recipe", "full"), selected = store.get("model", "");

function toast(msg) { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), 3200); }
const byId = (id) => state.models.find((m) => m.id === id);

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => (s.hidden = s.id !== "tab-" + name));
  store.set("tab", name);
  if (name === "browse") loadBrowse();
}

/* ---------------- model picker ---------------- */
function recipesAvailable() { return Object.keys(state.recipes).filter((r) => state.models.some((m) => m.recipe === r)); }
function renderRecipes() {
  const avail = recipesAvailable();
  if (!avail.includes(recipe)) recipe = avail[0] || "full";
  $("recipeSeg").innerHTML = avail.map((r) =>
    `<button aria-pressed="${r === recipe}" data-r="${r}" title="${esc(state.recipes[r])}">${r === "full" ? "Full recipe" : "Plain fine-tune"}</button>`).join("");
  $("recipeSeg").querySelectorAll("button").forEach((b) => (b.onclick = () => { recipe = b.dataset.r; store.set("recipe", recipe); renderModels(); renderRecipes(); }));
}
function renderModels() {
  const list = state.models.filter((m) => m.recipe === recipe);
  if (!list.length) { $("models").innerHTML = `<p class="empty">No trained models found yet. Put checkpoints in models/&lt;recipe&gt;_&lt;model&gt;/model.pt.</p>`; return; }
  if (!list.some((m) => m.id === selected)) selected = (list.find((m) => m.family === "Afro-centric") || list[0]).id;
  const best = Math.max(...list.map((m) => m.mean_test ?? m.ckpt_test ?? 0));
  $("models").innerHTML = list.map((m) => {
    const acc = m.mean_test ?? m.ckpt_test;
    return `<button class="model" aria-pressed="${m.id === selected}" data-id="${m.id}">
      <span class="name">${esc(m.label)}</span>
      <span class="meta"><span class="badge ${m.family === "Afro-centric" ? "afro" : "gen"}">${esc(m.family)}</span>${esc(m.params)} params
        ${(acc ?? 0) === best ? '<span class="badge best">Best</span>' : ""}${m.loaded ? '<span title="In memory, answers fast"><span class="dot"></span></span>' : ""}</span>
      <span class="acc">${fmt(acc)}%<small>test acc${m.std_test != null ? " ± " + fmt(m.std_test) : ""}</small></span>
      <span class="meta">${esc(m.note)}</span></button>`;
  }).join("");
  $("models").querySelectorAll(".model").forEach((b) => (b.onclick = () => { selected = b.dataset.id; store.set("model", selected); renderModels(); }));
  const p = byId(selected)?.partner;
  $("compare").disabled = !p;
  $("compare").parentElement.title = p ? "Partner: " + byId(p).label : "This model's partner has no checkpoint yet";
}

/* ---------------- tagging ---------------- */
SAMPLES.forEach((s, i) => { const b = document.createElement("button"); b.textContent = "Sample " + (i + 1); b.onclick = () => { $("txt").value = s; run(); }; $("samples").appendChild(b); });
$("legend").innerHTML = Object.entries(TAGS).map(([k, [g, n, e]]) => `<div class="g-${g}"><b>${k}</b><span>${n}<span class="hint"> · ${esc(e)}</span></span></div>`).join("");

async function run() {
  const text = $("txt").value.trim();
  if (!text) { $("status").textContent = "Type or paste some Kinyarwanda text first."; return; }
  const m = byId(selected); if (!m) return;
  const compare = $("compare").checked && !!m.partner;
  $("go").disabled = true;
  const needLoad = !m.loaded || (compare && !byId(m.partner).loaded);
  $("status").textContent = needLoad ? `Loading ${compare ? "both models" : m.label} into memory (first time can take 10 to 40 seconds)...` : "Tagging...";
  const t0 = performance.now();
  try {
    const r = await fetch("/api/tag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, model: m.id, compare }) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    render(d);
    const n = d.sentences.reduce((a, s) => a + s.length, 0);
    $("status").textContent = `${n} words in ${d.sentences.length} sentence${d.sentences.length > 1 ? "s" : ""}, ${Math.round(performance.now() - t0)} ms`;
    refreshState();
  } catch (e) { $("status").textContent = "Something went wrong: " + e.message; }
  finally { $("go").disabled = false; }
}
function render(d) {
  const A = byId(d.model), cnt = {};
  let diffs = 0;
  if (d.partner) {
    const B = byId(d.partner);
    const [gen, afro, gS, aS] = A.family === "General" ? [A, B, d.sentences, d.partner_sentences] : [B, A, d.partner_sentences, d.sentences];
    $("sents").innerHTML = gS.map((s, i) => `<div class="sent">${s.map(([w, tg], j) => {
      const ta = aS[i][j][1]; const diff = tg !== ta; diffs += diff; cnt[ta] = (cnt[ta] || 0) + 1;
      return `<div class="stack${diff ? " diff" : ""}"><span class="w">${esc(w)}</span><span class="t g-${grp(tg)}" title="${esc(gen.label)}">${tg}</span><span class="t g-${grp(ta)}" title="${esc(afro.label)}">${ta}</span></div>`;
    }).join("")}</div>`).join("");
    $("outHead").innerHTML = `<span>Top tag: <b>${esc(gen.label)}</b> (general)</span><span>Bottom tag: <b>${esc(afro.label)}</b> (Afro-centric)</span><span><b>${diffs}</b> word${diffs === 1 ? "" : "s"} tagged differently (outlined)</span>`;
    $("plain").textContent = aS.map((s) => s.map(([w, t]) => `${w}_${t}`).join(" ")).join("\n");
  } else {
    $("sents").innerHTML = d.sentences.map((s) => `<div class="sent">${s.map(([w, t]) => { cnt[t] = (cnt[t] || 0) + 1;
      return `<div class="tok g-${grp(t)}" title="${t}: ${esc((TAGS[t] || ["", "Other"])[1])}"><span class="w">${esc(w)}</span><span class="t">${t}</span></div>`; }).join("")}</div>`).join("");
    $("outHead").innerHTML = `<span>Model: <b>${esc(A.label)}</b> (${esc(A.family)}, ${A.recipe === "full" ? "full recipe" : "plain fine-tune"})</span>`;
    $("plain").textContent = d.sentences.map((s) => s.map(([w, t]) => `${w}_${t}`).join(" ")).join("\n");
  }
  $("counts").innerHTML = Object.entries(cnt).sort((a, b) => b[1] - a[1]).map(([t, n]) => `<span class="g-${grp(t)}">${t} ${n}</span>`).join("");
  $("out").hidden = false;
}
$("go").onclick = run;
$("clear").onclick = () => { $("txt").value = ""; $("out").hidden = true; $("status").textContent = ""; $("txt").focus(); };
$("txt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } });
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("plain").textContent); toast("Copied"); } catch { toast("Copy failed"); } };
$("compare").checked = store.get("compare", "0") === "1";
$("compare").onchange = () => store.set("compare", $("compare").checked ? "1" : "0");

/* ---------------- results ---------------- */
function label(key) { return (state.labels[key] || {}).label || key; }
function renderResults() {
  const { summary, pairs } = results;
  if (!Object.keys(summary).length) { $("verdicts").innerHTML = `<div class="card empty">Results will appear here after training finishes.</div>`; return; }
  $("verdicts").className = "verdicts";
  $("verdicts").innerHTML = pairs.map((p) => {
    const g = p.general_test[0], a = p.afro_test[0], afroWins = p.gap_afro_minus_general > 0;
    const winner = afroWins ? label(p.afro) : label(p.general);
    const sig = p.significant ? "a clear, statistically significant difference" : "too small to be sure it is real";
    return `<div class="card verdict"><h3>${esc(p.title)} · ${p.recipe === "full" ? "full recipe" : "plain fine-tune"}</h3>
      <div class="big">${p.significant ? `${esc(winner)} is more accurate by ${fmt(Math.abs(p.gap_afro_minus_general))} points`
        : `${esc(winner)} is ahead by only ${fmt(Math.abs(p.gap_afro_minus_general))} points, within run-to-run noise`}</div>
      <div class="vs"><div class="side${afroWins ? "" : " win"}"><div class="n">${esc(label(p.general))} · general</div><div class="a">${fmt(g)}%</div></div>
        <span class="hint">vs</span><div class="side${afroWins ? " win" : ""}"><div class="n">${esc(label(p.afro))} · Afro-centric</div><div class="a">${fmt(a)}%</div></div></div>
      <ul><li>95% confidence interval of the gap: ${fmt(p.ci95[0])} to ${fmt(p.ci95[1])} points, ${sig} (p = ${fmt(p.p_bootstrap, 4)}).</li>
        <li>Unseen words: ${fmt(p.general_oov[0])}% vs ${fmt(p.afro_oov[0])}% (gap ${p.oov_gap > 0 ? "+" : ""}${fmt(p.oov_gap)}).</li>
        <li>Words only the Afro-centric model gets right: ${p.tokens_only_afro_right}; only the general model: ${p.tokens_only_general_right}.</li></ul></div>`;
  }).join("");
  const rows = [];
  for (const r of Object.keys(state.recipes)) {
    const list = Object.values(summary).filter((s) => s.recipe === r).sort((x, y) => y.test_acc[0] - x.test_acc[0]);
    if (!list.length) continue;
    rows.push(`<tr class="group"><td colspan="7">${esc(state.recipes[r])}</td></tr>`);
    list.forEach((s) => rows.push(`<tr><td><b>${esc(s.label)}</b></td><td><span class="badge ${s.family === "Afro-centric" ? "afro" : "gen"}">${esc(s.family)}</span></td>
      <td class="num">${esc(s.params)}</td><td class="num"><b>${fmt(s.test_acc[0])}</b> ± ${fmt(s.test_acc[1])}</td><td class="num">${fmt(s.test_oov[0])}</td>
      <td class="num">${fmt(s.dev_acc[0])}</td><td class="num">${s.seeds.length}</td></tr>`));
  }
  $("board").innerHTML = `<thead><tr><th>Model</th><th>Type</th><th>Size</th><th>Test acc %</th><th>OOV acc %</th><th>Dev acc %</th><th>Seeds</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  $("tagPair").innerHTML = pairs.map((p, i) => `<option value="${i}">${esc(label(p.general))} vs ${esc(label(p.afro))} · ${p.recipe}</option>`).join("");
  $("tagPair").onchange = renderBars; renderBars();
}
function renderBars() {
  const p = results.pairs[+$("tagPair").value || 0]; if (!p) return;
  const entries = Object.entries(p.per_tag_gap);
  const max = Math.max(1, ...entries.map(([, v]) => Math.abs(v)));
  $("tagBars").innerHTML = entries.map(([t, v]) => `<div class="bar"><b class="g-${grp(t)}" style="color:var(--c)">${t}</b>
    <div class="track"><span class="mid"></span><span class="fill ${v >= 0 ? "pos" : "neg"}" style="width:${(50 * Math.abs(v)) / max}%"></span></div>
    <span class="v">${v > 0 ? "+" : ""}${fmt(v)}</span></div>`).join("") +
    `<div class="stack-legend"><span style="color:var(--afro)">Afro-centric better</span><span style="color:var(--gen)">General better</span></div>`;
}

/* ---------------- test-set browser ---------------- */
let shown = 0;
async function loadBrowse() {
  if (!disagreements) {
    try { disagreements = await (await fetch("/api/disagreements")).json(); } catch { disagreements = []; }
    const keys = [...new Set(disagreements.map((d) => d.recipe + "|" + d.pair))];
    $("bPair").hidden = !keys.length;
    $("bPair").innerHTML = keys.map((k) => { const [r, a, b] = k.split("|"); return `<option value="${k}">${esc(label(a))} vs ${esc(label(b))} · ${r === "full" ? "full recipe" : "plain"}</option>`; }).join("");
    $("bPair").onchange = $("bFilter").onchange = () => { shown = 0; $("bList").innerHTML = ""; more(); };
    $("bMore").onclick = more;
    more();
  }
}
function filtered() {
  const [r, a, b] = ($("bPair").value || "").split("|");
  const f = $("bFilter").value;
  return disagreements.filter((d) => d.recipe === r && d.pair === `${a}|${b}`)
    .filter((d) => f === "all" || (f === "afro" && d.b_err < d.a_err) || (f === "general" && d.a_err < d.b_err) || (f === "tie" && d.a_err === d.b_err));
}
function more() {
  const list = filtered();
  if (!list.length) { $("bList").innerHTML = `<div class="card empty">Nothing to show yet.</div>`; $("bMore").hidden = true; $("bCount").textContent = ""; return; }
  const [, a, b] = $("bPair").value.split("|");
  $("bCount").textContent = `${list.length} sentence${list.length === 1 ? "" : "s"}`;
  const chunk = list.slice(shown, shown + 15);
  $("bList").insertAdjacentHTML("beforeend", chunk.map((d) => `<div class="card bsent">
    <div class="head"><span>Sentence ${d.sid + 1}</span><span>Errors: <b>${esc(label(a))}</b> ${d.a_err} · <b>${esc(label(b))}</b> ${d.b_err}</span></div>
    <div class="stack-legend"><span style="color:var(--muted)">Gold</span><span style="color:var(--gen)">${esc(label(a))}</span><span style="color:var(--afro)">${esc(label(b))}</span></div>
    <div class="toks" style="margin-top:8px">${d.words.map((w, j) => { const g = d.gold[j], x = d.a[j], y = d.b[j];
      return `<div class="stack${x !== y ? " diff" : ""}"><span class="w">${esc(w)}</span><span class="t g-${grp(g)}">${g}</span>
        <span class="t g-${grp(x)}${x !== g ? " wrong" : ""}">${x}</span><span class="t g-${grp(y)}${y !== g ? " wrong" : ""}">${y}</span></div>`; }).join("")}</div></div>`).join(""));
  shown += chunk.length;
  $("bMore").hidden = shown >= list.length;
}

/* ---------------- boot ---------------- */
async function refreshState() {
  try { state = await (await fetch("/api/state")).json(); renderRecipes(); renderModels(); }
  catch { $("status").textContent = "Server not reachable. Start it with: python app/server.py"; }
}
(async function boot() {
  await refreshState();
  try { results = await (await fetch("/api/results")).json(); renderResults(); } catch {}
  showTab(store.get("tab", "tag"));
})();
