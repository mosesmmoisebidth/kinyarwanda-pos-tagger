"""Print queue progress: finished jobs with dev/test accuracy, failures, and the live log tail."""
import glob
import json
import os
import subprocess

OUT = "/content/drive/MyDrive/kinpos_pairs"
st = json.load(open(os.path.join(OUT, "status.json"))) if os.path.isfile(os.path.join(OUT, "status.json")) else {}
done = [d for d in st.get("done", []) if not d.get("skipped")]
print(f"total {st.get('total')}  done {len(st.get('done', []))} (new {len(done)})  failed {len(st.get('failed', []))}  "
      f"started {st.get('started')}  finished {st.get('finished')}")
for d in done[-12:]:
    print(" ", d["job"], f"{d['min']}min", "dev", d.get("dev"), "test", d.get("test"), "oov", d.get("test_oov"))
for f in st.get("failed", [])[-3:]:
    print("FAILED", f["job"], f.get("rc"), (f.get("tail") or f.get("why", ""))[-300:])
cur = st.get("current")
if cur:
    log = os.path.join(OUT, "logs", cur["job"].replace("/", "_") + ".log")
    print("NOW", cur["job"], "since", cur["since"])
    if os.path.isfile(log):
        lines = [l for l in open(log, encoding="utf-8", errors="replace").read().splitlines() if l.strip()]
        print("\n".join("   " + l[:200] for l in lines[-4:]))
alive = subprocess.run("pgrep -f run_queue.py", shell=True, capture_output=True, text=True).stdout.split()
print("queue process alive:", bool(alive))
print(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip())
