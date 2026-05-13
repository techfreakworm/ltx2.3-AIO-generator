# SKILLS.md — how to make changes in this project

Process rules and habits for agents working on this repo. Sits alongside:

- `AGENTS.md` — the tool-agnostic rulebook (locked decisions, out-of-scope list, commit + verification rules).
- `CLAUDE.md` — Claude-specific extensions + full gotchas catalogue (*what & why*).
- `README.md` — public-facing intro (different audience).

This file is the *how* — debugging patterns, verification habits, deployment workflow, useful one-liners.

> **Default rule when in doubt:** stop and ask the user. The user prefers a question over wrong work.

---

## Investigation before fix

### Reproduce the bug visually before patching CSS / UI

When the user reports a layout, color, click, or visibility issue, **the first action is Playwright + screenshot, not code**. The user has called this out explicitly:

> "Make sure to check playwright with screenshot to verify issues before making fix."

Skipping the visual repro twice in a row produced patches that addressed a different symptom than what the user was seeing. Reproduce, then fix, then re-screenshot to verify the fix.

**Tools:** local dev server (port 7860, see "Running locally" below) + `mcp__playwright__browser_*` tools. Resize to the affected viewport (typically 380 px / 900 px / 1280 px). `browser_evaluate` is the most reliable way to inspect DOM state — getBoundingClientRect, getComputedStyle, elementFromPoint.

### Pull HF Space logs first when something runs there

For Spaces failures, the run logs are the source of truth. Pull and search:

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/spaces/techfreakworm/LTX2.3-Studio/logs/run" \
  -o /tmp/hf_run.log

# Find last submit and tail from there
python3 << 'PY'
import json
events = []
for line in open('/tmp/hf_run.log'):
    line = line.strip()
    if line.startswith('data: '):
        try: events.append(json.loads(line[6:]))
        except Exception: pass
last = max(i for i, e in enumerate(events) if 'submitting workflow' in e.get('data', ''))
for ev in events[last:]:
    print(ev.get('timestamp', '')[:19], ev.get('data', '').rstrip()[:240])
PY
```

`/logs/build` is the other endpoint. Build logs show preload, image-build, pip; run logs show container output.

### Stage check before action

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/spaces/techfreakworm/LTX2.3-Studio" | jq -r '.runtime'
```

Stages: `BUILDING` (image), `APP_STARTING` (boot), `RUNNING`, `RUNTIME_ERROR`, `RUNNING_BUILDING` (live serving + new build queued). If `RUNTIME_ERROR` is non-null, that's your headline.

### Sequential thinking for repeated failures

The user has called this out:

> "On 2nd failed fix, stop patching; use sequential-thinking MCP + brainstorming skill"

If your first fix didn't land, **stop patching**. Use `mcp__sequential-thinking__sequentialthinking` to think through the failure mode end-to-end, plus web search for canonical solutions. Do not loop on speculative one-line patches.

### Web-search for HF / Gradio errors with the literal message

HF docs change. The `Spaces Configuration Reference` and `Spaces ZeroGPU` pages often have undocumented behavior captured in forum threads. When you hit a Gradio/Spaces error, web-search the literal exception message. Examples that paid off:

- `gradio.exceptions.InvalidPathError` → fix was `allowed_paths=` (Gradio 5 file-access policy)
- `'Workload evicted, storage limit exceeded (150G)'` → 150 GB ephemeral cap
- `'No @spaces.GPU function detected during startup'` → must be module-level decorator
- `'GPU task aborted'` → `@spaces.GPU(duration=...)` cap

---

## Verification

### Run the full repro in Playwright before declaring done

After a UI fix, re-run the same Playwright sequence that exposed the bug. Take a screenshot. Read the DOM state. Don't trust "it should work now" — show that it does.

### Local before push

When iterating on app behavior, the local dev server gives instant feedback. The user explicitly asks for this — they do most testing on the WiFi-accessible local URL. **Never push during HF testing windows.** When the user is testing on the live Space, hold local commits until they say push.

```bash
# In repo root
source .venv/bin/activate
python app.py  # or background it; see "Running locally"
```

The user has stated:

> "DO NOT PUSH since testing is happening on HF"

When in doubt, hold and ask.

### Smoke import + build_app after backend/app changes

```bash
python -c "import app; b = app.build_app(); print(type(b).__name__)"
```

Should print `Blocks`. Catches most syntax / import-cycle issues without spinning up the full server.

### Sanity-test isolated functions when changing logic

For workflow walkers, model registry, duration estimators — write a tiny `python3 -c '...'` or HEREDOC to feed synthetic inputs and verify outputs. Faster than running the full app, catches regressions that the full app would mask.

---

## Running locally

### Standard launch (port 7860)

```bash
cd /Users/techfreakworm/Projects/llm/ltx2.3-AIO-generator
source .venv/bin/activate
nohup python app.py > /tmp/ltx_studio_run.log 2>&1 &
echo $! > /tmp/ltx_studio.pid
```

Wait ~18 seconds for ComfyUI to import + Gradio to bind, then check:

```bash
lsof -nP -iTCP:7860 -sTCP:LISTEN
```

### LAN-accessible URL

Bound to `0.0.0.0:7860` by default. Get the LAN IP:

```bash
ipconfig getifaddr en0 || ipconfig getifaddr en1
```

Open `http://<LAN_IP>:7860` on phone/tablet on the same WiFi. macOS firewall: allow inbound for `python` if connection refused.

### Stop

```bash
PID=$(cat /tmp/ltx_studio.pid)
kill -9 $PID
lsof -nP -iTCP:7860 -sTCP:LISTEN | awk 'NR>1 {print $2}' | xargs -r kill -9
```

---

## Pushing changes

### Two remotes

```bash
git push origin master           # GitHub:  techfreakworm/ltx2.3-AIO-generator
git push space  master:main      # HF Space: techfreakworm/LTX2.3-Studio (deploys from main)
```

The repo has both remotes pre-configured (`origin` + `space`). HF credentials live in `~/.cache/huggingface/token`; git's credential helper picks them up automatically — no need to embed the token in the URL.

> ⚠ **Refspec matters for the Space push.** Local default branch is `master`; the HF Space deploys from `main`. A bare `git push space master` succeeds but creates an orphan `refs/heads/master` on the remote that does NOT trigger a deploy — the Space silently stays on the old build. Always push with the `master:main` refspec form.

If unsure, verify with `git ls-remote space` — `HEAD` should point at `refs/heads/main`.

### When to push

- Default: hold all commits locally, ask the user before pushing.
- The user usually says "push" or "push them" when ready.
- During the user's HF testing windows, NEVER push.
- After a successful local Playwright verification of a fix, summarize the queued commits and ask.

---

## Spaces deploy lifecycle

Each push triggers a Docker image rebuild. Most layers are cached unless requirements.txt or README YAML changes. The first push that adds/changes `preload_from_hub:` triggers a long preload step (download all listed files into `~/.cache/huggingface/hub`).

Container start sequence (after image push):
1. HF brings up the container as user 1000
2. Our `_bootstrap()` runs:
   - clones ComfyUI + custom nodes (cold-start only — frozen ZeroGPU containers retain them)
   - pip installs each custom node's requirements
   - `_mirror_preload_hf_cache()` builds writable cache mirror
   - copies seed inputs
   - sets HF_HOME / HF_HUB_CACHE env vars
3. `gr.Blocks(...).launch()` binds 7860
4. Stage transitions to `RUNNING`

ZeroGPU container freeze on idle: keeps `~/comfyui`, `~/hf-cache-rw`, etc. Wake on next request restores in seconds. Push or rebuild loses everything.

---

## When the user says "deep think"

The user explicitly invokes deeper investigation when stuck:

> "Use deep thinking using sequential thinking and web search and code exploration."

Use `mcp__sequential-thinking__sequentialthinking` to lay out the problem end-to-end. Web-search literal error messages. Read code beyond the immediate failure site. Avoid speculative one-line patches when in this mode.

---

## What never to do

- **Push without explicit permission** during HF test windows.
- **Add Co-Authored-By** or any agent attribution to commit messages.
- **Hand-edit `workflows/*.json`** — the user re-exports from ComfyUI editor.
- **`chmod` the HF preload cache** — we don't own it. See cache-mirror approach in CLAUDE.md.
- **Switch `sdk: gradio` → `sdk: docker`** in README. Loses ZeroGPU.
- **Move models into the repo via git LFS without asking.** Pro has 1 TB LFS but bandwidth is finite.
- **Implement out-of-scope v1.1+ features** without asking. See "Out of scope" in CLAUDE.md.
- **Eagerly load models at module import.** `_bootstrap()` only ensures clones + cache mirroring. Model load happens when ComfyUI's executor evaluates a node.

---

## Memory (cross-session)

The user's preferences live at `~/.claude/projects/-Users-techfreakworm-Projects/memory/`. Key entries:

- **Git authorship:** sole author, no co-author footers
- **Verify before fix:** Playwright + screenshot first
- **Don't push during HF testing:** hold local commits
- **Autonomous execution:** prefer scripts over notebooks, report results
- **No conda:** `python3.11 -m venv`, brew for system bins
- **Tests folder:** keep `~/Projects/tests/` separate from `~/Projects/`

When the user asks to remember something new, save it as a memory file and update `MEMORY.md` index.

---

## When stuck for too long

Three escalation steps:

1. **`mcp__sequential-thinking__sequentialthinking`** — think the whole flow through, identify the unknown.
2. **WebSearch + WebFetch** — find canonical fix or known issue.
3. **Ask the user** — describe what's been tried, what's still unknown, propose options.

Do not loop on patches when you've patched twice and it's still broken.

---

## Repo structure (high level)

```
.
├── app.py               # Gradio entry, _bootstrap, _on_generate, build_app
├── backend.py           # ComfyUILibraryBackend, _execute_workflow, _GPU
├── modes.py             # MODE_REGISTRY + per-mode parameterize_fn + node-id constants
├── models.py            # MODEL_REGISTRY, walk_workflow_for_models, ensure_models
├── ui.py                # render_status, _render_idle, mode-form layout primitives
├── workflow.py          # load_template, set_input
├── workflows/           # API-format mode JSONs (do not hand-edit)
│   ├── t2v.json
│   ├── i2v.json
│   ├── a2v.json
│   ├── lipsync.json
│   ├── keyframe.json
│   └── style.json
├── assets/seed_inputs/  # placeholder image/audio/video for cold-start (gitignored except this dir)
├── docs/
│   ├── superpowers/specs/    # design specs (per-feature)
│   ├── superpowers/plans/    # implementation plans (per-feature)
│   └── future_improvements.md
├── tools/extract_modes.py    # regenerate workflows/ from master
├── tests/
├── README.md            # HF Space YAML + project intro (public-facing)
├── AGENTS.md            # tool-agnostic agent rulebook (locked decisions, OoS)
├── CLAUDE.md            # what & why — full gotchas catalogue
├── SKILLS.md            # how — process, debugging, deployment (this file)
├── requirements.txt
└── comfyui/             # git submodule (local) / runtime clone target (Spaces)
```

---

## Useful one-liners

```bash
# What's the Space's current SHA vs local HEAD
hf_sha=$(curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "https://huggingface.co/api/spaces/techfreakworm/LTX2.3-Studio" \
  | jq -r '.sha')
echo "HF: ${hf_sha:0:8}  local: $(git rev-parse HEAD | cut -c1-8)"

# Local commits ahead of origin
git log origin/master..HEAD --oneline

# All class_types referenced by workflows (cross-check against custom_nodes)
python3 -c "import json, glob, sys
seen = set()
for p in glob.glob('workflows/*.json'):
    seen |= {n.get('class_type','') for n in json.load(open(p)).values()}
for c in sorted(seen): print(c)"

# Models referenced by workflows but not in registry
python3 -c "import json, glob, models
needed = set()
for p in glob.glob('workflows/*.json'):
    needed |= models.walk_workflow_for_models(json.load(open(p)))
unmapped = needed - set(models.MODEL_REGISTRY)
print('unmapped:', sorted(unmapped) or 'none')"
```
