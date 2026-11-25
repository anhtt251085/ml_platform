# SAI Platform (prototype v0.1)

[![tests](https://github.com/anhtt251085/ml_platform/actions/workflows/tests.yml/badge.svg)](https://github.com/anhtt251085/ml_platform/actions/workflows/tests.yml)

A web platform on top of the SAI compiler. A customer picks a **model type** from the model pool, brings
**data**, picks **boards and a goal**, and the platform does the rest:

```
data → train 3 sizes → INT8 (PTQ, QAT when needed) → SAI plan search on every board → pick model+plan
     → sai compile + sai verify → bit-exact accuracy of the deployed C → package.zip → (board farm run)
```

Researchers design new model types in **Model Studio** (layer builder restricted to SAI's operators, live
shapes/MACs, a one-click **SAI fit check** on every board) and publish them to the pool. **Plan Explorer**
analyses any INT8 `.onnx` / `.tflite`. **Board Farm** queues hardware-in-the-loop runs on lab agents (simulated
boards today) and calibrates latency from real measurements.

## Quick start

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
pip install /path/to/a/copy/of/mcu_edge_ai      # the SAI compiler (`sai` CLI); install from a copy, not the checkout
# optional MCU toolchains (exact flash + proven stack):
#   Arm:   apt install gcc-arm-none-eabi libnewlib-arm-none-eabi
#   RV32:  export SAI_MD6605_GCC=/path/to/riscv-none-elf-gcc   (xPack; or toolchains/riscv-none-elf-gcc shim)
./run.sh                                         # http://localhost:8000
```

Tests: `SAI_PLATFORM_DATA=$(mktemp -d) pytest -q tests` (21 tests, about 25 s, the end-to-end one included;
without the `sai` CLI the 3 compiler tests skip).

CI (`.github/workflows/tests.yml`) runs on every push and pull request:
* **unit**: no SAI, always runs.
* **full**: installs SAI from `anhtt251085/mcu_edge_ai` plus the Arm and RISC-V toolchains and runs everything.
  It needs a repository secret `SAI_REPO_TOKEN` (fine-grained token, read-only *Contents* on `mcu_edge_ai`);
  without it the job skips with a notice.

## Layout

| Path | What |
|---|---|
| `backend/app.py` | FastAPI: catalog, datasets, projects, jobs, studio, explorer, boards, agent API, static UI |
| `backend/pipeline.py` | Job orchestration: train / fitcheck / analyze, packaging, model card, `sai_app.h` |
| `backend/sai_engine.py` | The only door into SAI: `sai estimate/explore/compile/verify/inspect`; budgets, goals, latency labels |
| `backend/archspec.py` | JSON architecture specs restricted to SAI ops; torch model; BN folding; fake-quant for QAT |
| `backend/qexport.py` | Calibration and QDQ ONNX export in the exact form SAI's ONNX front end imports |
| `backend/exact_eval.py` + `runner_c/` | Host build + C batch runner: accuracy of the deployed program, bit-exact |
| `backend/trainer.py` | Training (classification / regression / anomaly autoencoder), metrics, QAT fine-tune |
| `backend/catalog.py`, `synth.py` | 11 built-in model types and their synthetic demo data generators |
| `backend/data.py` | CSV (windowed per segment) / NPZ / HTTPS ingestion |
| `backend/boards.py` | Board registry, HIL queue, calibration fit, simulated lab agent |
| `agent/board_agent.py` | Reference lab agent for real boards (driver interface; pyOCD/J-Link stubs) |
| `frontend/` | Dependency-free SPA |
| `docs/` | Brainstorm, implementation plan, board agent protocol |

## Honesty rules the platform follows

* Every number carries an evidence label. SAI's labels (`static_exact`, `linked_exact`, `proven_upper_bound`,
  `measured`, `estimate`) pass through unchanged. The platform adds `sai_cost_model`, `heuristic`, `calibrated`.
* Latency is a planning number until a real board measures the build. Simulated board runs are flagged and never
  enter calibration.
* Synthetic demo data is labelled as synthetic in the dataset, the job page and the model card.

## Known limitations (prototype)

No auth, SQLite, thread-pool jobs, CPU training, one input/one output models, no feature-extraction front end in C
yet (KWS takes MFCC features), board farm simulated. See `docs/IMPLEMENTATION_PLAN.md`.
