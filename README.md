# SAI Platform (v0.6: model pool program)

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

## Run with Docker (SaaS or on-prem)

```bash
export SAI_PLATFORM_ADMIN_PASSWORD='choose-a-strong-one'
SAI_SRC=/path/to/mcu_edge_ai docker compose up -d --build        # http://localhost:8000
```

* `SAI_PLATFORM_MODE=multi` (default): multi-tenant SaaS. SAI admins create customer organisations and invite users
  (no public sign-up).
* `SAI_PLATFORM_MODE=single SAI_PLATFORM_ORG_NAME="Acme"`: one customer per install (on-prem / customer VPC).
* The image pins the toolchains: Arm GNU (Debian) and **xPack `riscv-none-elf-gcc` 13.2.0-2** (checksummed), the
  toolchain SAI's MD6605 measurements use. Behind a TLS-inspecting proxy add
  `--secret id=extra_ca,src=proxy-ca.crt` to `docker build`.
* Jobs run in the `worker` service (`python -m backend.worker`); scale it with `docker compose up --scale worker=N`.

## Quick start (development)

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
pip install /path/to/a/copy/of/mcu_edge_ai      # the SAI compiler (`sai` CLI); install from a copy, not the checkout
# optional MCU toolchains (exact flash + proven stack):
#   Arm:   apt install gcc-arm-none-eabi libnewlib-arm-none-eabi
#   RV32:  export SAI_MD6605_GCC=/opt/xpack/.../bin/riscv-none-elf-gcc   (xPack 13.2.0-2; the Ubuntu-picolibc
#          shim in toolchains/ is dev-only: it links, but gives no proven stack bound)
export SAI_PLATFORM_ADMIN_PASSWORD='dev-password-1'   # first start creates admin@sai.local
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
| `backend/app.py` | FastAPI: auth, admin, catalog, datasets, projects, jobs, studio, explorer, boards, agent API, UI |
| `backend/auth.py` | Orgs, users, roles, sessions, agent tokens, audit log, usage metering |
| `backend/worker.py` | Standalone job worker (DB-backed queue) |
| `frontends/` + `backend/features.py` | C feature front ends (MFCC, band energy) and their Python binding |
| `harness/` + `backend/hil.py` | HIL harness firmware, board ports, QEMU runner |
| `backend/regression.py` | Nightly pool x boards regression |
| `Dockerfile`, `docker-compose.yml` | Image with SAI + pinned toolchains; API + worker services |
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

## Model pool program (M5)

* **First real-data model**: activity recognition on UCI HAR (CC BY 4.0), split by subject — 95.4 % bit-exact on
  held-out people with a 746-parameter model in 1.7 KB RAM / 5.4 KB flash on Cortex-M7
  ([model card](docs/model_cards/har-imu_uci-har.md), importer `tools/import_uci_har.py`).
* **Leaderboard** on the Model pool page: pool-benchmark projects (SAI research only) per model type, real data
  ranked before synthetic, linked to full reports.
* **Template versioning**: published versions, history, deprecation; every report records the template version.
* Status of the six production model types and their data/licences: `docs/POOL_PROGRAM.md`.

## Board farm (M4)

* **HIL harness firmware** (`harness/`): bare-metal C + startup + linker scripts per board, linked with the SAI
  package library using SAI's own target flags; prints outputs, cycles (median of 31) and painted stack.
* **Emulated lab out of the box**: with QEMU installed the platform runs two in-process boards — Cortex-M7
  (MPS2-AN500) and RV32IMAFC (virt) — executing the real firmware; outputs are checked bit-exactly, cycles are
  labelled emulated. A board run for another target compiles the model for that target on demand.
* **Physical boards**: NUCLEO-H753ZI (USART3 + DWT) and MD6605 EVB ports link in CI; lab agent drivers for pyOCD
  and J-Link, uhubctl power cycling, health reporting, quarantine after 3 failures.
* **Nightly regression** of the whole pool on every board (`python -m backend.regression`, nightly workflow).
* Protocol and ports: `docs/BOARD_AGENT_PROTOCOL.md`.

## Compiler integration (M3)

* **Explored plans become candidates**: the Pareto frontier (workspace vs SAI rank) of `sai explore` is built
  with `sai compile --plan NAME --candidate-id ID`, so intermediate RAM/latency trade-offs get exact flash and a
  proven stack bound and can be selected; rebuilds replay the same plan.
* **Certificate re-check**: every package's memory certificate is re-derived by `sai certify --check`.
* **`measured` evidence**: a real board-farm run stores cycles per `target:plan_digest`; every later plan search
  that produces the same program shows that measurement instead of an estimate.
* **SAI contract tests** (`tests/test_sai_contract.py`) pin the JSON fields and schema versions the platform reads;
  newer schemas surface as compatibility warnings on the Compiler & targets page.
* `tools/sai_findings_repro.sh` reproduces the open SAI findings (issue #9) for upstream.

## Data & training (M2)

* **C feature front ends** (`frontends/sai_features.[ch]`): MFCC and log band energies in portable C (no heap,
  ~1.6 KB on Cortex-M7). Training calls the same C through ctypes, so train and deploy features are identical;
  packages ship the sources plus `sai_frontend_config.h` with the exact parameters. Keyword spotting takes raw
  16 kHz WAV, vibration anomaly takes raw accelerometer windows.
* **Uploads**: CSV with a column-mapping step (label / recording / time / channels), NPZ, ZIP of `<label>/*.wav`.
* **Recording-level splits**: windows of one recording never straddle train and test (CSV segment, WAV speaker,
  NPZ `groups`); the job page states the split used.
* **Data quality report**: duplicates (and duplicates across recordings), clipping, recordings per class,
  sample-period jitter from the time column.
* **Licences**: every dataset records one; non-commercial, research-only, unknown and synthetic data produce
  *evaluation-only* packages (marker file, `#warning`, model card).
* **Pool base weights**: researchers train base weights per model type; customers fine-tune from them.
* **Learning-rate search** (optional) and CUDA training when a GPU worker has one.

## Security model (M1)

* Sign-in with email + password (scrypt); sessions are random tokens stored as SHA-256; HttpOnly cookie or bearer.
* Roles: `platform_admin`, `researcher`, `lab` (SAI staff), `org_admin`, `member` (customers).
* Every dataset, project, job, download and board run is scoped to the caller's organisation (other orgs get 404).
* Lab agents use per-agent tokens created in Admin → Lab agents (shown once, stored hashed, revocable).
* Audit log of sign-ins, uploads, projects, downloads, admin actions; usage metering per org.
* Reproducibility: every package records SAI version, toolchain and digests, dataset SHA-256, seeds;
  "Rebuild & compare" proves the package rebuilds byte for byte.

## Known limitations

SQLite (one host; Postgres is a later swap of `db.conn`), local file storage, CPU training, one input/one output
models; physical boards need a lab agent (the emulated lab runs without one). See `docs/IMPLEMENTATION_PLAN.md`.
