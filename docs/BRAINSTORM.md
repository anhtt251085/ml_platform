# SAI Platform – idea brainstorm

Ideas grouped by who they serve. ★ = built, ◐ = partly, ○ = not built (status as of the sequence-model / multi-model release).

## 1. Customer (OEM firmware / product engineer)

| | Idea | Why it matters |
|---|---|---|
| ★ | 4-step wizard: model type → data → board & goal → build | Zero ML knowledge needed |
| ★ | Goals in product language: fastest, least RAM, smallest flash, best accuracy, balanced + hard limits | Customers think in budgets, not schedules |
| ★ | Automatic size search (tiny/small/base) + plan search, pick the smallest model that meets the accuracy target | "Right-sized" is the SAI promise |
| ★ | Bit-exact accuracy of the deployed C program (not a framework's approximation) | Removes the #1 TinyML surprise: accuracy drop after deployment |
| ★ | Package: library, headers, `sai_app.h` (integer-only input quantization, decode), `main_example.c`, model card, SBOM, memory certificate | Drop-in for STM32CubeIDE / Eclipse projects |
| ★ | Evidence badges on every number | Trust: exact vs proven vs estimated |
| ◐ | "What-if" sliders: window length and channels → instant RAM/flash/latency scaled from the project's SAI numbers, exact SAI check on demand (accuracy still needs a retrain) | Lets the customer trade sensor cost against accuracy |
| ★ | Duty-cycle and battery-life estimate: inferences/s × energy per inference (assumed MCU profile, or measured `energy_uj` from physical board runs), sensor and radio budget | The question battery-product buyers actually ask |
| ◐ | Side-by-side vs TFLite Micro / CMSIS-NN: TFLM greedy arena replayed on the same graph + CMSIS-NN scratch, assumed code size and cycles/MAC, the customer's measured TFLM build replaces the model (no in-farm TFLM build yet) | The sales proof point, generated per customer |
| ○ | One-click CubeIDE / PlatformIO / Zephyr / MCUXpresso project export | Removes integration friction |
| ◐ | Model versioning + A/B and OTA-ready packages (signed with `sai_signing`) | Field updates, rollbacks |
| ◐ | Drift monitor: field recordings / datasets vs training data (PSI per channel feature, prediction mix, confidence), history per project (on-device statistics logging still open) |
| ★ | Operator-gap log: every model refused for a missing operator (SAI, SAI front end, model editor) with target and co-occurring ops, grouped for the compiler roadmap | Tells the compiler team what to build next | Lifecycle revenue |

## 2. Data

| | Idea |
|---|---|
| ★ | CSV (windowed per recording, majority label), NPZ, HTTPS, synthetic demo data per model type |
| ★ | Column-mapping wizard (choose time / channels / label / segment columns, units, sample rate) |
| ★ | Label studio for time series: zoomable min/max view, paint and erase label spans, suggestions from a finished project's INT8 model, recordings → dataset (groups per span) |
| ◐ | Live capture: a device / gateway / `tools/capture_client.py` (serial or stdin) posts samples with a capture token while the engineer watches and marks labels live (BLE / MQTT bridges still open) |
| ◐ | Connectors: S3 (+ GCS HMAC, MinIO, R2), Azure Blob, InfluxDB v2, PostgreSQL / TimescaleDB (read-only SELECT); secrets Fernet-encrypted; public endpoints only unless single-customer mode (OPC-UA, Edge Impulse import still open) |
| ★ | Data quality report: clipping, sample-rate jitter, class balance, leakage between recordings, duplicate windows |
| ◐ | Privacy modes: on-prem install, and customer-VPC workers for private data sources (credentials stay on the worker; data imported, training still on the platform)  |
| ★ | Physics augmentation: bearing kinematics (BPFO/BPFI/BSF/FTF) with resonance ringing, MCSA sidebands, DC arcs from the machine's own parameters; mix with real data plus gain / speed / noise variants |

## 3. Researcher (SAI model pool team)

| | Idea |
|---|---|
| ★ | Model Studio: layer builder restricted to SAI ops, live shapes/MACs, SAI fit check on every board before training |
| ★ | Publish to pool (draft / published), copy a built-in as a starting point |
| ◐ | NAS inside the SAI search space: search architectures with SAI's exact RAM/flash as constraints (hardware-in-the-loop NAS) |
| ★ | Pretrained base weights per model type + customer fine-tuning (few-shot on customer data) |
| ★ | Leaderboard per model type: accuracy × RAM × latency on every board, nightly on the board farm |
| ★ | Feature front ends as first-class pool blocks (FFT band energies, MFCC, envelope, RMS) with a C implementation that training uses too (no train/deploy skew) |
| ★ | Benchmark corpus: MLPerf Tiny reference tasks (KWS, VWW, IC, AD) + every model type, SAI on every board vs TFLM, compared run over run  |
| ○ | Operator-gap tracker: every refused customer model is logged with the missing op → drives SAI Core compatibility releases |

## 4. Compiler integration (SAI)

| | Idea |
|---|---|
| ★ | Plan search: `sai estimate` × 3 objectives × each board, `sai explore` plan space, budgets checked against board profiles |
| ★ | `sai compile` + `sai verify` (19 checks) on every delivered package |
| ★ | Latency: SAI kernel cost model where it exists, heuristic elsewhere, calibrated from real boards later |
| ★ | Build any explored candidate (`--plan NAME --candidate-id`), not only the three objective winners; `--certify-top N` |
| ★ | Edition / tier gating: SAI Standard vs Professional plans shown side by side ("Professional saves 31 % latency on your model") – an upsell surfaced by the product itself |
| ★ | Multi-model option: several pool models sharing one workspace on one MCU (e.g. anomaly + classifier) |
| ★ | Measured-row registry: every board-farm run becomes a SAI measurement record, so `sai estimate` reports `measured` |

## 5. Hardware / board farm

| | Idea |
|---|---|
| ★ | Board registry, HIL queue, bit-exact output check, calibration fit (simulated agent today) |
| ★ | Harness firmware per target (UART vectors, DWT/mcycle, stack painting) |
| ○ | Power: INA228 / Joulescope per board → µJ per inference, sleep current |
| ★ | Nightly regression: every pool model × every board × every SAI release; alerts on regressions |
| ◐ | Virtual dev board: the customer's firmware on QEMU boards with their own inputs, UART and bit-exact check (physical boards through the farm)  |
| ◐ | Partner boards: catalogue (planned / community / certified, mapped to a SAI build) and partner-hosted lab agents  |

## 6. Business

| Idea | Notes |
|---|---|
| Free tier: model pool + fit check + estimates | Lead generation; no package download |
| Pro seat: training, packages, board farm minutes | Subscription per seat / per project |
| Deployment rights: per-unit royalty or buyout | Matches SAI's existing licensing structure |
| Pre-certified model blocks (arc fault, BMS, PdM) | Highest margin; per-unit royalty; regulatory evidence pack |
| Silicon-vendor white label | Platform embedded in a vendor's SDK portal |
| Services: custom model + data collection campaigns | NRE, seeds the pool |

## 7. Trust, safety, compliance

* Reproducible builds: SAI version, toolchain digest, dataset hash, seed recorded in `report.json` (◐ today).
* Signed packages and SBOM (SAI already emits CycloneDX SBOM ★; signing ○).
* Functional-safety evidence bundle: memory certificate, proven stack bound, deterministic cycles, test vectors ○.
* Dataset licence tracking: refuse commercial packaging when a dataset licence is non-commercial ○.
