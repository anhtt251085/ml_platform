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
| ○ | "What-if" sliders: window length, channels, sample rate → instant RAM/latency/accuracy curve | Lets the customer trade sensor cost against accuracy |
| ○ | Duty-cycle and battery-life estimate (inferences/s × energy per inference from the board farm) | The question battery-product buyers actually ask |
| ○ | Side-by-side vs TFLite Micro / CMSIS-NN (same model, same board, measured) | The sales proof point, generated per customer |
| ○ | One-click CubeIDE / PlatformIO / Zephyr / MCUXpresso project export | Removes integration friction |
| ◐ | Model versioning + A/B and OTA-ready packages (signed with `sai_signing`) | Field updates, rollbacks |
| ○ | Drift monitor: the device logs feature statistics; the platform flags distribution shift and proposes retraining | Lifecycle revenue |

## 2. Data

| | Idea |
|---|---|
| ★ | CSV (windowed per recording, majority label), NPZ, HTTPS, synthetic demo data per model type |
| ★ | Column-mapping wizard (choose time / channels / label / segment columns, units, sample rate) |
| ○ | Label studio for time series: scrub a signal, paint label spans, auto-suggest from a pre-trained pool model |
| ○ | Live capture: the board farm or the customer's dev board streams sensor data (USB CDC / BLE / MQTT) straight into a dataset |
| ○ | Connectors: S3 / GCS / Azure, OPC-UA, InfluxDB / TimescaleDB, Edge Impulse import |
| ★ | Data quality report: clipping, sample-rate jitter, class balance, leakage between recordings, duplicate windows |
| ○ | Privacy modes: on-prem runner, or "bring the compiler to the data" (customer VPC worker, only metrics leave) |
| ○ | Synthetic augmentation from physics models (bearing defect frequencies from geometry, arc models) to bootstrap small datasets |

## 3. Researcher (SAI model pool team)

| | Idea |
|---|---|
| ★ | Model Studio: layer builder restricted to SAI ops, live shapes/MACs, SAI fit check on every board before training |
| ★ | Publish to pool (draft / published), copy a built-in as a starting point |
| ◐ | NAS inside the SAI search space: search architectures with SAI's exact RAM/flash as constraints (hardware-in-the-loop NAS) |
| ★ | Pretrained base weights per model type + customer fine-tuning (few-shot on customer data) |
| ★ | Leaderboard per model type: accuracy × RAM × latency on every board, nightly on the board farm |
| ★ | Feature front ends as first-class pool blocks (FFT band energies, MFCC, envelope, RMS) with a C implementation that training uses too (no train/deploy skew) |
| ○ | Benchmark corpus: MLPerf Tiny tasks as regression tests of the compiler + pool |
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
| ○ | Remote "virtual dev board" for customers: run *their* firmware on our boards before buying hardware |
| ○ | Partner boards: silicon vendors place boards in the farm to be "SAI-ready" (co-marketing) |

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
