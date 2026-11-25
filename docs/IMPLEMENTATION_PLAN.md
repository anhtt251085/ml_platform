# SAI Platform – implementation plan

Status 25 Nov 2026: prototype v0.1 works end to end (this repository). This plan takes it to a product.

## 1. Users and jobs to be done

| Persona | Job | Surface |
|---|---|---|
| OEM engineer (customer) | "Give me a model for *this* sensor on *this* MCU that fits and is accurate" | Wizard, job page, package, board run |
| SAI researcher | Grow and maintain the model pool; prove each model fits every board | Model Studio, fit check, leaderboard |
| SAI field / sales engineer | Build packages for Stage-1 OEMs; demo on real boards | Projects, Board Farm |
| Lab operator | Keep boards online, calibrate, run regressions | Board Farm, agents |
| Admin | Orgs, licences, tiers, quotas | Admin console (M1) |

## 2. Architecture

```
 Browser (SPA) ──HTTPS──► API (FastAPI) ──► Postgres (orgs, projects, jobs, templates, boards, calibration)
                              │        └──► Object store (datasets, ONNX, builds, packages, reports)
                              ▼
                        Job queue (Redis + Arq/Celery)
             ┌────────────────┼─────────────────────────┐
     train workers (GPU)  compile workers (CPU)      board-farm service
     torch, QAT           sai CLI + pinned            HIL queue ◄── lab agents (USB boards)
                          toolchains (container)       calibration ──► latency model
```

Rules that already hold in v0.1 and must stay:

* **SAI is used only through its CLI and JSON schemas** (`sai estimate/explore/compile/verify/inspect`). The
  platform never imports compiler internals, so SAI releases and the platform version independently.
* **Evidence labels end to end** (exact / linked / proven / measured / calibrated / SAI model / heuristic).
* **Bit-exact accuracy** of the deployed program is the reported accuracy; framework numbers are secondary.

## 3. What v0.1 delivers (done)

| Area | Delivered |
|---|---|
| Model pool | 11 built-in model types in 7 domains, three sizes each, all SAI-legal; researcher templates (draft/published) |
| Data | CSV windowing per recording, NPZ, HTTPS fetch, synthetic demo data per type, data summary + warnings |
| Training | Classification, regression, anomaly autoencoder; BatchNorm folded at export; QAT when INT8 loses more than 1 % (chosen on validation) |
| Quantization | Calibrated per-tensor activations, per-channel weights, QDQ ONNX that SAI imports (1-D and 2-D, residual Add) |
| SAI | Plan search over objectives × boards + explore space; board budgets from SAI profiles; goal-based selection; compile + verify; host build for exact accuracy |
| Package | SAI build + `sai_app.h` (integer-only Q16 input quantization, decode) + example main + model card + report |
| Studio | Layer builder, live shapes/MACs, SAI fit check (seconds), publish |
| Explorer | Upload INT8 ONNX/TFLite → inspect + plans |
| Board farm | Registry, queue, bit-exact output check, calibration fit, simulated agent, reference agent + protocol |
| Quality | 21 automated tests incl. end-to-end API + board run; UI tested in Chromium at desktop and phone widths |

## 4. Milestones

Estimates assume 2 backend, 1 ML, 1 frontend, 1 embedded engineer; total ≈ 6 months to GA.

### M1 – Production foundation (weeks 1–6)
* Auth (OIDC), organisations, roles: customer / researcher / sales / lab / admin; per-org data isolation.
* Postgres + migrations; S3-compatible object store; signed download URLs.
* Job queue with retries, cancellation, priorities, quotas; separate train and compile worker pools.
* Compile worker image with **pinned** SAI version and toolchains (Arm GNU, xPack riscv-none-elf-gcc). The v0.1
  RV32 shim (Ubuntu picolibc) is dev-only.
* Every project records SAI version, toolchain digests, dataset hash, seeds → reproducible rebuild button.
* Edition/tier gating: Standard (`--tier core`) vs Professional (`pro`); show both plans, mark the delta.
* **Exit:** two orgs cannot see each other's data; a rebuild of an old project is byte-identical.

### M2 – Data & training depth (weeks 4–12)
* Column-mapping wizard; units and sample-rate checks; recording-level split (no leakage); dataset versions.
* **Feature front ends in C** (FFT band energy, MFCC, RMS/envelope) shipped in the package and used by training
  through the same code (Python binding of the C), so train and deploy features are identical.
* GPU training, k-fold, augmentation per domain, hyperparameter search within the SAI op set.
* Pretrained base weights per pool model; fine-tuning on customer data.
* Dataset licence field; packaging blocked for non-commercial data.
* **Exit:** KWS from raw WAV works end to end; recording-level split is the default.

### M3 – Compiler integration depth (weeks 6–12)
* Build any explored candidate (`--plan --candidate-id`), `--certify-top N`, `sai certify` certificates in the package.
* Multi-model option (shared workspace) when SAI enables it; CMSIS-NN option when its attribution path is cleared.
* Board-farm runs written as SAI measurement records → `sai estimate` returns `measured` for known programs.
* Upstream the issues found while building v0.1 (section 7).
* **Exit:** a delivered package's latency is `measured` for every board in the farm.

### M4 – Board farm v1 (weeks 8–16)
* HIL harness firmware for STM32H753 and MD6605 (UART COBS vectors, DWT/mcycle, stack painting).
* Agent drivers: pyOCD, J-Link; board health checks; automatic recovery (power cycling via USB hub control).
* Power measurement (INA228 per board, Joulescope for reference) → µJ/inference, battery-life estimates.
* Nightly regression: pool × boards × SAI release; dashboard + alerts.
* **Exit:** 6 boards (2 targets × 3) online; calibration error below 10 % on the pool.

### M5 – Model pool program (weeks 6 → ongoing)
* First six production models with licensed or own data: DC arc fault, bearing + motor current, battery SoH,
  KWS, HAR/fall, thermal presence.
* Leaderboard per model type (accuracy × RAM × latency × energy per board); model cards; version and deprecation policy.
* MLPerf Tiny tasks as compiler + pool regression and as public proof.
* **Exit:** each of the six has a real-data model card and measured numbers on both boards.

### M6 – Commercial launch (weeks 12–24)
* Pricing: free tier (pool, fit check, estimates), Pro (training, packages, farm minutes), deployment royalty /
  buyout per SAI licensing spec; usage metering.
* Signed packages (`sai_signing`), customer portal, support workflow, SLA.
* Deployment options: SaaS, customer-VPC worker ("data never leaves"), fully on-prem.
* Silicon-vendor white-label variant.
* **Exit:** first paying customer through self-serve; one vendor pilot.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Customers will not upload proprietary data | VPC / on-prem workers; only metrics leave; clear data-retention policy |
| Latency estimates mistrusted | Evidence labels everywhere; board-farm measurement as one click; calibration error published |
| Synthetic demo results mistaken for real | Labelled in dataset, job page and model card; packaging of synthetic-data models watermarked |
| Train/deploy feature skew | One C feature implementation used by both (M2) |
| SAI CLI changes break the platform | Schema-versioned JSON; contract tests against each SAI release in CI |
| Dataset licences | Licence field + packaging block; legal review for the pool's datasets |
| Board-farm reliability | Health checks, power cycling, redundancy (≥ 2 boards per target) |

## 6. KPIs

Time from data upload to verified package (target < 15 min for 1-D models) · share of projects whose selected plan
fits on the first run · INT8-vs-float drop (target < 1 %) · calibration error (target < 10 %) · pool coverage
(model types × boards with measured numbers) · conversion free → Pro.

## 7. Findings about SAI from building v0.1 (not changed in the SAI repository)

1. `sai explore --one-axis` fails with `schema.invalid` (internal_error) on 1-D graphs where the plan set includes
   `+fcp` names (e.g. `full+bounded+fcp`): `explore-report-1d.schema.json`'s name pattern does not admit `+fcp`
   (added by D59/D63). The platform falls back to `sai estimate` per objective.
2. With a RISC-V GCC that lacks `nano.specs`, `sai estimate --target md6605` raises an unhandled `RuntimeError`
   (internal_error + traceback) instead of a structured `toolchain_missing` error.
3. A 1-D graph with a strided conv lowers on the 2-D path, so `explore --one-axis` reports `explore.two_axes`;
   the platform retries without the flag. Worth documenting for API users.
4. With the dev RV32 toolchain the MD6605 stack bound came back `unavailable` (Cortex-M7 gives
   `proven_upper_bound`); to re-check with the production xPack toolchain.

## 8. Open decisions for the owner

1. SaaS first, or on-prem/VPC first (affects M1 order)?
2. Do Stage-1 OEMs get platform accounts, or do SAI engineers operate it for them?
3. Which six pool models first, and what data can be licensed for them?
4. Pricing anchor: per seat, per project, or per deployed unit?
5. Will Standard vs Professional be visible side by side in the platform (upsell), or only the purchased tier?
