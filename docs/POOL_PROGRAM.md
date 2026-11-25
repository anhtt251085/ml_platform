# Model pool program (M5) — status

Each production model type needs data that may be used commercially, a subject/recording-level evaluation, and
numbers measured on both boards. The platform does everything after the data: import, train three sizes,
INT8/QAT, SAI plan search, bit-exact evaluation, HIL, model card, leaderboard (Model pool page).

| # | Model type | Data | Licence | Status |
|---|---|---|---|---|
| 1 | Activity & fall (`har-imu`) | UCI HAR (30 subjects) via `tools/import_uci_har.py` | CC BY 4.0 | **Done**: 95.4 % bit-exact on held-out subjects, 1.7 KB RAM / 5.4 KB flash on M7 — [model card](model_cards/har-imu_uci-har.md). Falls need SisFall/own data (SisFall licence: check) |
| 2 | Keyword spotting (`kws`) | Google Speech Commands v2 | CC BY 4.0 | Ingest ready (ZIP of `<word>/*.wav`, speaker-level split, C MFCC). Needs a 2.3 GB download on a lab worker |
| 3 | Bearing fault (`bearing-fault`) | CWRU / Paderborn | CWRU: no explicit licence; Paderborn: check terms | Licence review first; otherwise own test rig (recommended: customer data campaign) |
| 4 | Battery SoH (`battery-soh`) | NASA PCoE battery ageing | US government work (public domain) — confirm | Importer to write (cycle -> V/I/T windows, capacity label) |
| 5 | DC arc fault (`dc-arc-fault`) | none public at quality | — | Needs a UL 1699B-style arc generator bench; plan with a PV inverter partner |
| 6 | Thermal presence (`thermal-presence`) | none public for MLX90640 at quality | — | Capture campaign with the board-farm data logger (M2 connector) |

Leaderboard rules: only `pool_benchmark` projects run by SAI research appear; synthetic results are labelled and
ranked after real-data results; every row links to its report (provenance, SAI checks, plans per board).
