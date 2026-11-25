# Model card: Activity recognition (har-imu) on UCI HAR

First real-data entry of the SAI model pool, produced end to end by the platform (pool benchmark).

| | |
|---|---|
| Data | UCI HAR, Anguita et al., ESANN 2013 — [archive.ics.uci.edu/dataset/240](https://archive.ics.uci.edu/dataset/240), **CC BY 4.0** |
| Input | 6 channels (total acceleration x/y/z in g, gyroscope x/y/z in rad/s), 128 samples at 50 Hz (2.56 s) |
| Classes | walking, walking_upstairs, walking_downstairs, sitting, standing, laying |
| Windows | 10,299 from 30 subjects |
| Split | **by subject** (recording-level): no person appears in both training and test |
| Selected | size `tiny`, 746 parameters, 31,712 MACs |
| Board / plan | cortex-m7, `full+im2col+spec+pw+fcp+dsp` (SAI objective fast) |
| Accuracy (float / INT8 / deployed C, bit-exact) | 95.32% / 95.39% / **95.39%** |
| Macro F1 (deployed C) | 0.954 |
| RAM | 1,744 B (workspace 1540 B static_exact, stack 204 B proven_upper_bound) |
| Flash | 5,428 B (linked_exact) |
| Latency | 0.2527 ms (sai_cost_model: heuristic scaled x3.60 to SAI's kernel cost model for this graph (uncalibrated)) |
| SAI checks | verify 19/19, certificate re-derived 11/11 |
| HIL | outputs bit-exact on emulated Cortex-M7 (QEMU MPS2-AN500) and RV32IMAFC (QEMU virt); physical-board latency pending (M4 lab) |

## All sizes and boards

| Size | Params | MACs | Float | INT8 | Best plan per board (RAM / flash / latency) |
|---|---:|---:|---:|---:|---|
| tiny | 746 | 31,712 | 95.32% | 95.39% | cortex-m7: 1,160 B / 5,484 B / 0.3834 ms (sai_cost_model); md6605: 1,164 B / 6,084 B / 1.9252 ms (heuristic) |
| small | 2,218 | 68,928 | 93.75% | 93.47% | cortex-m7: 1,392 B / 8,630 B / 0.6742 ms (sai_cost_model); md6605: 1,388 B / 9,320 B / 4.1582 ms (heuristic) |
| base | 9,070 | 353,472 | 93.33% | 93.40% | cortex-m7: 1,944 B / 18,052 B / 2.2614 ms (sai_cost_model); md6605: 1,948 B / 18,802 B / 21.2308 ms (heuristic) |

## Confusion matrix (deployed C, test subjects)

| true \ predicted | walking | walking_upstairs | walking_downstairs | sitting | standing | laying |
|---|---|---|---|---|---|---|
| walking | 234 | 1 | 5 | 0 | 1 | 0 |
| walking_upstairs | 0 | 219 | 0 | 0 | 0 | 0 |
| walking_downstairs | 0 | 0 | 199 | 0 | 0 | 0 |
| sitting | 0 | 0 | 0 | 201 | 40 | 0 |
| standing | 0 | 0 | 0 | 18 | 209 | 0 |
| laying | 0 | 0 | 0 | 0 | 0 | 282 |

## Notes

* The smallest size is the most accurate here: the larger ones overfit 21 training subjects. The platform's size search found this; a fixed "big model" would have been worse and 10x larger.
* Sitting vs standing is the main confusion, as in the literature for waist-worn phones.
* Latency is a planning estimate until a physical NUCLEO-H753ZI / MD6605 EVB measures this build in the board farm; the plan rows then switch to `measured`.
* Attribution: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN 2013. Licensed CC BY 4.0.
* Reproduce: `python tools/import_uci_har.py <uci-har.zip> har_uci.npz`, upload with licence CC BY 4.0 to the *Activity & fall detection* model type, sizes tiny/small/base, 30 epochs, both boards, goal balanced.
