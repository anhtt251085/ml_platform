# Board agent protocol (v1)

A lab agent is a small process on a PC with MCU boards on USB (or QEMU boards). It owns its boards; the server
owns the queue. All calls carry `Authorization: Bearer <agent token>`: per-agent tokens created in
Admin → Lab agents (shown once, stored hashed, revocable).

| Call | Body | Answer |
|---|---|---|
| `POST /api/agent/register` | `{agent_id, boards: [{id, name, target, serial, probe, driver}]}` | registered board ids |
| `POST /api/agent/poll` (every 2 s, also the heartbeat) | `{health: {board_id: {probe, ...}}}` | `{run: null}` or a run for one of its idle boards |
| `GET /api/agent/runs/{id}/package` | – | the SAI build for the board's target + `hil_vectors.json` |
| `POST /api/agent/runs/{id}/result` | see below | the stored run |

A run: `{id, job_id, board_id, target, request: {objective, schedule, macs, estimated_cycles, stack_bound,
expected_outputs: [[int8...] x N]}}`. Test vectors are the package's own `tests/` plus the job's
`hil_vectors` (32 quantized test windows).

Result: `{ok, simulated, outputs: [[int8...]], cycles, latency_ms, stack_painted, energy_uj?, board, firmware,
n_vectors, log: [...], error?}`.

Server side: outputs are compared **bit-exactly** with the host build of the same package (`outputs_match`).
A successful, non-simulated run adds `(target, objective, MACs, cycles)` to the calibration set; the latency model
becomes `cycles = a·MACs + b` per target/objective, and the plan tables switch the label to `calibrated`.

## Harness firmware (`harness/`, built by `backend/hil.py`)

The agent builds `harness/sai_hil.c` + a board port + startup + linker script around the package's
`lib/libsai_model.a`, with SAI's own compile flags for the target and the run's vectors baked in
(`hil_vectors.h`). The firmware prints a line protocol on the UART:

```
SAI-HIL 1 <n_vectors> <in_bytes> <out_bytes>
OUT <i> <hex>                 outputs of vector i (checked bit-exactly by the server)
CYC <median> <min> <max>      vector 0: median of 31 runs after 4 warm-up, net of an empty call
STACK <bytes>                 painted stack high-water mark of the inference calls (includes the harness frames)
DONE ok
```

| Port | Board | Cycles | Status |
|---|---|---|---|
| `qemu-mps2-an500` | QEMU Cortex-M7 + FPU | SysTick on QEMU's instruction clock (emulated) | runs in CI + platform |
| `qemu-virt-rv32` | QEMU RV32IMAFC | `mcycle` on QEMU's instruction clock (emulated) | runs in CI + platform |
| `stm32h753-nucleo` | NUCLEO-H753ZI, USART3 (ST-LINK VCP) | DWT `CYCCNT` | links in CI; validate on the first lab board |
| `md6605-evb` | Sanken MD6605 EVB | `mcycle` | links in CI (fits 8 KB SRAM); UART from the vendor SDK |

Drivers (`agent/board_agent.py`): `qemu`, `pyocd` (flash + pyserial UART), `jlink` (JLinkExe + UART). A failing
run power-cycles the board once (`uhubctl`) and retries; the server quarantines a board after 3 failed runs in a
row until the lab resets it. Emulated and simulated results never calibrate latency or count as `measured`.

## Nightly regression

`python -m backend.regression` (and `tools/full_check.sh`, and Board Farm → Run regression now): every
published pool model type × every board on the current SAI release — RAM, flash, plans that fit, and emulated HIL
bit-exactness — compared with the previous run; alerts on +5 % RAM/flash, lost fits or HIL mismatches.
