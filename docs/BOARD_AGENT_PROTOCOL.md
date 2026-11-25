# Board agent protocol (v0.1)

A lab agent is a small process on a PC with MCU boards on USB. It owns its boards; the server owns the queue.
All calls carry `Authorization: Bearer <agent token>` (per-agent tokens and mTLS in production).

| Call | Body | Answer |
|---|---|---|
| `POST /api/agent/register` | `{agent_id, boards: [{id, name, target, serial, probe, driver}]}` | registered board ids |
| `POST /api/agent/poll` (every 2 s, also the heartbeat) | `{agent_id}` | `{run: null}` or a run for one of its idle boards |
| `GET /api/agent/runs/{id}/package` | – | the SAI `package.zip` |
| `POST /api/agent/runs/{id}/result` | see below | the stored run |

A run: `{id, job_id, board_id, target, request: {objective, schedule, macs, estimated_cycles, stack_bound,
expected_outputs: [[int8...] x N]}}`. Test vectors are the package's own `tests/` plus the job's
`hil_vectors` (32 quantized test windows).

Result: `{ok, simulated, outputs: [[int8...]], cycles, latency_ms, stack_painted, energy_uj?, board, firmware,
n_vectors, log: [...], error?}`.

Server side: outputs are compared **bit-exactly** with the host build of the same package (`outputs_match`).
A successful, non-simulated run adds `(target, objective, MACs, cycles)` to the calibration set; the latency model
becomes `cycles = a·MACs + b` per target/objective, and the plan tables switch the label to `calibrated`.

## Harness firmware (milestone M4)

Per target, `harness/<target>/` links the package's `lib/libsai_model.a` with a runner that:

1. paints the stack with a pattern and reads the high-water mark after the run;
2. receives vectors over UART (COBS frames), calls `sai_model_invoke`, returns outputs;
3. counts cycles with DWT `CYCCNT` (Cortex-M) or `mcycle` (RV32), net of an empty call, median of 31 after warm-up
   (the statistic the SAI measurement sessions already use);
4. optionally toggles a GPIO around inference for an external power analyser (energy per inference).

Drivers: pyOCD (ST-LINK / CMSIS-DAP), J-Link Commander, OpenOCD; one `BoardDriver` class each
(`agent/board_agent.py`).
