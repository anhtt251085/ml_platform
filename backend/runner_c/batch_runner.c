/* Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE. */
/* Batch runner for a SAI host-cpu build: reads N int8 inputs from stdin, writes N raw outputs to stdout.
   Linked against the build's lib/libsai_model.a, so the numbers are the deployed program's, bit for bit. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "sai_model.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: batch_runner N < in.bin > out.bin\n"); return 2; }
    long n = strtol(argv[1], NULL, 10);
    size_t in_nd = 0, out_nd = 0;
    const int64_t *in_shape = sai_model_input_shape(0, &in_nd);
    const int64_t *out_shape = sai_model_output_shape(0, &out_nd);
    size_t in_bytes = SAI_MODEL_INPUT0_BYTES, out_bytes = SAI_MODEL_OUTPUT0_BYTES;
    size_t ws_bytes = SAI_MODEL_WORKSPACE_BYTES ? SAI_MODEL_WORKSPACE_BYTES : 8;
    void *ws = aligned_alloc(64, (ws_bytes + 63) & ~(size_t)63);
    int8_t *in = malloc(in_bytes);
    int8_t *out = malloc(out_bytes);
    sai_model_t model;
    sai_run_context_t ctx = { ws, SAI_MODEL_WORKSPACE_BYTES, NULL };
    sai_status_t st = sai_model_init(&model, &ctx);
    if (st != SAI_OK) { fprintf(stderr, "init: %s\n", sai_model_status_string(st)); return 1; }
    for (long i = 0; i < n; i++) {
        if (fread(in, 1, in_bytes, stdin) != in_bytes) { fprintf(stderr, "short input at %ld\n", i); return 1; }
        sai_tensor_view_t iv = { in, in_bytes, in_shape, in_nd, SAI_MODEL_INPUT0_DTYPE };
        sai_tensor_buffer_t ob = { out, out_bytes, out_shape, out_nd, SAI_MODEL_OUTPUT0_DTYPE };
        st = sai_model_invoke(&model, &iv, 1, &ob, 1);
        if (st != SAI_OK) { fprintf(stderr, "invoke %ld: %s\n", i, sai_model_status_string(st)); return 1; }
        fwrite(out, 1, out_bytes, stdout);
    }
    sai_model_deinit(&model);
    return 0;
}
