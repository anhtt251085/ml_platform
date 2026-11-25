/* SAI HIL harness: runs a SAI package on the MCU with the test vectors baked in, and prints a line protocol on
   the board's UART:

     SAI-HIL 1 <n_vectors> <in_bytes> <out_bytes>
     OUT <i> <hex bytes>              one per vector (bit-exact output check, server side)
     CYC <median> <min> <max>         cycles of vector 0: median of 31 runs after 4 warm-up, net of an empty call
     STACK <bytes>                    painted stack high-water mark of the inference calls
     DONE <status>

   Board specifics live in a port (harness/ports/*.c): port_init, port_putc, port_cycles, port_exit. */
#include <stdint.h>
#include <stddef.h>
#include "sai_model.h"
#include "sai_hil_port.h"
#include "hil_vectors.h"          /* generated: SAI_HIL_N, sai_hil_inputs[SAI_HIL_N][SAI_MODEL_INPUT0_BYTES] */

#define REPS 31
#define WARMUP 4
#define PAINT 0xA5u

extern uint8_t _stack_bottom[];   /* linker script */

static uint8_t workspace[SAI_MODEL_WORKSPACE_BYTES + 8] __attribute__((aligned(SAI_MODEL_WORKSPACE_ALIGNMENT)));
static int8_t output[SAI_MODEL_OUTPUT0_BYTES];
static sai_model_t model;

static void puts_(const char *s) { while (*s) port_putc(*s++); }
static void putu(uint32_t v) {
    char b[11]; int i = 10; b[i] = 0;
    do { b[--i] = (char)('0' + v % 10u); v /= 10u; } while (v);
    puts_(b + i);
}
static void puthex8(uint8_t v) {
    static const char h[] = "0123456789abcdef";
    port_putc(h[v >> 4]); port_putc(h[v & 15]);
}

static sai_status_t run(const int8_t *in) {
    size_t ni, no;
    const int64_t *si = sai_model_input_shape(0, &ni), *so = sai_model_output_shape(0, &no);
    sai_tensor_view_t iv = { in, SAI_MODEL_INPUT0_BYTES, si, ni, SAI_MODEL_INPUT0_DTYPE };
    sai_tensor_buffer_t ob = { output, SAI_MODEL_OUTPUT0_BYTES, so, no, SAI_MODEL_OUTPUT0_DTYPE };
    return sai_model_invoke(&model, &iv, 1, &ob, 1);
}

static void sort(uint32_t *a, int n) {
    for (int i = 1; i < n; i++) { uint32_t v = a[i]; int j = i; while (j && a[j - 1] > v) { a[j] = a[j - 1]; j--; } a[j] = v; }
}

__attribute__((noinline)) static uint32_t empty_call(void) { uint32_t t0 = port_cycles(); return port_cycles() - t0; }

/* Paint from the stack bottom up to just below main's frame; after the calls, the first unpainted byte from the
   bottom is the deepest stack the inference reached. Reported: bytes from there up to main's marker, i.e. the
   frames of run() + sai_model_invoke + SAI kernels (what SAI's static bound covers, plus run()'s small frame). */
static uint8_t *volatile g_marker;

__attribute__((noinline)) static void paint_stack(void) {
    uint8_t *p = _stack_bottom, *end = g_marker - 64;     /* 64 B: this function's own frame */
    while (p < end) *p++ = PAINT;
}

static uint32_t stack_used(void) {
    uint8_t *p = _stack_bottom;
    while (p < g_marker && *p == PAINT) p++;
    return (uint32_t)(g_marker - p);
}

int main(void) {
    volatile uint8_t marker = 0;
    g_marker = (uint8_t *)&marker;
    port_init();
    sai_run_context_t ctx = { workspace, SAI_MODEL_WORKSPACE_BYTES, 0 };
    puts_("SAI-HIL 1 "); putu(SAI_HIL_N); port_putc(' '); putu(SAI_MODEL_INPUT0_BYTES); port_putc(' ');
    putu(SAI_MODEL_OUTPUT0_BYTES); puts_("\n");
    if (sai_model_init(&model, &ctx) != SAI_OK) { puts_("DONE init_failed\n"); port_exit(1); }

    /* outputs, with the stack painted around the calls */
    paint_stack();
    for (uint32_t i = 0; i < SAI_HIL_N; i++) {
        if (run(sai_hil_inputs[i]) != SAI_OK) { puts_("DONE invoke_failed\n"); port_exit(1); }
        puts_("OUT "); putu(i); port_putc(' ');
        for (uint32_t k = 0; k < SAI_MODEL_OUTPUT0_BYTES; k++) puthex8((uint8_t)output[k]);
        puts_("\n");
    }
    uint32_t stack = stack_used();

    /* timing: vector 0, median of REPS after WARMUP, net of an empty call */
    uint32_t t[REPS], overhead = empty_call();
    for (int r = 0; r < WARMUP; r++) run(sai_hil_inputs[0]);
    for (int r = 0; r < REPS; r++) {
        uint32_t t0 = port_cycles();
        run(sai_hil_inputs[0]);
        uint32_t dt = port_cycles() - t0;
        t[r] = dt > overhead ? dt - overhead : 0;
    }
    sort(t, REPS);
    puts_("CYC "); putu(t[REPS / 2]); port_putc(' '); putu(t[0]); port_putc(' '); putu(t[REPS - 1]); puts_("\n");
    puts_("STACK "); putu(stack); puts_("\n");
    puts_("DONE ok\n");
    port_exit(0);
    return 0;
}
