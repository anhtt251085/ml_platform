/* Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE. */
/* QEMU MPS2-AN500 (Cortex-M7 + FPU): CMSDK UART0, cycles from SysTick (24-bit) extended by its wrap interrupt.
   QEMU is not cycle-accurate: with -icount the count tracks executed instructions. The platform labels these
   numbers `emulated`; they never calibrate latency. */
#include <stdint.h>
#include "../sai_hil_port.h"

#define UART0_DATA (*(volatile uint32_t *)0x40004000u)
#define UART0_STATE (*(volatile uint32_t *)0x40004004u)
#define UART0_CTRL (*(volatile uint32_t *)0x40004008u)
#define UART0_BAUDDIV (*(volatile uint32_t *)0x40004010u)
#define SYST_CSR (*(volatile uint32_t *)0xE000E010u)
#define SYST_RVR (*(volatile uint32_t *)0xE000E014u)
#define SYST_CVR (*(volatile uint32_t *)0xE000E018u)
#define SCB_ICSR (*(volatile uint32_t *)0xE000ED04u)
#define PENDSTSET (1u << 26)

static volatile uint32_t wraps;
void SysTick_Handler(void) { wraps++; }

void port_init(void) {
    UART0_BAUDDIV = 16;
    UART0_CTRL = 1;                       /* TX enable */
    SYST_RVR = 0x00FFFFFFu;
    SYST_CVR = 0;
    SYST_CSR = 7;                         /* enable, interrupt, processor clock */
}

void port_putc(char c) {
    if (c == '\n') port_putc('\r');
    while (UART0_STATE & 1u) {}
    UART0_DATA = (uint32_t)(uint8_t)c;
}

uint32_t port_cycles(void) {
    uint32_t w, v;
    do {
        w = wraps;
        v = SYST_CVR;
        /* reloaded but the wrap interrupt not yet taken (masked or pending): count that wrap now */
        if ((SCB_ICSR & PENDSTSET) && v > 0x00800000u) w++;
    } while (w != wraps && !(SCB_ICSR & PENDSTSET));
    return w * 0x01000000u + (0x00FFFFFFu - v);
}

void port_quiet(int on) { SYST_CSR = on ? 5u : 7u; }        /* keep counting, mask the wrap interrupt */
const char *port_config(void) { return "clock=emulated icache=0 dcache=0"; }

void port_trap(void) {
    const char *m = "DONE trap\n";
    while (*m) port_putc(*m++);
    port_exit(3);
}

void port_exit(int code) {
    /* semihosting SYS_EXIT (ADP_Stopped_ApplicationExit) */
    register uint32_t r0 __asm__("r0") = 0x18;
    register uint32_t r1 __asm__("r1") = code ? 0x20024u : 0x20026u;
    __asm__ volatile("bkpt 0xab" : : "r"(r0), "r"(r1) : "memory");
    for (;;) {}
}
