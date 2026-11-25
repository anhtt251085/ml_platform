/* QEMU virt (RV32IMAFC): NS16550 UART at 0x10000000, cycles from `mcycle`, exit via the SiFive test device.
   With -icount the counter tracks executed instructions: numbers are labelled `emulated`, never calibration. */
#include <stdint.h>
#include "../sai_hil_port.h"

#define UART_THR (*(volatile uint8_t *)0x10000000u)
#define UART_LSR (*(volatile uint8_t *)0x10000005u)
#define TEST_DEV (*(volatile uint32_t *)0x00100000u)

void port_init(void) {}

void port_putc(char c) {
    while (!(UART_LSR & 0x20u)) {}
    UART_THR = (uint8_t)c;
}

uint32_t port_cycles(void) {
    uint32_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

void port_exit(int code) {
    TEST_DEV = code ? ((uint32_t)code << 16) | 0x3333u : 0x5555u;
    for (;;) {}
}
