/* Sanken MD6605 EVB: cycles from `mcycle` (RV32IMAFC). The UART registers come from the vendor SDK
   (`MDproject` firmware); fill port_putc from it when the first EVB joins the farm. Until then this port is a
   build check: it links the package with the production flags so a farm build fails early, not on the bench. */
#include <stdint.h>
#include "../sai_hil_port.h"

#ifndef MD6605_UART_TX
#define MD6605_UART_TX(c) ((void)(c))   /* TODO(M4 bring-up): vendor SDK UART transmit */
#endif

void port_init(void) {}

void port_putc(char c) { MD6605_UART_TX(c); }

uint32_t port_cycles(void) {
    uint32_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

void port_exit(int code) { (void)code; for (;;) {} }
void port_quiet(int on) { (void)on; }
const char *port_config(void) { return "clock=66666666 icache=0 dcache=0"; }
void port_trap(void) { port_exit(3); }
