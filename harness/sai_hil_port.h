/* Board port of the SAI HIL harness. One .c per board in harness/ports/. */
#ifndef SAI_HIL_PORT_H
#define SAI_HIL_PORT_H
#include <stdint.h>

void port_init(void);              /* clocks, UART, cycle counter */
void port_putc(char c);            /* blocking UART transmit */
uint32_t port_cycles(void);        /* free-running 32-bit core cycle counter (DWT CYCCNT, mcycle) */
void port_exit(int code) __attribute__((noreturn));   /* stop (QEMU: exit; board: spin) */

#endif
