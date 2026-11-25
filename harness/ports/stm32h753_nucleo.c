/* NUCLEO-H753ZI: USART3 on PD8/PD9 (the ST-LINK V3 virtual COM port), 115200 8N1 from the reset HSI clock
   (64 MHz), cycles from DWT CYCCNT. Board bring-up per RM0433; written for the M4 board farm, to be validated on
   the first lab board (the QEMU ports are what CI runs). Clock it to 480 MHz in port_init to match SAI's
   measured sessions, or report the HSI clock with the result. */
#include <stdint.h>
#include "../sai_hil_port.h"

#define REG(a) (*(volatile uint32_t *)(a))
#define RCC_AHB4ENR REG(0x580244E0u)
#define RCC_APB1LENR REG(0x580244E8u)
#define GPIOD_MODER REG(0x58020C00u)
#define GPIOD_AFRH REG(0x58020C24u)
#define USART3_CR1 REG(0x40004800u)
#define USART3_BRR REG(0x4000480Cu)
#define USART3_ISR REG(0x4000481Cu)
#define USART3_TDR REG(0x40004828u)
#define DEMCR REG(0xE000EDFCu)
#define DWT_CTRL REG(0xE0001000u)
#define DWT_CYCCNT REG(0xE0001004u)
#define DWT_LAR REG(0xE0001FB0u)

void port_init(void) {
    RCC_AHB4ENR |= 1u << 3;                                 /* GPIOD */
    RCC_APB1LENR |= 1u << 18;                               /* USART3 */
    (void)RCC_APB1LENR;
    GPIOD_MODER = (GPIOD_MODER & ~(0xFu << 16)) | (0xAu << 16);          /* PD8, PD9 alternate function */
    GPIOD_AFRH = (GPIOD_AFRH & ~0xFFu) | 0x77u;                          /* AF7 = USART3 */
    USART3_BRR = 64000000u / 115200u;
    USART3_CR1 = (1u << 3) | 1u;                            /* TE, UE */
    DEMCR |= 1u << 24;                                      /* TRCENA */
    DWT_LAR = 0xC5ACCE55u;
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;
}

void port_putc(char c) {
    if (c == '\n') port_putc('\r');
    while (!(USART3_ISR & (1u << 7))) {}                    /* TXE */
    USART3_TDR = (uint32_t)(uint8_t)c;
}

uint32_t port_cycles(void) { return DWT_CYCCNT; }

void port_exit(int code) { (void)code; for (;;) { __asm__ volatile("wfi"); } }
