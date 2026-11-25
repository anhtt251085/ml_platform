/* Copyright (c) 2026 TLA Lab. All rights reserved. Proprietary and confidential; see LICENSE. */
/* NUCLEO-H753ZI port of the SAI HIL harness.

   Clock: PLL1 from HSI (64 MHz / 4 = 16 MHz reference, wide VCO), SYSCLK 480 MHz on rev V silicon (VOS0 via the
   SYSCFG overdrive) or 400 MHz on rev Y (its maximum), AXI/AHB = SYSCLK/2, APBx = AHB/2 - the configuration of
   SAI's NUCLEO-H753ZI measurement sessions. I-cache and D-cache on. If any step does not complete, the port stays on
   HSI 64 MHz; either way the CFG line reports the clock actually running, and the server only calibrates from runs
   at the reference clock (480 MHz).
   UART: USART3 on PD8/PD9 (ST-LINK V3 virtual COM port), 115200 8N1 from PCLK1. Cycles: DWT CYCCNT.
   Register sequences follow RM0433 (PWR 6.8.4, RCC 8.5/8.7, FLASH 4.3.8); validate on the first lab board. */
#include <stdint.h>
#include "../sai_hil_port.h"

#define REG(a) (*(volatile uint32_t *)(a))
/* PWR */
#define PWR_CSR1 REG(0x58024804u)
#define PWR_CR3 REG(0x5802480Cu)
#define PWR_D3CR REG(0x58024818u)
/* SYSCFG */
#define SYSCFG_PWRCR REG(0x5800042Cu)
/* RCC */
#define RCC_CR REG(0x58024400u)
#define RCC_CFGR REG(0x58024410u)
#define RCC_D1CFGR REG(0x58024418u)
#define RCC_D2CFGR REG(0x5802441Cu)
#define RCC_D3CFGR REG(0x58024420u)
#define RCC_PLLCKSELR REG(0x58024428u)
#define RCC_PLLCFGR REG(0x5802442Cu)
#define RCC_PLL1DIVR REG(0x58024430u)
#define RCC_AHB4ENR REG(0x580244E0u)
#define RCC_APB1LENR REG(0x580244E8u)
#define RCC_APB4ENR REG(0x580244F4u)
/* FLASH, DBGMCU */
#define FLASH_ACR REG(0x52002000u)
#define DBGMCU_IDCODE REG(0x5C001000u)
/* GPIOD, USART3 */
#define GPIOD_MODER REG(0x58020C00u)
#define GPIOD_AFRH REG(0x58020C24u)
#define USART3_CR1 REG(0x40004800u)
#define USART3_BRR REG(0x4000480Cu)
#define USART3_ISR REG(0x4000481Cu)
#define USART3_TDR REG(0x40004828u)
/* Core debug, SCB */
#define DEMCR REG(0xE000EDFCu)
#define DWT_CTRL REG(0xE0001000u)
#define DWT_CYCCNT REG(0xE0001004u)
#define DWT_LAR REG(0xE0001FB0u)
#define SCB_CCR REG(0xE000ED14u)
#define SCB_ICIALLU REG(0xE000EF50u)
#define SCB_CCSIDR REG(0xE000ED80u)
#define SCB_CSSELR REG(0xE000ED84u)
#define SCB_DCISW REG(0xE000EF60u)

#define HSI_HZ 64000000u
static uint32_t sysclk_hz = HSI_HZ;

static int wait_set(volatile uint32_t *r, uint32_t mask, uint32_t want) {
    for (uint32_t n = 0; n < 2000000u; n++)
        if ((*r & mask) == want) return 1;
    return 0;
}

/* Returns SYSCLK in Hz; on any failure leaves the core on HSI and returns 64 MHz. */
static uint32_t clock_init(void) {
    int rev_v = ((DBGMCU_IDCODE >> 16) & 0xFFFFu) == 0x2003u;   /* rev V: 480 MHz capable; rev Y: 400 MHz */
    uint32_t divn = rev_v ? 60u : 50u;                           /* 16 MHz x N = 960 / 800 MHz VCO */
    uint32_t target = rev_v ? 480000000u : 400000000u;

    /* 1. supply from the LDO (locks the supply configuration until the next power-on) */
    PWR_CR3 = (PWR_CR3 & ~0x7u) | (1u << 1);                     /* LDOEN, BYPASS = 0, SCUEN cleared */
    if (!wait_set(&PWR_CSR1, 1u << 13, 1u << 13)) return HSI_HZ; /* ACTVOSRDY */
    /* 2. VOS1, then VOS0 (overdrive) for 480 MHz */
    PWR_D3CR = (PWR_D3CR & ~(3u << 14)) | (3u << 14);
    if (!wait_set(&PWR_D3CR, 1u << 13, 1u << 13)) return HSI_HZ; /* VOSRDY */
    if (rev_v) {
        RCC_APB4ENR |= 1u << 1;                                  /* SYSCFGEN */
        (void)RCC_APB4ENR;
        SYSCFG_PWRCR |= 1u;                                      /* ODEN */
        if (!wait_set(&PWR_D3CR, 1u << 13, 1u << 13)) return HSI_HZ;
    }
    /* 3. flash wait states for AXI = SYSCLK/2 (240 MHz VOS0: 4 WS; 200 MHz VOS1: 2 WS), WRHIGHFREQ = 2 */
    uint32_t ws = rev_v ? 4u : 2u;
    FLASH_ACR = (FLASH_ACR & ~0x3Fu) | ws | (2u << 4);
    if ((FLASH_ACR & 0xFu) != ws) return HSI_HZ;
    /* 4. PLL1: HSI / 4 = 16 MHz (range 8-16 MHz), wide VCO, P = 2 (SYSCLK), Q = 4, R = 2 */
    RCC_PLLCKSELR = (RCC_PLLCKSELR & ~((0x3Fu << 4) | 3u)) | (4u << 4);
    RCC_PLLCFGR = (RCC_PLLCFGR & ~((3u << 2) | (1u << 1) | 1u)) | (3u << 2) | (1u << 16) | (1u << 17) | (1u << 18);
    RCC_PLL1DIVR = (divn - 1u) | ((2u - 1u) << 9) | ((4u - 1u) << 16) | ((2u - 1u) << 24);
    /* bus prescalers: D1CPRE /1, HPRE /2, D1PPRE /2, D2PPRE1 /2, D2PPRE2 /2, D3PPRE /2 */
    RCC_D1CFGR = (8u << 0) | (4u << 4);
    RCC_D2CFGR = (4u << 4) | (4u << 8);
    RCC_D3CFGR = (4u << 4);
    RCC_CR |= 1u << 24;                                          /* PLL1ON */
    if (!wait_set(&RCC_CR, 1u << 25, 1u << 25)) return HSI_HZ;   /* PLL1RDY */
    /* 5. switch SYSCLK to PLL1 */
    RCC_CFGR = (RCC_CFGR & ~7u) | 3u;
    if (!wait_set(&RCC_CFGR, 7u << 3, 3u << 3)) return HSI_HZ;   /* SWS = PLL1 */
    return target;
}

/* CMSIS SCB_EnableICache / SCB_EnableDCache, register level (no CMSIS dependency). */
static void enable_caches(void) {
    __asm__ volatile("dsb\n isb");
    SCB_ICIALLU = 0;
    __asm__ volatile("dsb\n isb");
    SCB_CCR |= 1u << 17;                                         /* IC */
    __asm__ volatile("dsb\n isb");
    SCB_CSSELR = 0;                                              /* L1 data cache */
    __asm__ volatile("dsb");
    uint32_t ccsidr = SCB_CCSIDR;
    uint32_t sets = (ccsidr >> 13) & 0x7FFFu, ways = (ccsidr >> 3) & 0x3FFu;
    do {
        uint32_t w = ways;
        do { SCB_DCISW = (sets << 5) | (w << 30); } while (w--);
    } while (sets--);
    __asm__ volatile("dsb");
    SCB_CCR |= 1u << 16;                                         /* DC */
    __asm__ volatile("dsb\n isb");
}

void port_init(void) {
    sysclk_hz = clock_init();
    uint32_t pclk1 = sysclk_hz == HSI_HZ ? HSI_HZ : sysclk_hz / 4u;   /* AHB = SYSCLK/2, APB1 = AHB/2 */
    RCC_AHB4ENR |= 1u << 3;                                      /* GPIOD */
    RCC_APB1LENR |= 1u << 18;                                    /* USART3 */
    (void)RCC_APB1LENR;
    GPIOD_MODER = (GPIOD_MODER & ~(0xFu << 16)) | (0xAu << 16);  /* PD8, PD9 alternate function */
    GPIOD_AFRH = (GPIOD_AFRH & ~0xFFu) | 0x77u;                  /* AF7 = USART3 */
    USART3_BRR = (pclk1 + 115200u / 2u) / 115200u;
    USART3_CR1 = (1u << 3) | 1u;                                 /* TE, UE */
    DEMCR |= 1u << 24;                                           /* TRCENA */
    DWT_LAR = 0xC5ACCE55u;
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;
    enable_caches();
}

void port_quiet(int on) { (void)on; }

const char *port_config(void) {
    switch (sysclk_hz) {
    case 480000000u: return "clock=480000000 icache=1 dcache=1";
    case 400000000u: return "clock=400000000 icache=1 dcache=1";
    default: return "clock=64000000 icache=1 dcache=1";
    }
}

void port_putc(char c) {
    if (c == '\n') port_putc('\r');
    while (!(USART3_ISR & (1u << 7))) {}                         /* TXE */
    USART3_TDR = (uint32_t)(uint8_t)c;
}

uint32_t port_cycles(void) { return DWT_CYCCNT; }

void port_trap(void) {
    const char *m = "DONE trap\n";
    while (*m) port_putc(*m++);
    port_exit(3);
}

void port_exit(int code) { (void)code; for (;;) { __asm__ volatile("wfi"); } }
