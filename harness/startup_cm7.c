/* Cortex-M startup for the SAI HIL harness: vector table, .data/.bss init, FPU enable. */
#include <stdint.h>
extern uint32_t _stack_top[], _data_start[], _data_end[], _data_load[], _bss_start[], _bss_end[];
int main(void);
void Reset_Handler(void);
static void Default_Handler(void) { for (;;) {} }
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));
void port_trap(void) __attribute__((weak));
void HardFault_Handler(void) { if (port_trap) port_trap(); for (;;) {} }

__attribute__((section(".vectors"), used)) static void (*const vectors[16])(void) = {
    (void (*)(void))_stack_top, Reset_Handler, Default_Handler, HardFault_Handler, Default_Handler, Default_Handler,
    Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler, 0, Default_Handler, SysTick_Handler,
};

__attribute__((section(".text.start"))) void Reset_Handler(void) {
    for (uint32_t *s = _data_load, *d = _data_start; d < _data_end;) *d++ = *s++;
    for (uint32_t *d = _bss_start; d < _bss_end;) *d++ = 0;
    *(volatile uint32_t *)0xE000ED88u |= 0xFu << 20;          /* CPACR: CP10, CP11 full access (FPU) */
    __asm__ volatile("dsb\n isb");
    main();
    for (;;) {}
}
