/* SAI HIL harness: STM32H753 (flash 0x08000000, DTCM 0x20000000) */
ENTRY(Reset_Handler)
MEMORY {
  ROM (rx) : ORIGIN = 0x08000000, LENGTH = 2M
  RAM (rw) : ORIGIN = 0x20000000, LENGTH = 128K
}
STACK_SIZE = 0x4000;
SECTIONS {
  .text : { KEEP(*(.vectors)) KEEP(*(.text.start)) *(.text*) *(.rodata*) *(.srodata*) . = ALIGN(4); } > ROM
  .data : { _data_start = .; *(.data*) *(.sdata*) . = ALIGN(4); _data_end = .; } > RAM AT > ROM
  _data_load = LOADADDR(.data);
  .bss (NOLOAD) : { _bss_start = .; *(.bss*) *(.sbss*) *(COMMON) . = ALIGN(8); _bss_end = .; } > RAM
  .stack (NOLOAD) : { . = ALIGN(8); _stack_bottom = .; . += STACK_SIZE; _stack_top = .; } > RAM
  PROVIDE(__global_pointer$ = _data_start + 0x800);
  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
