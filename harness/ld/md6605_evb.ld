/* SAI HIL harness: Sanken MD6605: 128 KB ReRAM, 8 KB SRAM (6 KB + 2 KB stack, as the reference link.ld) */
ENTRY(Reset_Handler)
MEMORY {
  ROM (rx) : ORIGIN = 0x00000000, LENGTH = 128K
  RAM (rw) : ORIGIN = 0x20000000, LENGTH = 8K
}
STACK_SIZE = 0x800;
SECTIONS {
  .text : { KEEP(*(.vectors)) KEEP(*(.text.start)) *(.text*) *(.rodata*) *(.srodata*) . = ALIGN(4); } > ROM
  .data : { _data_start = .; *(.data*) *(.sdata*) . = ALIGN(4); _data_end = .; } > RAM AT > ROM
  _data_load = LOADADDR(.data);
  .bss (NOLOAD) : { _bss_start = .; *(.bss*) *(.sbss*) *(COMMON) . = ALIGN(8); _bss_end = .; } > RAM
  .stack (NOLOAD) : { . = ALIGN(16); _stack_bottom = .; . += STACK_SIZE; . = ALIGN(16); _stack_top = .; } > RAM
  PROVIDE(__global_pointer$ = _data_start + 0x800);
  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
