/* SAI HIL harness: QEMU virt RV32 (RAM at 0x80000000) */
ENTRY(Reset_Handler)
MEMORY {
  ROM (rx) : ORIGIN = 0x80000000, LENGTH = 8M
  RAM (rw) : ORIGIN = 0x80800000, LENGTH = 8M
}
STACK_SIZE = 0x8000;
SECTIONS {
  .text : { KEEP(*(.vectors)) KEEP(*(.text.start)) *(.text*) *(.rodata*) *(.srodata*) . = ALIGN(4); } > ROM
  .data : { _data_start = .; *(.data*) *(.sdata*) . = ALIGN(4); _data_end = .; } > RAM AT > ROM
  _data_load = LOADADDR(.data);
  .bss (NOLOAD) : { _bss_start = .; *(.bss*) *(.sbss*) *(COMMON) . = ALIGN(8); _bss_end = .; } > RAM
  .stack (NOLOAD) : { . = ALIGN(16); _stack_bottom = .; . += STACK_SIZE; . = ALIGN(16); _stack_top = .; } > RAM
  PROVIDE(__global_pointer$ = _data_start + 0x800);
  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
