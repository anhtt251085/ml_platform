/* RV32 startup for the SAI HIL harness: stack, gp, FPU on, .data/.bss init, main. */
    .section .text.start, "ax"
    .globl Reset_Handler
Reset_Handler:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, _stack_top
    la t0, trap_entry          /* traps report "DONE trap" instead of jumping to an undefined address */
    csrw mtvec, t0
    li t0, 0x2000              /* mstatus.FS = initial: enable the F extension */
    csrs mstatus, t0
    la t0, _data_load
    la t1, _data_start
    la t2, _data_end
1:  bgeu t1, t2, 2f
    lw t3, 0(t0)
    sw t3, 0(t1)
    addi t0, t0, 4
    addi t1, t1, 4
    j 1b
2:  la t1, _bss_start
    la t2, _bss_end
3:  bgeu t1, t2, 4f
    sw zero, 0(t1)
    addi t1, t1, 4
    j 3b
4:  call main
5:  j 5b

    .align 4
trap_entry:
    la sp, _stack_top
    call port_trap
6:  j 6b
