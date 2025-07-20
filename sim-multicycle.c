/*
 * sim-multicycle.c – SimpleScalar **multi‑cycle, non‑pipelined** CPU model
 * ------------------------------------------------------------------------
 * Minimal demonstrator that links cleanly with the generic SimpleScalar
 * framework (main.c).  Stub functions are provided for the callbacks that
 * main.c expects: sim_init, sim_load_prog, sim_uninit, sim_aux_config,
 * sim_aux_stats.  Only one real instruction (ADD/ADDU/ADDQ) is executed; all
 * other opcodes trigger fatal().
 */

#include "host.h"
#include "misc.h"
#include "regs.h"        /* struct regs_t */
#include "memory.h"      /* mem_create(), mem_access() */
#include "machine.h"     /* md_inst_t, opcode macros */
#include "options.h"
#include "stats.h"
#include "sim.h"
#include "loader.h"      /* ld_load_prog() prototype */

/* ======== global state (each simulator defines its own) ========== */
struct regs_t regs;          /* defined in regs.c */
struct mem_t *mem = NULL;           /* defined in main/common */
counter_t   sim_max_insn = 0;    /* global in main.c via sim.h */       /* --max:inst (0 == no limit) */

/* ======== multicycle FSM ======== */
enum mc_stage_t { MC_IF, MC_ID, MC_EX, MC_MEM, MC_WB };
static enum mc_stage_t cur_stage;

static md_inst_t IF_ir, ID_ir;
static md_addr_t pc, IF_NPC;
static enum md_opcode op;
static tick_t cycles = 0;

/* forward */
static void fsm_step(void);

/********************************************************************/
/* Required simulator callback stubs                                */
/********************************************************************/
void sim_reg_options(struct opt_odb_t *odb) { /* no new options */ }
void sim_check_options(struct opt_odb_t *odb, int argc, char **argv) {}
void sim_reg_stats(struct stat_sdb_t *sdb) {
  stat_reg_counter(sdb, "sim_cycles", "total cycles", &cycles, 0, NULL);
  stat_reg_counter(sdb, "sim_num_insn", "instructions executed", &sim_num_insn, 0, NULL);
}

/* called very early by main.c */
void sim_init(void) {
  mem = mem_create("mem");
}

/* program loader hook */
void sim_load_prog(char *fname, int argc, char **argv, char **envp) {
  /* use stock loader */
  ld_load_prog(fname, argc, argv, envp, &regs, mem, 0);
}

void sim_uninit(void) { /* nothing */ }
void sim_aux_stats(FILE *stream) { /* no extra stats */ }
void sim_aux_config(FILE *stream) { /* none */ }

/********************************************************************/
/* main loop                                                        */
/********************************************************************/
void sim_main(void) {
  pc = regs.regs_PC;
  IF_NPC = pc;
  cur_stage = MC_IF;
  cycles = 0;
  sim_num_insn = 0;

  while (1) {
    if (sim_max_insn && sim_num_insn >= sim_max_insn) break;
    if (pc == 0) break;    /* simplistic halt */

    fsm_step();
    cycles++;
  }
}

/********************************************************************/
/* FSM implementation                                               */
/********************************************************************/
static void fsm_step(void) {
  switch (cur_stage) {
  case MC_IF:
    pc = IF_NPC;
    mem_access(mem, Read, pc, &IF_ir, sizeof(md_inst_t));
    IF_NPC = pc + sizeof(md_inst_t);
    cur_stage = MC_ID;
    break;

  case MC_ID:
    ID_ir = IF_ir;
    MD_SET_OPCODE(op, ID_ir);
    cur_stage = MC_EX;
    break;

  case MC_EX:
#if defined(TARGET_PISA)
    if (op == ADDU) {
      regs.regs_R[MD_RD(ID_ir)] = regs.regs_R[MD_RS(ID_ir)] + regs.regs_R[MD_RT(ID_ir)];
      cur_stage = MC_WB;
    } else
#elif defined(TARGET_ALPHA)
    /* Alpha: macros RA/RB/RC refer to local variable 'inst',
       so extract ID_ir into that name first. */
    {
      md_inst_t inst = ID_ir; /* for RA/RB/RC field macros */
      if (op == ADDQ) {
        regs.regs_R[RC] = regs.regs_R[RA] + regs.regs_R[RB];
        cur_stage = MC_WB;
      } else {
        fatal("sim-multicycle: opcode %d not implemented", op);
      }
    }
#else
#endif
    {
      fatal("sim-multicycle: opcode %d not implemented", op);
    }
    break;

  case MC_MEM:
    /* not yet implemented */
    cur_stage = MC_WB;
    break;

  case MC_WB:
    sim_num_insn++;
    cur_stage = MC_IF;
    break;
  }
}
