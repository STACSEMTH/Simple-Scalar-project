/*
 * sim-smt.c – SimpleScalar **Simultaneous Multi‑Threading (SMT)** core
 * --------------------------------------------------------------------
 * • Based on sim-outorder’s OOO pipeline, but supports up to MAX_HW_THREAD
 *   hardware threads that share the fetch/rename/issue/commit resources.
 * • Each thread has its own architectural state (regs, PC, rename map, ROB
 *   head/tail).  Pipeline structures are tagged with tid.
 * • Fetch policy = simple round‑robin (can be swapped for ICOUNT/etc.)
 * • Commit policy = oldest‑ready per‑thread (fair).
 *
 * Build:
 *   1) Add sim-smt.o to Makefile (do NOT add to common OBJS).
 *   2) $ make sim-smt
 *
 * TODO list:
 *   ▸ Complete ROB, LSQ, IQ structures with .tid fields
 *   ▸ Implement fetch_tid selection policy
 *   ▸ Wire rename maps per thread
 *   ▸ Commit/flush per tid
 *   ▸ Add statistics counters (per‑thread IPC, stalls, fairness)
 */

#include "host.h"
#include "misc.h"
#include "regs.h"
#include "memory.h"
#include "machine.h"
#include "options.h"
#include "stats.h"
#include "sim.h"
#include "loader.h"

static inline int get_latency(enum md_opcode op)
{
    /* ALU = 1cy, LD/ST = 2cy, MUL = 4cy, DIV = 12cy, FP‑div = 16cy */
    int opc = (op >> 26) & 0x3F;         /* Alpha primary opcode */
    if (opc == 0x10) return 4;      /* MULx */
    if (opc == 0x11) return 12;     /* DIVx */
    if ((opc >> 3) == 0x04) return 2; /* LDx/STx group */
    return 1; /* default ALU */
}
/* ===== PRF ready bitmap ============================================ */
#define PRF_NUM (MD_TOTAL_REGS * 8)
static char prf_ready[PRF_NUM];   /* 0 = not ready, 1 = ready */

/* ===== S I M  P A R A M E T E R S =================================== */
#define MAX_HW_THREAD 2           /* changeable with –threads N */
int sim_outorder_width = 4;
counter_t sim_max_insn = 0;
static counter_t fastfwd = 0;
static counter_t warmup = 0;
static counter_t sim_num_insn_tid[MAX_HW_THREAD] = {0};
/* ===== T H R E A D   C O N T E X T ================================== */
struct thread_ctx {
  md_addr_t pc;
  struct regs_t regs;             /* architectural registers */
  int rename_map[MD_TOTAL_REGS];  /* arch→phys map */
  int rob_head, rob_tail;         /* per‑thread ROB ptrs */
  /* add LSQ head/tail if separate */
  int active;                     /* 1 = running, 0 = halted */
};
static struct thread_ctx tctx[MAX_HW_THREAD];
static int num_hw_threads = 1;

/* ===== I F Q ( I N S T R U C T I O N    F E T C H    Q U E U E) ========================= */
struct ifq_entry {
  md_inst_t inst;				/* inst register */
  md_addr_t PC;		/* current PC, predicted next PC */
  int tid;
};
#define IFQ_SIZE 16
static struct ifq_entry IFQ[IFQ_SIZE];
static int ifq_head=0, ifq_tail=0;
/* ===== L S Q ( L O A D    S T O R E    Q U E U E) ========================= */
struct lsq_entry {
  md_addr_t addr;
  int tid;
  int is_load; /* 1 = LD, 0 = ST */
  tick_t done;
};
#define LSQ_SIZE 32
static struct lsq_entry LSQ[LSQ_SIZE];
static int lsq_head = 0, lsq_tail = 0;
static inline int is_load(enum md_opcode op){
    /* Alpha: primary opcode 0x08~0x0F = LDQ/LDSx … */
    int opc = (op >> 26) & 0x3F;
    return (opc >= 0x08 && opc <= 0x0F);
}
static inline int is_store(enum md_opcode op){
    int opc = (op >> 26) & 0x3F;
    return (opc >= 0x0C && opc <= 0x13);   /* STQ/STx */
}
/* ===== P I P E L I N E   S T R U C T U R E S ========================= */
struct rob_entry {
  int tid;
  int ready;
  md_inst_t inst;
  md_addr_t PC;
  tick_t done_cycle;
  int new_phys;
  int old_phys;
  int src1, src2;
  /* destination phys reg, exceptions, etc. */
};
#define ROB_SIZE 128
static struct rob_entry ROB[ROB_SIZE];
static int rob_head_global = 0, rob_tail_global = 0;

/* IQ, LSQ, BTB, caches… share similar tid tagging */

/* ===== S I M   G L O B A L S ======================================= */
static struct mem_t *mem = NULL;
static tick_t cycles = 0;

/* =====  F O R W A R D S ============================================ */
static void fetch_stage(void);
static void rename_stage(void);
static void issue_stage(void);
static void writeback_stage(void);
static void commit_stage(void);

/* =====  R E Q U I R E D   C A L L B A C K S ======================== */
void sim_reg_options(struct opt_odb_t *odb) {
  opt_reg_int(odb,
            /* switch  */ "-threads",
            /* help    */ "number of hardware threads to create",
            /* var     */ &num_hw_threads,
            /* default */ 2,
            /* print   */ NULL,          
            /* format  */ NULL);
  opt_reg_uint(odb, "-fastfwd", 
  "number of insts to fast-forward before timing", &fastfwd, 
  /*default*/0, NULL, NULL);
  opt_reg_uint(odb, "-max:inst", "maximum number of instructions to simulate",
  &sim_max_insn, /*default*/0, NULL, NULL);
}
void sim_check_options(struct opt_odb_t *odb, int argc, char **argv) {
  if (num_hw_threads < 1 || num_hw_threads > MAX_HW_THREAD)
    fatal("threads must be 1-%d", MAX_HW_THREAD);
}
void sim_reg_stats(struct stat_sdb_t *sdb) {
  stat_reg_counter(sdb, "sim_cycles", "total cycles", &cycles, 0, NULL);
  stat_reg_counter(sdb, "sim_num_insn", "instructions committed", &sim_num_insn, 0, NULL);
  
  for (int t = 0; t < num_hw_threads; ++t){
    char name[32], desc[64], expr[128];
    sprintf(name, "sim_num_insn_t%d", t);
    sprintf(desc, "commits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &sim_num_insn_tid[t], 0, NULL);

    sprintf(name, "IPC_t%d", t);
    sprintf(desc, "IPC, thread %d", t);
    sprintf(expr, "sim_num_insn_t%d / sim_cycles", t);
    stat_reg_formula(sdb, name, desc, expr, NULL);
  }
}

/* ===== P R F   F R E E ‑ L I S T =================================== */   
static int free_list[PRF_NUM];
static int free_head = 0, free_tail = 0;

static inline int prf_alloc(void) {
  if (free_head == free_tail) return -1;          /* NULL → rename stall */
  int p = free_list[free_head];
  free_head = (free_head + 1) % PRF_NUM;
  return p;
}
static inline void prf_free(int p) {
  free_list[free_tail] = p;
  free_tail = (free_tail + 1) % PRF_NUM;
}
/* ===== latency table (opcode → latency) ============================= */
void sim_init(void) {
  int pos = 0;
  mem = mem_create("mem");
  for (int t=0; t<MAX_HW_THREAD; ++t) tctx[t].active = 0;
  for (int i = MD_TOTAL_REGS; i < PRF_NUM; ++i) {
    free_list[pos++] = i;
    prf_ready[i] = 0;
  }
  for (int i = 0; i < MD_TOTAL_REGS; ++i) prf_ready[i] = 1;
  free_head = 0;
  free_tail = pos;
}
void sim_load_prog(char *fname, int argc, char **argv, char **envp) {
  /* load same binary into *each* thread context */
  for (int tid=0; tid<num_hw_threads; ++tid) {
    ld_load_prog(fname, argc, argv, envp, &tctx[tid].regs, mem, 0);
    tctx[tid].pc = tctx[tid].regs.regs_PC;
    tctx[tid].active = 1;
  }
}
void sim_uninit(void) {}
void sim_aux_stats(FILE *stream) {}
void sim_aux_config(FILE *stream) {
  fprintf(stream, "threads %d\n", num_hw_threads);
}

/* =====  M A I N   L O O P ========================================== */
void sim_main(void) {
  while (1) {
    /* stop when all threads inactive or –max:inst reached */
    int alive = 0; for (int t=0;t<num_hw_threads;++t) alive |= tctx[t].active;
    if (!alive) break;
    if (sim_max_insn && sim_num_insn >= sim_max_insn) break;
    /* ---- fast‑forward window ---- */
    if (fastfwd && warmup < fastfwd){
      /* skip timing stats but still drive pipeline */
      fetch_stage();
      rename_stage();
      warmup++;
      continue;
    }
    /* ---- timed region ---- */
    commit_stage();
    writeback_stage();
    issue_stage();
    rename_stage();
    fetch_stage();

    cycles++;
  }
}

/* =====  S T A G E   S T U B S ====================================== */
static int fetch_tid_rr = 0; /* round‑robin pointer */
static void fetch_stage(void) {
  /* simple rr policy */
  int fetched = 0;
  for (int tries = 0;tries<num_hw_threads && fetched < sim_outorder_width;++tries) {
    int tid = (fetch_tid_rr+tries)%num_hw_threads;
    if (!tctx[tid].active) continue;
    md_inst_t inst;
    mem_access(mem, Read, tctx[tid].pc, &inst, sizeof(md_inst_t));
    if ((rand() & 0xFF) == 0){ /* mispred */
        for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE) {
          if (ROB[i].new_phys != -1) prf_free(ROB[i].new_phys);   /* new PRF return */
        }
        for (int t = 0; t < num_hw_threads; ++t) {
          for (int r = 0; r < MD_TOTAL_REGS; ++r) tctx[t].rename_map[r] = r;   /* 간단히 초기 상태로 */
        }
        rob_head_global = rob_tail_global; /* flush ROB */
        ifq_head = ifq_tail; /* flush IFQ */
        lsq_head = lsq_tail; /* flush LSQ */
        continue;
      }
    /* ---------- push into IFQ ---------- */
    if (((ifq_tail+1)%IFQ_SIZE)==ifq_head) break;   /* IFQ full → stall */
    IFQ[ifq_tail].inst = inst;
    IFQ[ifq_tail].PC = tctx[tid].pc;
    IFQ[ifq_tail].tid = tid;
    ifq_tail = (ifq_tail+1)%IFQ_SIZE;

    tctx[tid].pc += sizeof(md_inst_t);
    fetched++;
  }
  fetch_tid_rr = (fetch_tid_rr+1)%num_hw_threads;
}

static inline int alpha_dest_reg(md_inst_t inst)
{
  return inst & 0x1F;   /* bits [4:0] */
}
static inline int alpha_writes_dest(md_inst_t inst)
{
  int opc = (inst >> 26) & 0x3F;      /* opcode[31:26] */
  /* 0x24–0x2F = STxx , 0x30–0x3F = branch·JSR, etc */
  if (opc >= 0x24 && opc <= 0x2F) return 0;   /* stores */
  if (opc >= 0x30 && opc <= 0x3F) return 0;   /* branches */
  return 1;                                   /* others use dest*/
}
static inline int alpha_src1(md_inst_t inst){ return (inst >> 21) & 0x1F; }
static inline int alpha_src2(md_inst_t inst){ return (inst >> 16) & 0x1F; }
static void rename_stage()  {
  int renamed = 0;
  while (renamed < sim_outorder_width) {
    if (ifq_head == ifq_tail) break; /* IFQ empty */

    if (((rob_tail_global+1)%ROB_SIZE)==rob_head_global) break; /* ROB full */
    
    /* pop IFQ */
    struct ifq_entry fe = IFQ[ifq_head];
    ifq_head = (ifq_head+1)%IFQ_SIZE;

    int tid = fe.tid;
    if (!tctx[tid].active) continue;

    /* enqueue ROB */
    struct rob_entry *re = &ROB[rob_tail_global];
    memset(re, 0, sizeof(*re));
    re->tid = tid;
    re->inst = fe.inst;
    re->PC = fe.PC;

    /* Source physical regs (true‑dep check later) */
    int a1 = alpha_src1(fe.inst);
    int a2 = alpha_src2(fe.inst);
    re->src1 = (a1 == 31) ? -1 : tctx[tid].rename_map[a1];
    re->src2 = (a2 == 31) ? -1 : tctx[tid].rename_map[a2];
    
    enum md_opcode op;
    MD_SET_OPCODE(op, fe.inst);
    if (alpha_writes_dest(fe.inst)){
      int arch = alpha_dest_reg(fe.inst);
      if (arch != 31) {
        int phys = prf_alloc();
        if (phys < 0) {
          /* rollback IFQ pointer and exit the loop */
          ifq_head = (ifq_head-1+IFQ_SIZE)%IFQ_SIZE;
          break;
        }

        re->new_phys = phys;
        re->old_phys = tctx[tid].rename_map[arch];
        tctx[tid].rename_map[arch] = phys;
        prf_ready[phys] = 0;
      } else {
        re->new_phys = re->old_phys = -1;
      }
    }
    if (is_load(op) || is_store(op)) {
      if (((lsq_tail + 1) % LSQ_SIZE) == lsq_head) {
        /* LSQ full → rename stall (rollback) */
        ifq_head = (ifq_head - 1 + IFQ_SIZE) % IFQ_SIZE;
        break;
      }
      LSQ[lsq_tail] = (struct lsq_entry){
        .addr = 0, .tid = tid, .is_load = is_load(op), .done = 0
      };
      lsq_tail = (lsq_tail + 1) % LSQ_SIZE;
    }
    rob_tail_global = (rob_tail_global+1)%ROB_SIZE;
    renamed++;
  }
}
static void issue_stage()   { 
  int issued = 0;
  for (int idx = rob_head_global; idx != rob_tail_global && issued < sim_outorder_width; idx = (idx+1)%ROB_SIZE){
    struct rob_entry *re = &ROB[idx];
    if (re->ready) continue; /* finished entry */
    if ((re->src1 != -1 && !prf_ready[re->src1]) || (re->src2 != -1 && !prf_ready[re->src2])) continue; /* true dependency stall */
    enum md_opcode op;
    MD_SET_OPCODE(op, ROB[idx].inst);           
    if (is_load(op) || is_store(op)) {
      re->done_cycle = cycles + 30; /* memory latency 30cy */
    } else {
      re->done_cycle = cycles + get_latency(op);  /* opcode → FU class */
    }
    issued++;
  }
}
static void writeback_stage(){ 
  for (int idx = rob_head_global; idx != rob_tail_global;idx = (idx+1)%ROB_SIZE){
    struct rob_entry *re = &ROB[idx];
    if (re->ready) continue;
    if (cycles < re->done_cycle) continue;

    /* mark destination PRF ready */
    if (re->new_phys != -1) prf_ready[re->new_phys] = 1;
    enum md_opcode op;
    MD_SET_OPCODE(op, re->inst);
    if (is_load(op) || is_store(op)) {
      lsq_head = (lsq_head + 1) % LSQ_SIZE;
    }
    re->ready = 1;
  }
}
static void commit_stage(void)
{
    int commits = 0;                                   /* # retired this cycle */

    /* Walk the ROB from the global head and stop when either:
       1) we have retired sim_outorder_width instructions, or
       2) we reach the tail (no more in‑flight entries).               */
    for (int idx = rob_head_global;
         idx != rob_tail_global && commits < sim_outorder_width;
         idx = (idx + 1) % ROB_SIZE)
    {
        /* Not completed yet → skip */
        if (!ROB[idx].ready)
            continue;

        int tid = ROB[idx].tid;

        /* Thread already halted → skip */
        if (!tctx[tid].active)
            continue;

        /* Free the old physical register, if any */
        if (ROB[idx].old_phys != -1)
            prf_free(ROB[idx].old_phys);

        /* Retire one architectural instruction */
        sim_num_insn++;
        sim_num_insn_tid[tid]++;
        commits++;

        /* Mark the ROB entry free by moving the global head pointer.
           Because we retire in program order, the current idx is
           always rob_head_global.                                     */
        ROB[idx].ready = 0;                       /* entry is now free  */
        rob_head_global = (idx + 1) % ROB_SIZE;   /* advance head       */
    }
}