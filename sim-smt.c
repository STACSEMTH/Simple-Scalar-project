/***********************************************************************
 * sim-smt.c – SimpleScalar **Simultaneous Multi‑Threading (SMT)** core
 * --------------------------------------------------------------------
 *  • Out‑of‑order SMT simulator for the PISA/Alpha binaries distributed
 *    with SimpleScalar 3.0.
 *  • Shares fetch/rename/issue/commit resources across up to
 *    MAX_HW_THREAD hardware threads.
 *  • No external headers or libraries are required beyond stock
 *    SimpleScalar.  Build:
 *        1) add “sim-smt.o” to the Makefile’s PROGS section
 *        2) $ make sim-smt
 **********************************************************************/
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "host.h"
#include "misc.h"
#include "regs.h"
#include "memory.h"
#include "machine.h"
#include "options.h"
#include "stats.h"
#include "sim.h"
#include "loader.h"
#ifndef CALL_PAL
#define CALL_PAL 0x00
#endif
#ifndef OSF_SYS_exit
#define OSF_SYS_exit 1  /* Alpha exit system call */
#endif
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
#define MAX_HW_THREAD 8           /* changeable with –threads N */
int sim_outorder_width = 4;
counter_t sim_max_insn = 0;
static counter_t fastfwd = 0;
static counter_t warmup = 0;
static counter_t sim_num_insn_tid[MAX_HW_THREAD] = {0};
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
/* ===== S I M   G L O B A L S ======================================= */
static struct mem_t *mem = NULL;
static tick_t cycles = 0;
static counter_t lsq_store_forwards = 0;    /* store-to-load forwards */
static counter_t lsq_load_violations = 0;   /* load-store violations */
static counter_t lsq_addr_conflicts = 0;   
static counter_t lsq_partial_forwards = 0;
static void flush_thread(int tid);
static inline int addr_overlap(md_addr_t, int, md_addr_t, int);
/* ===== T H R E A D   C O N T E X T ================================== */
struct thread_ctx {
  md_addr_t pc;
  struct regs_t regs;             /* architectural registers */
  int rename_map[MD_TOTAL_REGS];  /* arch→phys map */
  int rob_head, rob_tail;         /* per‑thread ROB ptrs */
  unsigned long long seq;
  int icount; /* fetch scoreboard */
  int active;                     /* 1 = running, 0 = halted */

  /* Speculation state */
  md_addr_t speculative_pc;
  int speculation_depth;
  tick_t last_flush_cycle;
  counter_t flush_count;

  /* Per-thread performance counters */
  counter_t branches_executed;
  counter_t branches_mispredicted;
  counter_t icache_accesses;
  counter_t icache_misses;
  counter_t dcache_accesses;
  counter_t dcache_misses;
};
static struct thread_ctx tctx[MAX_HW_THREAD];
static int num_hw_threads = 1;
static int smart_thread_selection(int *fetch_order, int *fetch_count);
/* ===== I F Q ( I N S T R U C T I O N    F E T C H    Q U E U E) ========================= */
struct ifq_entry {
  md_inst_t inst;				/* inst register */
  md_addr_t PC;		/* current PC, predicted next PC */
  int tid;
};
#define IFQ_SIZE 16
static struct ifq_entry IFQ[IFQ_SIZE];
static int ifq_head=0, ifq_tail=0;
/* ===== M E M O R Y     D E P E N D E N C E     P R E D I C T I O N ======== */
typedef struct {
  md_addr_t load_pc;
  md_addr_t store_pc;
  int confidence;
  tick_t last_update;
} mem_dep_entry_t;

#define MEM_DEP_TABLE_SIZE 256
static mem_dep_entry_t mem_dep_table[MEM_DEP_TABLE_SIZE];
static int enable_dynamic_partitioning = 1;
static int enable_stride_prefetcher = 1;
static int enable_runahead_execution = 0;
static int memory_dependency_prediction = 1;
static void update_memory_dependence_predictor(md_addr_t load_pc, md_addr_t store_pc, 
  int violation_occurred) {
  if (!memory_dependency_prediction) return;                                               
  unsigned idx = (load_pc ^ store_pc) % MEM_DEP_TABLE_SIZE;
  
  mem_dep_entry_t *entry = &mem_dep_table[idx];
  
  if (entry->load_pc != load_pc || entry->store_pc != store_pc) {
    /* New entry */
    entry->load_pc = load_pc;
    entry->store_pc = store_pc;
    entry->confidence = violation_occurred ? 4 : 1;
  } else {
    /* Update existing entry */
    if (violation_occurred) {
      entry->confidence = MIN(entry->confidence + 2, 7);
    } else {
      entry->confidence = MAX(entry->confidence - 1, 0);
    }
  }
  entry->last_update = cycles;
}

static int predict_memory_dependence(md_addr_t load_pc, md_addr_t store_pc) {
  if (!memory_dependency_prediction) return 0;
  unsigned idx = (load_pc ^ store_pc) % MEM_DEP_TABLE_SIZE;
  mem_dep_entry_t *entry = &mem_dep_table[idx];
  
  if (entry->load_pc == load_pc && entry->store_pc == store_pc) {
    if (cycles - entry->last_update > 10000) {
      entry->confidence = MAX(entry->confidence-1, 0);
    }
   return entry->confidence > 3; /* Predict dependence if confident */
  }
  return 0; /* No prediction available */
}
/* ===== I Q ( I N S T R U C T I O N     Q U E U E) ========================= */
struct iq_entry {
    md_inst_t inst;
    int rob_idx, tid;
    int src1;
    int src2;
    int dst;
    unsigned is_load:1, is_store:1;
    tick_t done;
    char issued, ready;
};
#define IQ_SIZE 32
struct iq_entry IQ[IQ_SIZE];
static int iq_cnt;
/* ===== L S Q ( L O A D    S T O R E    Q U E U E) ========================= */
struct lsq_entry {
  md_addr_t addr;
  md_addr_t vaddr; /* virtual address */
  union
  {
    __quad_t as_quad;
    dfloat_t as_double;
    sfloat_t as_float;
    word_t as_word;
    half_t as_half;
    byte_t as_byte;
  } data;
  int rob_idx, tid;
  unsigned size:4;
  unsigned is_load:1;
  unsigned is_store:1;
  unsigned addr_ready:1;  
  unsigned data_ready:1;  /* store data prepared */
  unsigned forwarded:1;    
  unsigned committed:1;   /* store commited */
  tick_t addr_ready_cycle;  
  tick_t data_ready_cycle;
  char load;
  tick_t done;
};
#define LSQ_SIZE 32
static struct lsq_entry LSQ[LSQ_SIZE];
static int lsq_head = 0, lsq_tail = 0;
static inline int is_load(enum md_opcode op){
    /* Alpha: primary opcode 0x08~0x0F = LDQ/LDSx … */
    return (MD_OP_FLAGS(op) & F_LOAD) != 0;
}
static inline int is_store(enum md_opcode op){
    return (MD_OP_FLAGS(op) & F_STORE) != 0;   /* STQ/STx */
}
#define SLAP_SIZE 1024
struct slap_entry {
  md_addr_t pc;
  md_addr_t last_addr;
  int confidence;
  int hits, misses;
};
static struct slap_entry slap[SLAP_SIZE];

typedef enum {
  FORWARD_NONE = 0,
  FORWARD_FULL = 1,
  FORWARD_PARTIAL = 2,
  FORWARD_CONFLICT = 3
} forward_result_t;
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
  unsigned is_load:1;
  unsigned is_store:1;
  unsigned long long seq;
  /* destination phys reg, exceptions, etc. */
};
#define ROB_SIZE 128
static struct rob_entry ROB[ROB_SIZE];
static int rob_head_global = 0, rob_tail_global = 0;

/* IQ, LSQ, BTB, caches… share similar tid tagging */
/* ===== F O R W A R D I N G ======================================= */
static forward_result_t check_store_forwarding(int load_lsq_idx) {
  struct lsq_entry *load = &LSQ[load_lsq_idx];
  
  if (!load->is_load || !load->addr_ready) return FORWARD_NONE;
  
  /* Check store-load aliasing predictor */
  unsigned slap_idx = (load->rob_idx) % SLAP_SIZE;
  struct slap_entry *se = &slap[slap_idx];
  
  for (int i = (load_lsq_idx - 1 + LSQ_SIZE) % LSQ_SIZE; 
        i != lsq_head; 
        i = (i - 1 + LSQ_SIZE) % LSQ_SIZE) {
      
    struct lsq_entry *store = &LSQ[i];
    
    if (store->tid != load->tid) continue;
    if (!store->is_store || !store->addr_ready) continue;
    
    if (addr_overlap(store->addr, store->size, load->addr, load->size)) {
      
      /* Full forwarding case */
      if (store->addr == load->addr && 
          store->size == load->size && 
          store->data_ready) {
          
        /* Simulate forwarding network delay */
        int bypass_latency = 1;
        if (store->size > 8) bypass_latency = 2; /* wider data */
        
        load->data = store->data;
        load->forwarded = 1;
        load->done = cycles + bypass_latency;
        
        /* Update predictor */
        se->hits++;
        se->confidence = MIN(se->confidence + 1, 7);
        
        lsq_store_forwards++;
        return FORWARD_FULL;
      }
      
      /* Partial forwarding case */
      else if (store->data_ready) {
        md_addr_t overlap_start = MAX(store->addr, load->addr);
        md_addr_t overlap_end = MIN(store->addr + store->size, 
                                  load->addr + load->size);
        int overlap_size = overlap_end - overlap_start;
        
        if (overlap_size > 0 && overlap_size < load->size) {
          /* Partial data available - need memory for rest */
          load->done = cycles + 30; /* memory latency */
          lsq_partial_forwards++;
          return FORWARD_PARTIAL;
        }
      }
      
      /* Address conflict but data not ready */
      else {
        se->misses++;
        se->confidence = MAX(se->confidence - 1, 0);
        lsq_addr_conflicts++;
        return FORWARD_CONFLICT;
      }
    }
  }
  
  /* No forwarding possible */
  load->done = cycles + 30; /* memory access latency */
  return FORWARD_NONE;
}

static void check_load_store_violations(void) {
  /* for commited store, check conflicts with loads */
  for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE) {
    struct rob_entry *re = &ROB[i];
    
    if (!re->ready || !re->is_store) continue;
    
    int store_lsq_idx = -1;
    for (int j = lsq_head; j != lsq_tail; j = (j + 1) % LSQ_SIZE) {
      if (LSQ[j].rob_idx == i) {
        store_lsq_idx = j;
        break;
      }
    }
    
    if (store_lsq_idx == -1) continue;
    struct lsq_entry *store = &LSQ[store_lsq_idx];
    
    if (!store->addr_ready) continue;
    
    for (int j = (store_lsq_idx + 1) % LSQ_SIZE; 
        j != lsq_tail; 
        j = (j + 1) % LSQ_SIZE) {
      struct lsq_entry *load = &LSQ[j];
      
      if (load->tid != store->tid || !load->is_load) continue;
      if (!load->addr_ready || load->done > cycles) continue;
      
      if (addr_overlap(store->addr, store->size, load->addr, load->size)) {
        lsq_load_violations++;
        
        /* Update memory dependence predictor */
        struct rob_entry *load_re = &ROB[load->rob_idx];
        update_memory_dependence_predictor(load_re->PC, re->PC, 1);

        /* Handle violation */
        load_re->ready = 0; 
        load->done = cycles + 30; 
        
        tctx[load->tid].branches_mispredicted++; /* Count as misprediction */
        flush_thread(load->tid);
        return;
      }
    }
  }
}
/* =====  B R A C N C H ============================================== */

/* 2-bit saturating counter */
typedef enum {
  STRONGLY_NOT_TAKEN = 0,
  WEAKLY_NOT_TAKEN = 1,
  WEAKLY_TAKEN = 2,
  STRONGLY_TAKEN = 3
} branch_state_t;

/* Branch Predictor Structures */
typedef struct {
  branch_state_t state;
  md_addr_t tag;
  int valid;
} local_predictor_entry_t;

typedef struct {
    branch_state_t state;
} global_predictor_entry_t;

typedef struct {
  md_addr_t target;
  md_addr_t tag;
  int valid;
} btb_entry_t;

#define LOCAL_PRED_SIZE 1024
#define GLOBAL_PRED_SIZE 4096
#define GLOBAL_HIST_LEN 12
#define BTB_SIZE 512

static local_predictor_entry_t local_predictor[LOCAL_PRED_SIZE];
static global_predictor_entry_t global_predictor[GLOBAL_PRED_SIZE];
static btb_entry_t btb[BTB_SIZE];
static unsigned global_history = 0;

/* Branch Predictor Statistics */
static counter_t bp_lookups = 0;
static counter_t bp_correct = 0;
static counter_t bp_mispred = 0;
static counter_t btb_hits = 0;
static counter_t btb_misses = 0;

static void init_branch_predictor() {
  for (int i = 0; i < LOCAL_PRED_SIZE; i++) {
    local_predictor[i].state = WEAKLY_NOT_TAKEN;
    local_predictor[i].valid = 0;
  }
  
  for (int i = 0; i < GLOBAL_PRED_SIZE; i++) {
    global_predictor[i].state = WEAKLY_NOT_TAKEN;
  }
  
  for (int i = 0; i < BTB_SIZE; i++) {
    btb[i].valid = 0;
    btb[i].tag = 0;
    btb[i].target = 0;
  }
    
  global_history = 0;
}

static int predict_branch(md_addr_t pc, md_addr_t *pred_target) {
  bp_lookups++;
  
  /* Local predictor lookup */
  unsigned local_idx = (pc >> 2) % LOCAL_PRED_SIZE;
  local_predictor_entry_t *local_entry = &local_predictor[local_idx];
  
  /* Global predictor lookup */
  unsigned global_idx = ((pc >> 2) ^ global_history) % GLOBAL_PRED_SIZE;
  global_predictor_entry_t *global_entry = &global_predictor[global_idx];
  
  /* Hybrid selection (choose global for now) */
  int prediction = (global_entry->state >= WEAKLY_TAKEN) ? 1 : 0;
  
  /* BTB lookup for target */
  unsigned btb_idx = (pc >> 2) % BTB_SIZE;
  btb_entry_t *btb_entry = &btb[btb_idx];
  
  if (btb_entry->valid && btb_entry->tag == pc) {
    *pred_target = btb_entry->target;
    btb_hits++;
    return prediction;
  } else {
    *pred_target = pc + 4; /* Default: fall-through */
    btb_misses++;
    return 0; /* Predict not taken if not in BTB */
  }
}

static void update_branch_predictor(md_addr_t pc, int taken, md_addr_t actual_target) {
  /* Update local predictor */
  unsigned local_idx = (pc >> 2) % LOCAL_PRED_SIZE;
  local_predictor_entry_t *local_entry = &local_predictor[local_idx];
  
  if (!local_entry->valid || local_entry->tag != pc) {
    local_entry->tag = pc;
    local_entry->valid = 1;
    local_entry->state = taken ? WEAKLY_TAKEN : WEAKLY_NOT_TAKEN;
  } else {
    if (taken) {
      if (local_entry->state < STRONGLY_TAKEN) local_entry->state++;
    } else {
      if (local_entry->state > STRONGLY_NOT_TAKEN) local_entry->state--;
    }
  }
  
  /* Update global predictor */
  unsigned global_idx = ((pc >> 2) ^ global_history) % GLOBAL_PRED_SIZE;
  global_predictor_entry_t *global_entry = &global_predictor[global_idx];
  
  if (taken) {
    if (global_entry->state < STRONGLY_TAKEN) global_entry->state++;
  } else {
    if (global_entry->state > STRONGLY_NOT_TAKEN) global_entry->state--;
  }
  
  /* Update BTB */

  unsigned btb_idx = (pc >> 2) % BTB_SIZE;
  btb_entry_t *btb_entry = &btb[btb_idx];
  btb_entry->tag = pc;
  btb_entry->target = actual_target;
  btb_entry->valid = 1;

  /* Update global history */
  global_history = ((global_history << 1) | (taken ? 1 : 0)) & 
                    ((1 << GLOBAL_HIST_LEN) - 1);
}
/* Enhanced Branch Resolution */
static int resolve_branch(md_inst_t inst, md_addr_t pc, struct regs_t *regs, md_addr_t *target) {
  int opcode = (inst >> 26) & 0x3F;
  int ra = (inst >> 21) & 0x1F;  /* source register */
  int displacement = (int)((inst & 0x1FFFFF) << 2);
  if (displacement & 0x200000) displacement |= 0xFFC00000; /* sign extend */
  
  *target = pc + 4 + displacement;
  
  switch (opcode) {
    case 0x30: /* BR - unconditional */
      return 1;
        
    case 0x34: /* BSR - branch subroutine */
      return 1;
        
    case 0x38: /* BLBC - branch if low bit clear */
      return (ra == 31) ? 0 : !(regs->regs_R[ra] & 1);
        
    case 0x39: /* BEQ - branch if equal to zero */
      return (ra == 31) ? 1 : (regs->regs_R[ra] == 0);
        
    case 0x3A: /* BLT - branch if less than zero */
      return (ra == 31) ? 0 : ((sword_t)regs->regs_R[ra] < 0);
        
    case 0x3B: /* BLE - branch if less than or equal to zero */
      return (ra == 31) ? 1 : ((sword_t)regs->regs_R[ra] <= 0);
        
    case 0x3C: /* BLBS - branch if low bit set */
      return (ra == 31) ? 0 : (regs->regs_R[ra] & 1);
        
    case 0x3D: /* BNE - branch if not equal to zero */
      return (ra == 31) ? 0 : (regs->regs_R[ra] != 0);
        
    case 0x3E: /* BGE - branch if greater than or equal to zero */
      return (ra == 31) ? 1 : ((sword_t)regs->regs_R[ra] >= 0);
        
    case 0x3F: /* BGT - branch if greater than zero */
      return (ra == 31) ? 0 : ((sword_t)regs->regs_R[ra] > 0);
        
    /* Floating point branches */
    case 0x31: /* FBEQ */
      return (ra == 31) ? 1 : (regs->regs_F.d[ra] == 0.0);
    case 0x32: /* FBLT */
      return (ra == 31) ? 0 : (regs->regs_F.d[ra] < 0.0);
    case 0x33: /* FBLE */
      return (ra == 31) ? 1 : (regs->regs_F.d[ra] <= 0.0);
    case 0x35: /* FBNE */
      return (ra == 31) ? 0 : (regs->regs_F.d[ra] != 0.0);
    case 0x36: /* FBGE */
      return (ra == 31) ? 1 : (regs->regs_F.d[ra] >= 0.0);
    case 0x37: /* FBGT */
        return (ra ==  31) ? 0 : (regs->regs_F.d[ra] > 0.0);
        
    default:
      return 0; /* Not a branch */
  }
}
static void check_branch_misprediction(void) {
  // Check committed branches for mispredictions
  for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE) {
    struct rob_entry *re = &ROB[i];
    if (!re->ready) continue;
    
    enum md_opcode op;
    MD_SET_OPCODE(op, re->inst);
    
    if (MD_OP_FLAGS(op) & F_CTRL) {
      md_addr_t actual_target;
      int actual_taken = resolve_branch(re->inst, re->PC, &tctx[re->tid].regs, &actual_target);
      
      // Get prediction
      md_addr_t pred_target;
      int pred_taken = predict_branch(re->PC, &pred_target);
      
      if (pred_taken != actual_taken || 
          (actual_taken && pred_target != actual_target)) {
        bp_mispred++;
        tctx[re->tid].branches_mispredicted++;
        
        // Flush pipeline
        flush_thread(re->tid);
        tctx[re->tid].pc = actual_taken ? actual_target : (re->PC + 4);
        return;
      } else {
        bp_correct++;
      }
      
      // Update predictor
      update_branch_predictor(re->PC, actual_taken, actual_target);
    }
  }
}
/* ===== ENHANCED ADDRESS GENERATION AND LSQ ======================= */

/* Address Generation Unit */
typedef struct {
  md_addr_t addr;
  int rob_idx;
  int tid;
  tick_t ready_cycle;
  int valid;
} agu_entry_t;

#define AGU_SIZE 8
static agu_entry_t AGU[AGU_SIZE];
static void enhanced_lsq_access(struct lsq_entry *lsq, int lsq_idx);
/* Enhanced LSQ with proper address generation timing */
static void address_generation_stage() {
  /* Process Address Generation Unit */
  for (int i = 0; i < AGU_SIZE; i++) {
    agu_entry_t *agu = &AGU[i];
    if (!agu->valid || cycles < agu->ready_cycle) continue;
    
    /* Find corresponding LSQ entry */
    for (int j = lsq_head; j != lsq_tail; j = (j + 1) % LSQ_SIZE) {
      if (LSQ[j].rob_idx == agu->rob_idx && LSQ[j].tid == agu->tid) {
        LSQ[j].vaddr = agu->addr;
        LSQ[j].addr = agu->addr;
        LSQ[j].addr_ready = 1;
        LSQ[j].addr_ready_cycle = cycles;
        
        /* 개선된 LSQ 액세스 */
        if (LSQ[j].is_load) enhanced_lsq_access(&LSQ[j], j);
        break;
      }
    }
    
    /* Clear AGU entry */
    agu->valid = 0;
  }
}
/* ===== T L B ( T R A N S L A T I O N     L O O K A S I D E     B U F F E R) ===== */
typedef struct tlb_entry {
    md_addr_t vpn;          /* Virtual Page Number */
    md_addr_t ppn;          /* Physical Page Number */
    int valid;
    int dirty;
    int thread_id;
    tick_t last_access;
    struct tlb_entry *next; /* LRU chain */
} tlb_entry_t;

typedef struct tlb {
    tlb_entry_t *entries;
    int size;
    int assoc;
    counter_t hits;
    counter_t misses;
    counter_t page_faults;
} tlb_t;

#define TLB_SIZE 64
#define PAGE_SIZE 4096
#define PAGE_SHIFT 12

static tlb_t *dtlb = NULL;  /* Data TLB */
static tlb_t *itlb = NULL;  /* Instruction TLB */

/* TLB Statistics per Thread */
static counter_t dtlb_hits_tid[MAX_HW_THREAD] = {0};
static counter_t dtlb_misses_tid[MAX_HW_THREAD] = {0};
static counter_t itlb_hits_tid[MAX_HW_THREAD] = {0};
static counter_t itlb_misses_tid[MAX_HW_THREAD] = {0};

static tlb_t* tlb_create(int size, int assoc) {
    tlb_t *tlb = (tlb_t*)calloc(1, sizeof(tlb_t));
    tlb->entries = (tlb_entry_t*)calloc(size, sizeof(tlb_entry_t));
    tlb->size = size;
    tlb->assoc = assoc;
    return tlb;
}

typedef enum {
    TLB_HIT = 0,
    TLB_MISS = 1,
    TLB_PAGE_FAULT = 2
} tlb_access_result_t;

static tlb_access_result_t tlb_access(tlb_t *tlb, md_addr_t vaddr, 
                                     int thread_id, md_addr_t *paddr) {
    counter_t *hits_counter = thread_id ? &dtlb_hits_tid[thread_id] : &itlb_hits_tid[thread_id];
    counter_t *misses_counter = thread_id ? &dtlb_misses_tid[thread_id] : &itlb_misses_tid[thread_id];
    
    md_addr_t vpn = vaddr >> PAGE_SHIFT;
    int set_idx = vpn % (tlb->size / tlb->assoc);
    
    /* TLB lookup */
    for (int i = 0; i < tlb->assoc; i++) {
        tlb_entry_t *entry = &tlb->entries[set_idx * tlb->assoc + i];
        if (entry->valid && entry->vpn == vpn && entry->thread_id == thread_id) {
            /* TLB hit */
            *paddr = (entry->ppn << PAGE_SHIFT) | (vaddr & (PAGE_SIZE - 1));
            entry->last_access = cycles;
            tlb->hits++;
            /* Update per-thread counters based on TLB type */
            if (tlb == itlb) {
                itlb_hits_tid[thread_id]++;
            } else if (tlb == dtlb) {
                dtlb_hits_tid[thread_id]++;
            }
            return TLB_HIT;
        }
    }
    
    /* TLB miss - page table walk */
    tlb->misses++;
    /* Update per-thread counters based on TLB type */
    if (tlb == itlb) {
        itlb_misses_tid[thread_id]++;
    } else if (tlb == dtlb) {
        dtlb_misses_tid[thread_id]++;
    }
    
    /* Realistic page table walk latency */
    int ptw_latency = 20 + (rand() % 10); /* 20-30 cycles */
    
    /* Simple but realistic translation */
    md_addr_t ppn = (vpn ^ 0x80000) + thread_id * 0x1000; /* Thread separation */
    *paddr = (ppn << PAGE_SHIFT) | (vaddr & (PAGE_SIZE - 1));
    
    /* Install in TLB */
    tlb_entry_t *victim = &tlb->entries[set_idx * tlb->assoc];
    tick_t oldest_time = victim->last_access;
    
    for (int i = 1; i < tlb->assoc; i++) {
        tlb_entry_t *entry = &tlb->entries[set_idx * tlb->assoc + i];
        if (!entry->valid || entry->last_access < oldest_time) {
            victim = entry;
            oldest_time = entry->last_access;
        }
    }
    
    victim->vpn = vpn;
    victim->ppn = ppn;
    victim->valid = 1;
    victim->thread_id = thread_id;
    victim->last_access = cycles + ptw_latency;
    
    return TLB_MISS;
}

static void execute_alpha_instruction(md_inst_t inst, struct regs_t *regs, md_addr_t pc) {
    int opcode = (inst >> 26) & 0x3F;
    int ra = (inst >> 21) & 0x1F;
    int rb = (inst >> 16) & 0x1F;
    int rc = inst & 0x1F;
    
    qword_t va = (ra == 31) ? 0 : regs->regs_R[ra];
    qword_t vb = (rb == 31) ? 0 : regs->regs_R[rb];
    qword_t result = 0;
    
    /* ALU operations */
    if (opcode == 0x10) { /* INTA (integer arithmetic) */
        int func = (inst >> 5) & 0x7F;
        switch (func) {
            case 0x00: result = va + vb; break;     /* ADDL */
            case 0x02: result = va + vb; break;     /* S4ADDL */
            case 0x09: result = va - vb; break;     /* SUBL */
            case 0x0B: result = va - vb; break;     /* S4SUBL */
            case 0x0F: result = ~(va | vb); break;  /* ORNOT */
            case 0x20: result = va + vb; break;     /* ADDQ */
            case 0x29: result = va - vb; break;     /* SUBQ */
            case 0x2D: result = va * vb; break;     /* UMULH */
            default: result = va; break;
        }
    } else if (opcode == 0x11) { /* INTL (integer logical) */
        int func = (inst >> 5) & 0x7F;
        switch (func) {
            case 0x00: result = va & vb; break;     /* AND */
            case 0x08: result = ~va & vb; break;    /* BIC */
            case 0x14: result = va ^ ~vb; break;    /* ORNOT */
            case 0x20: result = va | vb; break;     /* BIS */
            case 0x24: result = va | ~vb; break;    /* ORNOT */
            case 0x40: result = va ^ vb; break;     /* XOR */
            default: result = va; break;
        }
    } else if (opcode == 0x12) { /* INTS (integer shift) */
        int func = (inst >> 5) & 0x7F;
        int shift_amount = (inst >> 10) & 0x3F;
        switch (func) {
            case 0x39: result = va << shift_amount; break;  /* SLL */
            case 0x3C: result = va >> shift_amount; break;  /* SRL */
            case 0x3A: result = (sword_t)va >> shift_amount; break; /* SRA */
            default: result = va; break;
        }
    }
    
    /* Write result back */
    if (rc != 31 && opcode >= 0x10 && opcode <= 0x13) {
        regs->regs_R[rc] = result;
    }
}
/* ===== Dynamic Resource Allocation ===== */

typedef struct resource_partition {
    int fetch_slots;     /* IFQ slots per thread */
    int rename_slots;    /* ROB slots per thread */
    int issue_slots;     /* IQ slots per thread */
    int lsq_slots;       /* LSQ slots per thread */
    int cache_ways;      /* Cache ways per thread */
} resource_partition_t;

static resource_partition_t resource_partitions[MAX_HW_THREAD];
static int dynamic_partitioning_enabled = 1;

/* Resource usage tracking */
typedef struct resource_usage {
    int ifq_usage;
    int rob_usage;
    int iq_usage;
    int lsq_usage;
    double ipc;
    double cache_miss_rate;
    int priority_score;
} resource_usage_t;

static resource_usage_t usage_stats[MAX_HW_THREAD];

/* ===== Cache Coherence ===== */

typedef enum {
    MESI_INVALID = 0,
    MESI_SHARED = 1,
    MESI_EXCLUSIVE = 2,
    MESI_MODIFIED = 3
} mesi_state_t;

typedef struct coherence_entry {
    md_addr_t addr;
    mesi_state_t state;
    int owner_thread;
    int sharers_mask;  /* Bitmask of sharing threads */
    tick_t last_access;
} coherence_entry_t;

#define COHERENCE_TABLE_SIZE 1024
static coherence_entry_t coherence_table[COHERENCE_TABLE_SIZE];

typedef enum {
    BUS_READ = 0,
    BUS_WRITE = 1,
    BUS_INVALIDATE = 2,
    BUS_FLUSH = 3
} bus_transaction_t;

static counter_t bus_transactions = 0;
static counter_t coherence_misses = 0;
static counter_t invalidations = 0;

/* ===== Advanced Prefetcher ===== */

typedef struct stride_entry {
    md_addr_t pc;
    md_addr_t last_addr;
    int stride;
    int confidence;
    int active;
} stride_entry_t;

#define STRIDE_TABLE_SIZE 64
static stride_entry_t stride_table[STRIDE_TABLE_SIZE];

typedef struct prefetch_request {
    md_addr_t addr;
    int thread_id;
    tick_t issue_time;
    int useful;  /* Was this prefetch actually used? */
    struct prefetch_request *next;
} prefetch_request_t;

#define PREFETCH_QUEUE_SIZE 16
static prefetch_request_t prefetch_queue[PREFETCH_QUEUE_SIZE];
static int prefetch_head = 0, prefetch_tail = 0;

static counter_t prefetches_issued = 0;
static counter_t prefetches_useful = 0;
static counter_t prefetches_late = 0;

/* ===== Precise Exception Handling ===== */

typedef enum {
    EXCEPTION_NONE = 0,
    EXCEPTION_PAGE_FAULT = 1,
    EXCEPTION_PROTECTION_VIOLATION = 2,
    EXCEPTION_DIVIDE_BY_ZERO = 3,
    EXCEPTION_OVERFLOW = 4,
    EXCEPTION_SYSTEM_CALL = 5
} exception_type_t;

typedef struct exception_entry {
    exception_type_t type;
    md_addr_t pc;
    md_addr_t fault_addr;  /* For memory exceptions */
    int thread_id;
    int rob_idx;
    tick_t detection_cycle;
} exception_entry_t;

#define EXCEPTION_BUFFER_SIZE 8
static exception_entry_t exception_buffer[EXCEPTION_BUFFER_SIZE];
static int exception_head = 0, exception_tail = 0;

static counter_t exceptions_detected = 0;
static counter_t exceptions_handled = 0;
static counter_t precise_exceptions = 0;

static void detect_exception(int rob_idx, exception_type_t type, 
                           md_addr_t fault_addr) {
    if (((exception_tail + 1) % EXCEPTION_BUFFER_SIZE) == exception_head) {
        return; /* Exception buffer full */
    }
    
    struct rob_entry *re = &ROB[rob_idx];
    exception_entry_t *exc = &exception_buffer[exception_tail];
    
    exc->type = type;
    exc->pc = re->PC;
    exc->fault_addr = fault_addr;
    exc->thread_id = re->tid;
    exc->rob_idx = rob_idx;
    exc->detection_cycle = cycles;
    
    exception_tail = (exception_tail + 1) % EXCEPTION_BUFFER_SIZE;
    exceptions_detected++;
}
static void handle_syscall_exit(int tid){
  tctx[tid].active = 0;
}
static void handle_precise_exceptions(void) {
    while (exception_head != exception_tail) {
        exception_entry_t *exc = &exception_buffer[exception_head];
        
        /* Check if this is the oldest instruction in the ROB */
        if (exc->rob_idx != rob_head_global) {
            break; /* Wait for precise exception point */
        }
        
        /* Handle the exception */
        switch (exc->type) {
            case EXCEPTION_PAGE_FAULT:
                printf("Page fault at PC 0x%llx, addr 0x%llx, thread %d\n",
                       exc->pc, exc->fault_addr, exc->thread_id);
                break;
                
            case EXCEPTION_SYSTEM_CALL:
                handle_syscall_exit(exc->thread_id);
                break;
                
            default:
                printf("Exception type %d at PC 0x%llx, thread %d\n",
                       exc->type, exc->pc, exc->thread_id);
                break;
        }
        
        /* Flush pipeline for this thread */
        flush_thread(exc->thread_id);
        
        exception_head = (exception_head + 1) % EXCEPTION_BUFFER_SIZE;
        exceptions_handled++;
        precise_exceptions++;
    }
}
/* =====  F O R W A R D S ============================================ */
static void fetch_stage(void);
static void rename_stage(void);
static void issue_stage(void);
static void writeback_stage(void);
static void commit_stage(void);
static void flush_thread(int tid);
/* =====  S T A T I S T I C S ======================================== */
static counter_t fetch_ifq_full  = 0;   /* if IFQ full, then fetch stall */
static counter_t rename_rob_full = 0;   /* ROB FULL -> rename stall     */
static counter_t rename_iq_full = 0;   /* IQ FULL -> rename stall        */
static counter_t rename_lsq_full = 0;   /* LSQ FULL -> rename stall        */
static counter_t issue_iq_empty = 0;
static counter_t fair_rounds = 0;
static inline int addr_overlap(md_addr_t addr1, int size1, md_addr_t addr2, int size2) {
  md_addr_t end1 = addr1 + size1 - 1;
  md_addr_t end2 = addr2 + size2 - 1;
  return !(end1 < addr2 || end2 < addr1);
}
typedef struct performance_counters {
    /* Execution stats */
    counter_t cycles_executed;
    counter_t instructions_committed;
    counter_t branches_executed;
    counter_t branches_mispredicted;
    
    /* Memory system stats */
    counter_t loads_executed;
    counter_t stores_executed;
    counter_t load_store_forwards;
    counter_t memory_violations;
    
    /* Cache stats */
    counter_t l1i_accesses;
    counter_t l1i_misses;
    counter_t l1d_accesses;
    counter_t l1d_misses;
    counter_t l2_accesses;
    counter_t l2_misses;
    
    /* TLB stats */
    counter_t itlb_accesses;
    counter_t itlb_misses;
    counter_t dtlb_accesses;
    counter_t dtlb_misses;
    
    /* Resource utilization */
    double avg_ifq_occupancy;
    double avg_rob_occupancy;
    double avg_iq_occupancy;
    double avg_lsq_occupancy;
    
    /* Thread interaction */
    counter_t resource_conflicts;
    counter_t thread_switches;
    counter_t flush_events;
} performance_counters_t;

static performance_counters_t perf_counters[MAX_HW_THREAD];

static void update_performance_counters(void) {
    for (int t = 0; t < num_hw_threads; t++) {
        if (!tctx[t].active) continue;
        
        /* Update basic stats */
        perf_counters[t].cycles_executed = cycles;
        perf_counters[t].instructions_committed = sim_num_insn_tid[t];
        
        /* Calculate resource occupancy */
        int ifq_count = 0, rob_count = 0, iq_count = 0, lsq_count = 0;
        
        for (int i = ifq_head; i != ifq_tail; i = (i + 1) % IFQ_SIZE) {
            if (IFQ[i].tid == t) ifq_count++;
        }
        
        for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE) {
            if (ROB[i].tid == t) rob_count++;
        }
        
        for (int i = 0; i < IQ_SIZE; i++) {
            if (IQ[i].ready && IQ[i].tid == t) iq_count++;
        }
        
        for (int i = lsq_head; i != lsq_tail; i = (i + 1) % LSQ_SIZE) {
            if (LSQ[i].tid == t) lsq_count++;
        }
        
        /* Update running averages */
        double alpha = 0.1; /* Exponential smoothing factor */
        perf_counters[t].avg_ifq_occupancy = 
            (1 - alpha) * perf_counters[t].avg_ifq_occupancy + alpha * ifq_count;
        perf_counters[t].avg_rob_occupancy = 
            (1 - alpha) * perf_counters[t].avg_rob_occupancy + alpha * rob_count;
        perf_counters[t].avg_iq_occupancy = 
            (1 - alpha) * perf_counters[t].avg_iq_occupancy + alpha * iq_count;
        perf_counters[t].avg_lsq_occupancy = 
            (1 - alpha) * perf_counters[t].avg_lsq_occupancy + alpha * lsq_count;
    }
}
static void init_advanced_features(void) {
    /* Initialize TLB */
    dtlb = tlb_create(TLB_SIZE, 4);
    itlb = tlb_create(TLB_SIZE, 4);
    
    /* Initialize coherence table */
    memset(coherence_table, 0, sizeof(coherence_table));
    
    /* Initialize stride prefetcher */
    memset(stride_table, 0, sizeof(stride_table));
    
    /* Initialize performance counters */
    memset(perf_counters, 0, sizeof(perf_counters));
    
    /* Initialize resource partitions */
    for (int t = 0; t < MAX_HW_THREAD; t++) {
        resource_partitions[t].fetch_slots = IFQ_SIZE / num_hw_threads;
        resource_partitions[t].rename_slots = ROB_SIZE / num_hw_threads;
        resource_partitions[t].issue_slots = IQ_SIZE / num_hw_threads;
        resource_partitions[t].lsq_slots = LSQ_SIZE / num_hw_threads;
    }
}
/* =====  C A C H E S ======================================== */
/* Cache Configuration */
typedef struct cache_config {
  int size;           /* total size in bytes */
  int line_size;      /* cache line size */
  int assoc;          /* associativity */
  int latency;        /* access latency */
  char *replacement;  /* replacement policy: LRU, FIFO, RANDOM */
} cache_config_t;

/* Cache Line Structure */
typedef struct cache_line {
  md_addr_t tag;
  int valid;
  int dirty;
  tick_t last_access;
  int thread_id;      /* for SMT coherency */
  struct cache_line *next; /* for LRU chain */
} cache_line_t;

/* Cache Set Structure */
typedef struct cache_set {
  cache_line_t *lines;
  cache_line_t *lru_head;
  cache_line_t *lru_tail;
  int occupancy;
} cache_set_t;

/* Cache Structure */
typedef struct cache {
  char *name;
  cache_config_t config;
  cache_set_t *sets;
  int num_sets;
  
  /* Statistics */
  counter_t hits;
  counter_t misses;
  counter_t writebacks;
  counter_t replacements;
  
  /* MSHR */
  struct mshr *mshr;
  int mshr_size;
  
  /* Next level cache */
  struct cache *next_level;
} cache_t;

/* MSHR Entry */
typedef struct mshr_entry {
  md_addr_t addr;
  int valid;
  int thread_id;
  tick_t issue_time;
  
  /* Multiple pending requests to same line */
  int pending_loads[MAX_HW_THREAD];
  int pending_stores[MAX_HW_THREAD];
  int num_pending;
  
  struct mshr_entry *next;
} mshr_entry_t;

/* MSHR Structure */
typedef struct mshr {
  mshr_entry_t *entries;
  int size;
  int occupancy;
  mshr_entry_t *free_list;
} mshr_t;

/* Global Cache Hierarchy */
static cache_t *il1_cache = NULL;   /* L1 Instruction Cache */
static cache_t *dl1_cache = NULL;   /* L1 Data Cache */
static cache_t *dl2_cache = NULL;   /* L2 Unified Cache */

/* Cache Statistics per Thread */
static counter_t il1_hits_tid[MAX_HW_THREAD] = {0};
static counter_t il1_misses_tid[MAX_HW_THREAD] = {0};
static counter_t dl1_hits_tid[MAX_HW_THREAD] = {0};
static counter_t dl1_misses_tid[MAX_HW_THREAD] = {0};
static counter_t dl2_hits_tid[MAX_HW_THREAD] = {0};
static counter_t dl2_misses_tid[MAX_HW_THREAD] = {0};

/* Cache Access Result */
typedef enum {
  CACHE_HIT = 0,
  CACHE_MISS = 1,
  CACHE_MSHR_HIT = 2  /* Miss but already in MSHR */
} cache_access_result_t;
/* Declare some functions */
static void complete_mshr_entry(cache_t *cache, md_addr_t addr);
static void handle_cache_writeback(cache_t *cache, cache_line_t *victim_line, int set_idx);
/* ===== Cache Functions ===== */

static cache_t* cache_create(char *name, cache_config_t config, int mshr_size) {
    cache_t *cache = (cache_t*)calloc(1, sizeof(cache_t));
  
  cache->name = strdup(name);
  cache->config = config;
  cache->num_sets = config.size / (config.line_size * config.assoc);
  
  /* Allocate sets */
  cache->sets = (cache_set_t*)calloc(cache->num_sets, sizeof(cache_set_t));
  
  for (int i = 0; i < cache->num_sets; i++) {
    cache->sets[i].lines = (cache_line_t*)calloc(config.assoc, sizeof(cache_line_t));
    /* Initialize LRU chain */
    for (int j = 0; j < config.assoc; j++) {
      cache_line_t *line = &cache->sets[i].lines[j];
      if (j == 0) {
        cache->sets[i].lru_head = line;
      } else {
        cache->sets[i].lines[j-1].next = line;
      }
      if (j == config.assoc - 1) {
        cache->sets[i].lru_tail = line;
          line->next = NULL;
      }
    }
  }
  
  /* Create MSHR */
  if (mshr_size > 0) {
    cache->mshr = (mshr_t*)calloc(1, sizeof(mshr_t));
    cache->mshr->entries = (mshr_entry_t*)calloc(mshr_size, sizeof(mshr_entry_t));
    cache->mshr->size = mshr_size;
    
    /* Initialize free list */
    for (int i = 0; i < mshr_size - 1; i++) {
      cache->mshr->entries[i].next = &cache->mshr->entries[i+1];
    }
    cache->mshr->free_list = &cache->mshr->entries[0];
  }
  
  return cache;
}
/* ===== MSHR EVENT QUEUE SYSTEM ===================================== */
typedef struct mshr_event {
  tick_t ready_cycle;
  md_addr_t addr;
  int cache_level;  /* 1=L1, 2=L2 */
  int is_writeback;
  int thread_id;
  struct mshr_event *next;
} mshr_event_t;

typedef struct event_queue {
  mshr_event_t *head;
  mshr_event_t *tail;
  int size;
} event_queue_t;

static event_queue_t *mshr_events = NULL;

/* Event Queue Operations */
static void event_queue_init() {
    mshr_events = (event_queue_t*)calloc(1, sizeof(event_queue_t));
}
static void handle_line_fill_completion(cache_t *cache, md_addr_t addr, int thread_id);
static void handle_writeback_completion(cache_t *cache, md_addr_t addr, int thread_id);
static void event_queue_insert(tick_t ready_cycle, md_addr_t addr, 
                              int cache_level, int is_writeback, int thread_id) {
  mshr_event_t *event = (mshr_event_t*)malloc(sizeof(mshr_event_t));
  event->ready_cycle = ready_cycle;
  event->addr = addr;
  event->cache_level = cache_level;
  event->is_writeback = is_writeback;
  event->thread_id = thread_id;
  event->next = NULL;
  
  /* Insert in chronological order */
  if (!mshr_events->head || ready_cycle < mshr_events->head->ready_cycle) {
    event->next = mshr_events->head;
    mshr_events->head = event;
    if (!mshr_events->tail) mshr_events->tail = event;
  } else {
    mshr_event_t *curr = mshr_events->head;
    while (curr->next && curr->next->ready_cycle <= ready_cycle) {
        curr = curr->next;
    }
    event->next = curr->next;
    curr->next = event;
    if (!event->next) mshr_events->tail = event;
  }
  mshr_events->size++;
}

static void process_mshr_events() {
  while (mshr_events->head && mshr_events->head->ready_cycle <= cycles) {
    mshr_event_t *event = mshr_events->head;
    mshr_events->head = event->next;
    if (!mshr_events->head) mshr_events->tail = NULL;
    mshr_events->size--;
    
    cache_t *cache = (event->cache_level == 1) ? dl1_cache : dl2_cache;
    
    if (event->is_writeback) {
      /* Handle writeback completion */
      handle_writeback_completion(cache, event->addr, event->thread_id);
    } else {
      /* Handle line fill completion */
      handle_line_fill_completion(cache, event->addr, event->thread_id);
    }
    
    free(event);
  }
}
static void handle_line_fill_completion(cache_t *cache, md_addr_t addr, int thread_id) {
  /* Complete MSHR entry */
  complete_mshr_entry(cache, addr);
  
  /* Update pending LSQ entries */
  md_addr_t line_addr = addr / cache->config.line_size;
  for (int i = lsq_head; i != lsq_tail; i = (i + 1) % LSQ_SIZE) {
    if (LSQ[i].tid == thread_id && 
      (LSQ[i].addr / cache->config.line_size) == line_addr &&
      LSQ[i].done > cycles) {
      LSQ[i].done = cycles; /* Ready now */
    }
  }
}

static void handle_writeback_completion(cache_t *cache, md_addr_t addr, int thread_id) {
  /* Writeback completed - victim line can be reused */
  cache->writebacks++;
  
  /* If this was blocking other misses, they can now proceed */
  md_addr_t line_addr = addr / cache->config.line_size;
  int set_idx = (line_addr % cache->num_sets);
  
  /* Check if any pending requests can now proceed */
  for (int i = 0; i < cache->mshr->size; i++) {
    mshr_entry_t *entry = &cache->mshr->entries[i];
    if (entry->valid) {
      md_addr_t entry_line = entry->addr / cache->config.line_size;
      int entry_set = entry_line % cache->num_sets;
      
      if (entry_set == set_idx && entry->issue_time < cycles) {
        /* This request can now proceed */
        tick_t completion_time = cycles + cache->config.latency;
        event_queue_insert(completion_time, entry->addr, 
                          (cache == dl1_cache) ? 1 : 2, 0, entry->thread_id);
      }
    }
  }
}
static void cache_update_lru(cache_t *cache, int set_idx, cache_line_t *accessed_line) {
  cache_set_t *set = &cache->sets[set_idx];
  
  /* If already at head, nothing to do */
  if (set->lru_head == accessed_line) return;
  
  /* Remove from current position */
  cache_line_t *prev = NULL;
  for (cache_line_t *curr = set->lru_head; curr; prev = curr, curr = curr->next) {
    if (curr == accessed_line) {
      if (prev) prev->next = curr->next;
      if (set->lru_tail == curr) set->lru_tail = prev;
      break;
    }
  }
  
  /* Insert at head */
  accessed_line->next = set->lru_head;
  set->lru_head = accessed_line;
  accessed_line->last_access = cycles;
}

static cache_line_t* cache_find_victim(cache_t *cache, int set_idx) {
  cache_set_t *set = &cache->sets[set_idx];
  
  /* LRU replacement */
  if (strcmp(cache->config.replacement, "LRU") == 0) {
    return set->lru_tail;
  }
  
  /* FIFO replacement */
  else if (strcmp(cache->config.replacement, "FIFO") == 0) {
    tick_t oldest_time = UINT64_MAX;
    cache_line_t *oldest = NULL;
    for (int i = 0; i < cache->config.assoc; i++) {
      if (set->lines[i].last_access < oldest_time) {
        oldest_time = set->lines[i].last_access;
        oldest = &set->lines[i];
      }
    }
    return oldest;
  }
  
  /* Random replacement */
  else {
    return &set->lines[rand() % cache->config.assoc];
  }
}

static cache_access_result_t cache_access(cache_t *cache, md_addr_t addr, 
                                         int is_write, int thread_id, 
                                         tick_t *ready_time) {
    
  md_addr_t tag = addr / cache->config.line_size;
  int set_idx = tag % cache->num_sets;
  tag = tag / cache->num_sets;
  
  cache_set_t *set = &cache->sets[set_idx];
  
  /* Search for hit */
  for (int i = 0; i < cache->config.assoc; i++) {
    cache_line_t *line = &set->lines[i];
    if (line->valid && line->tag == tag) {
      /* Cache hit */
      cache_update_lru(cache, set_idx, line);
      line->thread_id = thread_id;
      if (is_write) line->dirty = 1;
      
      cache->hits++;
      if (cache == il1_cache) il1_hits_tid[thread_id]++;
      else if (cache == dl1_cache) dl1_hits_tid[thread_id]++;
      else if (cache == dl2_cache) dl2_hits_tid[thread_id]++;
      
      *ready_time = cycles + cache->config.latency;
      return CACHE_HIT;
    }
  }
  
  /* Cache miss - check MSHR */
  if (cache->mshr) {
    md_addr_t line_addr = addr / cache->config.line_size;
    for (int i = 0; i < cache->mshr->size; i++) {
      mshr_entry_t *entry = &cache->mshr->entries[i];
      if (entry->valid && (entry->addr / cache->config.line_size) == (addr / cache->config.line_size)) {
        /* MSHR hit - merge request */
        if (is_write) {
          entry->pending_stores[thread_id]++;
        } else {
          entry->pending_loads[thread_id]++;
        }
        entry->num_pending++;
        
        /* Estimated completion time */
        *ready_time = entry->issue_time + cache->config.latency + 
                      (cache->next_level ? cache->next_level->config.latency * 10 : 100);
        return CACHE_MSHR_HIT;
      }
    }
    
    /* Allocate new MSHR entry if available */
    if (cache->mshr->free_list && cache->mshr->occupancy < cache->mshr->size) {
      mshr_entry_t *entry = cache->mshr->free_list;
      cache->mshr->free_list = entry->next;
      cache->mshr->occupancy++;
      
      entry->valid = 1;
      entry->addr = addr;
      entry->thread_id = thread_id;
      entry->issue_time = cycles;
      memset(entry->pending_loads, 0, sizeof(entry->pending_loads));
      memset(entry->pending_stores, 0, sizeof(entry->pending_stores));
      
      if (is_write) {
        entry->pending_stores[thread_id] = 1;
      } else {
        entry->pending_loads[thread_id] = 1;
      }
      entry->num_pending = 1;

      /* Schedule completion event */
      tick_t completion_time = cycles + cache->config.latency;
      if (cache->next_level) {
        completion_time += cache->next_level->config.latency * 5;
      } else {
        completion_time += 100; /* Memory latency */
      }
      event_queue_insert(completion_time, addr, (cache == dl1_cache) ? 1 : 2, 0, thread_id);
    }
  }
  
  /* Miss handling */
  cache->misses++;
  if (cache == il1_cache) il1_misses_tid[thread_id]++;
  else if (cache == dl1_cache) dl1_misses_tid[thread_id]++;
  else if (cache == dl2_cache) dl2_misses_tid[thread_id]++;
  
  /* Check MSHR */
  if (cache->mshr) {
    md_addr_t line_addr = addr / cache->config.line_size;
    for (int i = 0; i < cache->mshr->size; i++) {
      mshr_entry_t *entry = &cache->mshr->entries[i];
      if (entry->valid && (entry->addr / cache->config.line_size) == (addr / cache->config.line_size)) {
        /* MSHR hit - merge request */
        if (is_write) {
          entry->pending_stores[thread_id]++;
        } else {
          entry->pending_loads[thread_id]++;
        }
        entry->num_pending++;
        
        /* Estimated completion time */
        *ready_time = entry->issue_time + cache->config.latency + 
                      (cache->next_level ? cache->next_level->config.latency * 10 : 100);
        return CACHE_MSHR_HIT;
      }
    }
    
    /* Allocate new MSHR entry if available */
    if (cache->mshr->free_list && cache->mshr->occupancy < cache->mshr->size) {
      mshr_entry_t *entry = cache->mshr->free_list;
      cache->mshr->free_list = entry->next;
      cache->mshr->occupancy++;
      
      entry->valid = 1;
      entry->addr = addr;
      entry->thread_id = thread_id;
      entry->issue_time = cycles;
      memset(entry->pending_loads, 0, sizeof(entry->pending_loads));
      memset(entry->pending_stores, 0, sizeof(entry->pending_stores));
      
      if (is_write) {
        entry->pending_stores[thread_id] = 1;
      } else {
        entry->pending_loads[thread_id] = 1;
      }
      entry->num_pending = 1;

      /* Schedule completion event */
      tick_t completion_time = cycles + cache->config.latency;
      if (cache->next_level) {
        completion_time += cache->next_level->config.latency * 5;
      } else {
        completion_time += 100; /* Memory latency */
      }
      event_queue_insert(completion_time, addr, (cache == dl1_cache) ? 1 : 2, 0, thread_id);
    }
  }

  /* Handle replacement and potential writeback */
  cache_line_t *victim = cache_find_victim(cache, set_idx);
    if (victim->valid && victim->dirty) {
      /* Schedule writeback event */
      md_addr_t wb_addr = (victim->tag * cache->num_sets + set_idx) * cache->config.line_size;
      tick_t wb_completion = cycles + cache->config.latency;
      if (cache->next_level) wb_completion += cache->next_level->config.latency;        
      event_queue_insert(wb_completion, wb_addr, 
                          (cache == dl1_cache) ? 1 : 2, 1, victim->thread_id);
  }

  /* Handle replacement and potential writeback */
 
  if (victim->valid && victim->dirty) {
    /* Schedule writeback event */
    md_addr_t wb_addr = (victim->tag * cache->num_sets + set_idx) * cache->config.line_size;
    tick_t wb_completion = cycles + cache->config.latency;
    if (cache->next_level) wb_completion += cache->next_level->config.latency;
    
    event_queue_insert(wb_completion, wb_addr, 
                      (cache == dl1_cache) ? 1 : 2, 1, victim->thread_id);
  }
    
  /* Install new line */
  victim->tag = tag;
  victim->valid = 1;
  victim->dirty = is_write;
  victim->thread_id = thread_id;
  victim->last_access = cycles;
  cache_update_lru(cache, set_idx, victim);

  cache_update_lru(cache, set_idx, victim);

  /* Calculate miss penalty */
  int miss_latency = cache->config.latency;
  if (cache->next_level) {
    tick_t next_ready;
    cache_access_result_t next_result = cache_access(cache->next_level, addr, is_write, thread_id, &next_ready);
    miss_latency += (next_ready - cycles);
  } else {
    miss_latency += 100; /* Main memory latency */
  }
  
  *ready_time = cycles + miss_latency;

  /* MSHR 완료 처리 (시뮬레이션을 위해 즉시 처리) */
  complete_mshr_entry(cache, addr);

  return CACHE_MISS;
}

/* ===== Cache Initialization ===== */
static void init_cache_hierarchy() {
  /* L1 Instruction Cache: 32KB, 32B line, 4-way, 1 cycle */
  cache_config_t il1_config = {32*1024, 32, 4, 1, "LRU"};
  il1_cache = cache_create("IL1", il1_config, 4);
  
  /* L1 Data Cache: 32KB, 32B line, 4-way, 1 cycle */
  cache_config_t dl1_config = {32*1024, 32, 4, 1, "LRU"};
  dl1_cache = cache_create("DL1", dl1_config, 8);
  
  /* L2 Unified Cache: 256KB, 64B line, 8-way, 10 cycles */
  cache_config_t dl2_config = {256*1024, 64, 8, 10, "LRU"};
  dl2_cache = cache_create("DL2", dl2_config, 16);
  
  /* Set up hierarchy */
  il1_cache->next_level = dl2_cache;
  dl1_cache->next_level = dl2_cache;
  memset(il1_hits_tid, 0, sizeof(il1_hits_tid));
  memset(il1_misses_tid, 0, sizeof(il1_misses_tid));
  memset(dl1_hits_tid, 0, sizeof(dl1_hits_tid));
  memset(dl1_misses_tid, 0, sizeof(dl1_misses_tid));
  memset(dl2_hits_tid, 0, sizeof(dl2_hits_tid));
  memset(dl2_misses_tid, 0, sizeof(dl2_misses_tid));
}
/* Function prototype to avoid implicit declaration */
static void handle_coherence_transaction(md_addr_t addr, bus_transaction_t trans, 
                                       int requesting_thread) {
    unsigned idx = (addr >> 6) % COHERENCE_TABLE_SIZE; /* Cache line granularity */
    coherence_entry_t *entry = &coherence_table[idx];
    
    bus_transactions++;
    
    if (entry->addr != addr || entry->state == MESI_INVALID) {
        /* New cache line */
        entry->addr = addr;
        entry->owner_thread = requesting_thread;
        entry->sharers_mask = (1 << requesting_thread);
        entry->state = (trans == BUS_WRITE) ? MESI_MODIFIED : MESI_EXCLUSIVE;
        entry->last_access = cycles;
        return;
    }
    
    switch (trans) {
        case BUS_READ:
            if (entry->state == MESI_MODIFIED) {
                /* Modified -> Shared, writeback required */
                entry->state = MESI_SHARED;
                coherence_misses++;
            } else if (entry->state == MESI_EXCLUSIVE) {
                entry->state = MESI_SHARED;
            }
            entry->sharers_mask |= (1 << requesting_thread);
            break;
            
        case BUS_WRITE:
            /* Invalidate all other copies */
            for (int t = 0; t < num_hw_threads; t++) {
                if (t != requesting_thread && (entry->sharers_mask & (1 << t))) {
                    invalidations++;
                    /* Simulate invalidation latency */
                    if (dl1_cache) {
                        // Mark cache line as invalid in thread t's view
                    }
                }
            }
            entry->state = MESI_MODIFIED;
            entry->owner_thread = requesting_thread;
            entry->sharers_mask = (1 << requesting_thread);
            break;
            
        case BUS_INVALIDATE:
            entry->state = MESI_INVALID;
            entry->sharers_mask = 0;
            invalidations++;
            break;
    }
    
    entry->last_access = cycles;
}
static void stride_prefetcher_access(md_addr_t pc, md_addr_t addr, int thread_id) {
  if (!enable_stride_prefetcher) return;  
  
  unsigned idx = (pc >> 2) % STRIDE_TABLE_SIZE;
  stride_entry_t *entry = &stride_table[idx];
  
  if (entry->pc != pc || !entry->active) {
    /* New PC */
    entry->pc = pc;
    entry->last_addr = addr;
    entry->stride = 0;
    entry->confidence = 0;
    entry->active = 1;
    return;
  }
  
  /* Calculate stride */
  int new_stride = (int)(addr - entry->last_addr);
  
  if (new_stride == entry->stride) {
    /* Stride confirmed */
    entry->confidence = MIN(entry->confidence + 1, 7);
    
    /* Issue prefetch if confident */
    if (entry->confidence >= 3) {
      md_addr_t prefetch_addr = addr + entry->stride;
      
      /* Add to prefetch queue */
      if (((prefetch_tail + 1) % PREFETCH_QUEUE_SIZE) != prefetch_head) {
        prefetch_request_t *req = &prefetch_queue[prefetch_tail];
        req->addr = prefetch_addr;
        req->thread_id = thread_id;
        req->issue_time = cycles;
        req->useful = 0;
        
        prefetch_tail = (prefetch_tail + 1) % PREFETCH_QUEUE_SIZE;
        prefetches_issued++;
        
        /* Trigger cache access */
        tick_t ready_time;
        cache_access(dl1_cache, prefetch_addr, 0, thread_id, &ready_time);
      }
    }
  } else {
    /* Stride changed */
    entry->stride = new_stride;
    entry->confidence = MAX(entry->confidence - 1, 0);
  }
  
  entry->last_addr = addr;
}
/* ===== Integration with LSQ ===== */
static void enhanced_lsq_access(struct lsq_entry *lsq, int lsq_idx) {
  if (!lsq->addr_ready) return;
  
  /* TLB access */
  md_addr_t physical_addr;
  tlb_access_result_t tlb_result = tlb_access(dtlb, lsq->vaddr, lsq->tid, &physical_addr);
  
  if (tlb_result == TLB_MISS) {
      lsq->done = cycles + 25; /* TLB miss penalty */
      return;
  }
  
  lsq->addr = physical_addr; /* Use physical address */

  tick_t cache_ready_time;
  cache_access_result_t result;

  /* TLB access for data */
  if (dtlb) {
    tlb_access_result_t tlb_result = tlb_access(dtlb, lsq->vaddr, lsq->tid, &physical_addr);
    if (tlb_result == TLB_MISS) {
      lsq->done = cycles + 25; /* Page table walk penalty */
      return;
    }
    lsq->addr = physical_addr; /* Use physical address for cache */
  }

  if (lsq->is_load) {
    /* Trigger stride prefetcher */
    struct rob_entry *re = &ROB[lsq->rob_idx];
    stride_prefetcher_access(re->PC, lsq->addr, lsq->tid);
    
    /* Check store forwarding first */
    forward_result_t forward_result = check_store_forwarding(lsq_idx);
    
    if (forward_result == FORWARD_FULL) {
      return; /* Forwarding handled completion */
    }
    
    if (forward_result == FORWARD_CONFLICT) {
      lsq->done = cycles + 30; /* Conflict resolution delay */
      return;
    }
    
    /* Handle coherence */
    handle_coherence_transaction(lsq->addr, BUS_READ, lsq->tid);
    
    /* Access cache hierarchy */
    result = cache_access(dl1_cache, lsq->addr, 0, lsq->tid, &cache_ready_time);
    lsq->done = cache_ready_time;
    
  } else { /* store */
    /* Handle coherence */
    handle_coherence_transaction(lsq->addr, BUS_WRITE, lsq->tid);
    result = cache_access(dl1_cache, physical_addr, 1, lsq->tid, &cache_ready_time);
    lsq->done = cache_ready_time;
    lsq->data_ready = 1;
  }
}
/* ===== MSHR managing functions ===== */
static void complete_mshr_entry(cache_t *cache, md_addr_t addr) {
  if (!cache->mshr) return;
  
  md_addr_t line_addr = addr / cache->config.line_size;
  
  for (int i = 0; i < cache->mshr->size; i++) {
    mshr_entry_t *entry = &cache->mshr->entries[i];
    if (entry->valid && (entry->addr / cache->config.line_size) == line_addr) {
      
      /* MSHR 엔트리 해제 */
      entry->valid = 0;
      entry->next = cache->mshr->free_list;
      cache->mshr->free_list = entry;
      cache->mshr->occupancy--;
      
      /* 대기 중인 요청들 완료 처리 */
      for (int t = 0; t < MAX_HW_THREAD; t++) {
        if (entry->pending_loads[t] > 0 || entry->pending_stores[t] > 0) {
          /* LSQ에서 해당하는 엔트리들의 완료 시간 업데이트 */
          for (int j = lsq_head; j != lsq_tail; j = (j + 1) % LSQ_SIZE) {
            if (LSQ[j].tid == t && 
                (LSQ[j].addr / cache->config.line_size) == line_addr &&
                LSQ[j].done > cycles) {
              LSQ[j].done = cycles + cache->config.latency;
            }
          }
        }
      }
      break;
    }
  }
}
static void handle_cache_writeback(cache_t *cache, cache_line_t *victim_line, int set_idx) {
  if (victim_line->valid && victim_line->dirty) {
    cache->writebacks++;
      
    /* 다음 레벨 캐시나 메모리로 writeback */
    if (cache->next_level) {
      tick_t wb_ready_time;
      md_addr_t wb_addr = (victim_line->tag * cache->num_sets + set_idx) * cache->config.line_size;
      cache_access(cache->next_level, wb_addr, 1, victim_line->thread_id, &wb_ready_time);
    }
  }
}
/* Integration Functions */
static void init_enhanced_simulator() {
  /* Initialize enhanced thread contexts */
  for (int t = 0; t < MAX_HW_THREAD; t++) {
    // tctx[t] = (struct thread_ctx){0};
    tctx[t].speculation_depth = 0;
    tctx[t].last_flush_cycle = 0;
  }
  
  /* Initialize memory dependence table */
  memset(mem_dep_table, 0, sizeof(mem_dep_table));
  
  /* Initialize existing components */
  event_queue_init();
  init_branch_predictor();
}
static void update_resource_partitions(void) {
    if (!dynamic_partitioning_enabled) return;
    
    /* Calculate current usage and performance */
    for (int t = 0; t < num_hw_threads; t++) {
        if (!tctx[t].active) continue;
        
        /* Calculate IPC */
        usage_stats[t].ipc = (cycles > 0) ? 
            (double)sim_num_insn_tid[t] / cycles : 0.0;
        
        /* Calculate cache miss rate */
        counter_t total_accesses = dl1_hits_tid[t] + dl1_misses_tid[t];
        usage_stats[t].cache_miss_rate = (total_accesses > 0) ?
            (double)dl1_misses_tid[t] / total_accesses : 0.0;
        
        /* Calculate priority score */
        usage_stats[t].priority_score = (int)(usage_stats[t].ipc * 100) - 
                                       (int)(usage_stats[t].cache_miss_rate * 50);
    }
    
    /* Redistribute resources based on performance */
    int total_ifq_slots = IFQ_SIZE;
    int total_rob_slots = ROB_SIZE;
    int total_iq_slots = IQ_SIZE;
    int total_lsq_slots = LSQ_SIZE;
    
    /* Simple proportional allocation based on priority */
    int total_priority = 0;
    for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) total_priority += MAX(usage_stats[t].priority_score, 10);
    }
    
    for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
            int thread_priority = MAX(usage_stats[t].priority_score, 10);
            resource_partitions[t].fetch_slots = 
                (total_ifq_slots * thread_priority) / total_priority;
            resource_partitions[t].rename_slots = 
                (total_rob_slots * thread_priority) / total_priority;
            resource_partitions[t].issue_slots = 
                (total_iq_slots * thread_priority) / total_priority;
            resource_partitions[t].lsq_slots = 
                (total_lsq_slots * thread_priority) / total_priority;
        }
    }
}
/* Runahead Execution */

typedef struct {
    int runahead_mode;
    md_addr_t runahead_pc;
    md_addr_t normal_pc;
    struct regs_t checkpoint_regs;
    tick_t runahead_start_cycle;
    int prefetches_generated;
} runahead_state_t;

static runahead_state_t runahead_state[MAX_HW_THREAD];

static void enter_runahead_mode(int tid, md_addr_t stall_pc) {
    if (!enable_runahead_execution) return;
    
    runahead_state_t *ra = &runahead_state[tid];
    if (ra->runahead_mode) return; /* Already in runahead */
    
    /* Save checkpoint */
    ra->checkpoint_regs = tctx[tid].regs;
    ra->normal_pc = stall_pc;
    ra->runahead_pc = stall_pc + 4;
    ra->runahead_mode = 1;
    ra->runahead_start_cycle = cycles;
    ra->prefetches_generated = 0;
    
    printf("Thread %d entering runahead mode at PC 0x%llx, cycle %lld\n", 
           tid, stall_pc, cycles);
}
static void exit_runahead_mode(int tid);
static void execute_runahead_instruction(int tid) {
    runahead_state_t *ra = &runahead_state[tid];
    if (!ra->runahead_mode) return;
    
    md_inst_t inst;
    mem_access(mem, Read, ra->runahead_pc, &inst, sizeof(md_inst_t));
    
    enum md_opcode op;
    MD_SET_OPCODE(op, inst);
    
    /* Execute instruction in runahead mode */
    if (is_load(op)) {
        /* Generate prefetch for loads */
        int base_reg = (inst >> 16) & 0x1F;
        short displacement = (short)(inst & 0xFFFF);
        
        md_addr_t base_value = (base_reg == 31) ? 0 : tctx[tid].regs.regs_R[base_reg];
        md_addr_t load_addr = base_value + displacement;
        
        /* Issue prefetch */
        tick_t ready_time;
        if (dl1_cache) {
            cache_access(dl1_cache, load_addr, 0, tid, &ready_time);
            ra->prefetches_generated++;
        }
    } else if (!(MD_OP_FLAGS(op) & F_CTRL)) {
        /* Execute non-control instructions */
        execute_alpha_instruction(inst, &tctx[tid].regs, ra->runahead_pc);
    }
    
    /* Update runahead PC */
    if (MD_OP_FLAGS(op) & F_CTRL) {
        md_addr_t target;
        int taken = resolve_branch(inst, ra->runahead_pc, &tctx[tid].regs, &target);
        ra->runahead_pc = taken ? target : (ra->runahead_pc + 4);
    } else {
        ra->runahead_pc += 4;
    }
    
    /* Exit conditions */
    if (cycles - ra->runahead_start_cycle > 100 || ra->prefetches_generated > 16) {
        exit_runahead_mode(tid);
    }
}

static void exit_runahead_mode(int tid) {
    runahead_state_t *ra = &runahead_state[tid];
    if (!ra->runahead_mode) return;
    
    /* Restore checkpoint */
    tctx[tid].regs = ra->checkpoint_regs;
    tctx[tid].pc = ra->normal_pc;
    
    printf("Thread %d exiting runahead mode, generated %d prefetches\n", 
           tid, ra->prefetches_generated);
    
    ra->runahead_mode = 0;
}
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
  opt_reg_flag(odb, "-smt:dynamic_partition", 
                 "enable dynamic resource partitioning",
                 &enable_dynamic_partitioning, /*default*/0, NULL, NULL);
  opt_reg_flag(odb, "-smt:stride_prefetch", 
                 "enable stride prefetcher",
                 &enable_stride_prefetcher, /*default*/0, NULL, NULL);
  opt_reg_flag(odb, "-smt:runahead", 
                 "enable runahead execution",
                 &enable_runahead_execution, /*default*/0, NULL, NULL);
  opt_reg_flag(odb, "-smt:mem_dep_pred", 
                 "enable memory dependency prediction",
                 &memory_dependency_prediction, /*default*/0, NULL, NULL);
}
void sim_check_options(struct opt_odb_t *odb, int argc, char **argv) {
  if (num_hw_threads < 1 || num_hw_threads > MAX_HW_THREAD)
    fatal("threads must be 1-%d", MAX_HW_THREAD);
}

void sim_reg_stats(struct stat_sdb_t *sdb) {
  stat_reg_counter(sdb, "sim_cycles", "total cycles", &cycles, 0, NULL);
  stat_reg_counter(sdb, "sim_num_insn", "instructions committed", &sim_num_insn, 0, NULL);
  
  /* Branch Predictor Stats */
  stat_reg_counter(sdb, "bp_lookups", "branch predictor lookups", &bp_lookups, 0, NULL);
  stat_reg_counter(sdb, "bp_correct", "correct branch predictions", &bp_correct, 0, NULL);
  stat_reg_counter(sdb, "bp_mispred", "branch mispredictions", &bp_mispred, 0, NULL);
  stat_reg_counter(sdb, "btb_hits", "BTB hits", &btb_hits, 0, NULL);
  stat_reg_counter(sdb, "btb_misses", "BTB misses", &btb_misses, 0, NULL);
  
  stat_reg_formula(sdb, "bp_accuracy", "branch prediction accuracy", 
                   "bp_correct / (bp_correct + bp_mispred)", NULL);
  stat_reg_formula(sdb, "btb_hit_rate", "BTB hit rate", 
                   "btb_hits / (btb_hits + btb_misses)", NULL);
  stat_reg_formula(sdb, "overall_ipc", "overall IPC", 
                     "sim_num_insn / (sim_cycles + 0.000001)", NULL);
  /* TLB Statistics */
  if (dtlb) {
    stat_reg_counter(sdb, "dtlb_hits", "DTLB hits", &(dtlb->hits), 0, NULL);
    stat_reg_counter(sdb, "dtlb_misses", "DTLB misses", &(dtlb->misses), 0, NULL);
    stat_reg_formula(sdb, "dtlb_hit_rate", "DTLB hit rate",
                     "dtlb_hits / (dtlb_hits + dtlb_misses + 0.000001)", NULL);
  }
  
  if (itlb) {
    stat_reg_counter(sdb, "itlb_hits", "ITLB hits", &(itlb->hits), 0, NULL);
    stat_reg_counter(sdb, "itlb_misses", "ITLB misses", &(itlb->misses), 0, NULL);
    stat_reg_formula(sdb, "itlb_hit_rate", "ITLB hit rate",
                     "itlb_hits / (itlb_hits + itlb_misses + 0.000001)", NULL);
  }
  /* Cache stats */
  if (il1_cache) {
    stat_reg_counter(sdb, "il1_hits", "IL1 cache hits", &(il1_cache->hits), 0, NULL);
    stat_reg_counter(sdb, "il1_misses", "IL1 cache misses", &(il1_cache->misses), 0, NULL);
    stat_reg_counter(sdb, "il1_writebacks", "IL1 writebacks", &(il1_cache->writebacks), 0, NULL);
    stat_reg_formula(sdb, "il1_hit_rate", "IL1 hit rate",
                      "il1_hits / (il1_hits + il1_misses + 0.000001)", NULL);
  }
  
  if (dl1_cache) {
    stat_reg_counter(sdb, "dl1_hits", "DL1 cache hits", &(dl1_cache->hits), 0, NULL);
    stat_reg_counter(sdb, "dl1_misses", "DL1 cache misses", &(dl1_cache->misses), 0, NULL);
    stat_reg_counter(sdb, "dl1_writebacks", "DL1 writebacks", &(dl1_cache->writebacks), 0, NULL);
    stat_reg_formula(sdb, "dl1_hit_rate", "DL1 hit rate",
                      "dl1_hits / (dl1_hits + dl1_misses + 0.000001)", NULL);
  }
  if (dl2_cache) {
    stat_reg_counter(sdb, "dl2_hits", "DL2 cache hits", &(dl2_cache->hits), 0, NULL);
    stat_reg_counter(sdb, "dl2_misses", "DL2 cache misses", &(dl2_cache->misses), 0, NULL);
    stat_reg_counter(sdb, "dl2_writebacks", "DL2 writebacks", &(dl2_cache->writebacks), 0, NULL);
    stat_reg_formula(sdb, "dl2_hit_rate", "DL2 hit rate",
                      "dl2_hits / (dl2_hits + dl2_misses + 0.000001)", NULL);
  }
  /* Cache Coherence Stats */
  stat_reg_counter(sdb, "bus_transactions", "bus transactions", &bus_transactions, 0, NULL);
  stat_reg_counter(sdb, "coherence_misses", "coherence misses", &coherence_misses, 0, NULL);
  stat_reg_counter(sdb, "invalidations", "cache invalidations", &invalidations, 0, NULL);
  
  /* Prefetching Stats */
  stat_reg_counter(sdb, "prefetches_issued", "prefetches issued", &prefetches_issued, 0, NULL);
  stat_reg_counter(sdb, "prefetches_useful", "useful prefetches", &prefetches_useful, 0, NULL);
  stat_reg_formula(sdb, "prefetch_accuracy", "prefetch accuracy",
                   "prefetches_useful / prefetches_issued", NULL);
  
  /* Exception Stats */
  stat_reg_counter(sdb, "exceptions_detected", "exceptions detected", &exceptions_detected, 0, NULL);
  stat_reg_counter(sdb, "precise_exceptions", "precise exceptions", &precise_exceptions, 0, NULL);
  
  /* LSQ stats */
  stat_reg_counter(sdb, "lsq_store_forwards", "store-to-load forwards", 
                    &lsq_store_forwards, 0, NULL);
  stat_reg_counter(sdb, "lsq_load_violations", "load-store violations", 
                    &lsq_load_violations, 0, NULL);
  stat_reg_counter(sdb, "lsq_addr_conflicts", "address conflicts", 
                    &lsq_addr_conflicts, 0, NULL);
  stat_reg_counter(sdb, "lsq_partial_forwards", "partial forwards", 
                    &lsq_partial_forwards, 0, NULL);
  /* Pipeline stall stats */
  stat_reg_counter(sdb, "fetch_ifq_full", "fetch stalls (IFQ full)", 
                    &fetch_ifq_full, 0, NULL);
  stat_reg_counter(sdb, "rename_rob_full", "rename stalls (ROB full)", 
                    &rename_rob_full, 0, NULL);
  stat_reg_counter(sdb, "rename_iq_full", "rename stalls (IQ full)", 
                    &rename_iq_full, 0, NULL);
  stat_reg_counter(sdb, "rename_lsq_full", "rename stalls (LSQ full)", 
                    &rename_lsq_full, 0, NULL);
  stat_reg_counter(sdb, "issue_iq_empty", "issue stalls (IQ empty)", 
                    &issue_iq_empty, 0, NULL);
  /* Per-thread statistics */
  for (int t = 0; t < num_hw_threads; ++t) {
    char name[32], desc[64], expr[128];
    
    sprintf(name, "sim_num_insn_t%d", t);
    sprintf(desc, "commits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &sim_num_insn_tid[t], 0, NULL);

    sprintf(name, "IPC_t%d", t);
    sprintf(desc, "IPC, thread %d", t);
    sprintf(expr, "sim_num_insn_t%d / sim_cycles", t);
    stat_reg_formula(sdb, name, desc, expr, NULL);
    
    sprintf(name, "branch_accuracy_t%d", t);
    sprintf(desc, "Branch prediction accuracy, thread %d", t);
    if (tctx[t].branches_executed > 0) {
      double accuracy = (double)(tctx[t].branches_executed - tctx[t].branches_mispredicted) 
                       / tctx[t].branches_executed * 100.0;
      stat_reg_counter(sdb, name, desc, (counter_t*)&accuracy, 0, NULL);
    }
    
    sprintf(name, "flush_count_t%d", t);
    sprintf(desc, "Pipeline flushes, thread %d", t);
    stat_reg_counter(sdb, name, desc, &tctx[t].flush_count, 0, NULL);

    /* Per-thread cache statistics */
    sprintf(name, "il1_hits_t%d", t);
    sprintf(desc, "IL1 hits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &il1_hits_tid[t], 0, NULL);
    
    sprintf(name, "il1_misses_t%d", t);
    sprintf(desc, "IL1 misses, thread %d", t);
    stat_reg_counter(sdb, name, desc, &il1_misses_tid[t], 0, NULL);
    
    sprintf(name, "dl1_hits_t%d", t);
    sprintf(desc, "DL1 hits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &dl1_hits_tid[t], 0, NULL);
    
    sprintf(name, "dl1_misses_t%d", t);
    sprintf(desc, "DL1 misses, thread %d", t);
    stat_reg_counter(sdb, name, desc, &dl1_misses_tid[t], 0, NULL);

    sprintf(name, "dl2_hits_t%d", t);
    sprintf(desc, "DL2 hits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &dl2_hits_tid[t], 0, NULL);
    
    sprintf(name, "dl2_misses_t%d", t);
    sprintf(desc, "DL2 misses, thread %d", t);
    stat_reg_counter(sdb, name, desc, &dl2_misses_tid[t], 0, NULL);

    /* Per-thread TLB statistics */
    sprintf(name, "itlb_hits_t%d", t);
    sprintf(desc, "ITLB hits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &itlb_hits_tid[t], 0, NULL);
    
    sprintf(name, "itlb_misses_t%d", t);
    sprintf(desc, "ITLB misses, thread %d", t);
    stat_reg_counter(sdb, name, desc, &itlb_misses_tid[t], 0, NULL);
    
    sprintf(name, "dtlb_hits_t%d", t);
    sprintf(desc, "DTLB hits, thread %d", t);
    stat_reg_counter(sdb, name, desc, &dtlb_hits_tid[t], 0, NULL);
    
    sprintf(name, "dtlb_misses_t%d", t);
    sprintf(desc, "DTLB misses, thread %d", t);
    stat_reg_counter(sdb, name, desc, &dtlb_misses_tid[t], 0, NULL);

    /* Per-thread hit rate formulas */
    sprintf(name, "il1_hit_rate_t%d", t);
    sprintf(desc, "IL1 hit rate, thread %d", t);
    sprintf(expr, "il1_hits_t%d / (il1_hits_t%d + il1_misses_t%d + 0.000001)", t, t, t);
    stat_reg_formula(sdb, name, desc, expr, NULL);
    
    sprintf(name, "dl1_hit_rate_t%d", t);
    sprintf(desc, "DL1 hit rate, thread %d", t);
    sprintf(expr, "dl1_hits_t%d / (dl1_hits_t%d + dl1_misses_t%d + 0.000001)", t, t, t);
    stat_reg_formula(sdb, name, desc, expr, NULL);
    
    sprintf(name, "itlb_hit_rate_t%d", t);
    sprintf(desc, "ITLB hit rate, thread %d", t);
    sprintf(expr, "itlb_hits_t%d / (itlb_hits_t%d + itlb_misses_t%d + 0.000001)", t, t, t);
    stat_reg_formula(sdb, name, desc, expr, NULL);
    
    sprintf(name, "dtlb_hit_rate_t%d", t);
    sprintf(desc, "DTLB hit rate, thread %d", t);
    sprintf(expr, "dtlb_hits_t%d / (dtlb_hits_t%d + dtlb_misses_t%d + 0.000001)", t, t, t);
    stat_reg_formula(sdb, name, desc, expr, NULL);
  }
  
  mem_reg_stats(mem, sdb);
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
  if (p >= 0 && p < PRF_NUM) {
    free_list[free_tail] = p;
    free_tail = (free_tail + 1) % PRF_NUM;
  }
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

  /* Initialize structures */
  memset(IFQ, 0, sizeof(IFQ));
  memset(IQ, 0, sizeof(IQ));
  memset(LSQ, 0, sizeof(LSQ));
  memset(ROB, 0, sizeof(ROB));
  init_cache_hierarchy();

  for (int i = 0; i < MD_TOTAL_REGS; ++i) {
    prf_ready[i] = 1;
  }
  init_branch_predictor();
  event_queue_init();
}
void sim_load_prog(char *fname, int argc, char **argv, char **envp) {
  /* load same binary into *each* thread context */
  for (int tid=0; tid<num_hw_threads; ++tid) {
    ld_load_prog(fname, argc, argv, envp, &tctx[tid].regs, mem, 0);
    tctx[tid].pc = tctx[tid].regs.regs_PC;
    tctx[tid].active = 1;
    for (int i = 0; i < MD_TOTAL_REGS; i++) {
      tctx[tid].rename_map[i] = i;
    }
    tctx[tid].rob_head = tctx[tid].rob_tail = 0;
    tctx[tid].seq = tctx[tid].icount = 0;
  }
}
void sim_uninit(void) {}
void sim_aux_stats(FILE *stream) {}
void sim_aux_config(FILE *stream) {
  fprintf(stream, "threads %d\n", num_hw_threads);
}
static int thread_empty(int tid)
{
    for (int i = ifq_head; i != ifq_tail; i = (i + 1) % IFQ_SIZE) {
      if (IFQ[i].tid == tid) return 0;
    }
    for (int i = 0; i < IQ_SIZE; ++i) {
      if (IQ[i].ready && IQ[i].tid == tid) return 0;
    }
    for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE){
      if (ROB[i].tid == tid) return 0;
    }
    for (int i = lsq_head; i != lsq_tail; i = (i + 1) % LSQ_SIZE){
      if (LSQ[i].tid == tid) return 0;
    }
    return 1;
}
static double safe_ratio(counter_t numerator, counter_t denominator) {
    return (denominator > 0) ? (double)numerator / denominator : 0.0;
}
/* Performance Analysis Functions */
static void print_performance_analysis() {
  printf("\n=== SMT Performance Analysis ===\n");
  printf("Total Cycles: %lld\n", cycles);
  printf("Total Instructions: %lld\n", sim_num_insn);
  printf("Overall IPC: %.3f\n", safe_ratio(sim_num_insn, cycles));
  
  printf("\n--- Per-Thread Statistics ---\n");
  for (int t = 0; t < num_hw_threads; t++) {
    if (sim_num_insn_tid[t] > 0) {
      printf("Thread %d:\n", t);
      printf("  Instructions: %lld\n", sim_num_insn_tid[t]);
      printf("  IPC: %.3f\n", safe_ratio(sim_num_insn_tid[t], cycles));
      
      if (tctx[t].branches_executed > 0) {
        double bp_accuracy = safe_ratio(tctx[t].branches_executed - 
                                        tctx[t].branches_mispredicted,
                                        tctx[t].branches_executed) * 100.0;
        printf("  Branch Accuracy: %.1f%%\n", bp_accuracy);
      } else {
        printf("  Branch Accuracy: N/A\n");
      }
      
      printf("  Flush Count: %lld\n", tctx[t].flush_count);
      
      /* 캐시 통계 */
      double il1_hit_rate = safe_ratio(il1_hits_tid[t], 
                                      il1_hits_tid[t] + il1_misses_tid[t]) * 100.0;
      double dl1_hit_rate = safe_ratio(dl1_hits_tid[t], 
                                      dl1_hits_tid[t] + dl1_misses_tid[t]) * 100.0;
      
      printf("  I-Cache Hit Rate: %.1f%%\n", il1_hit_rate);
      printf("  D-Cache Hit Rate: %.1f%%\n", dl1_hit_rate);
      
      /* TLB 통계 */
      double itlb_hit_rate = safe_ratio(itlb_hits_tid[t], 
                                      itlb_hits_tid[t] + itlb_misses_tid[t]) * 100.0;
      double dtlb_hit_rate = safe_ratio(dtlb_hits_tid[t], 
                                      dtlb_hits_tid[t] + dtlb_misses_tid[t]) * 100.0;
      
      printf("  I-TLB Hit Rate: %.1f%%\n", itlb_hit_rate);
      printf("  D-TLB Hit Rate: %.1f%%\n", dtlb_hit_rate);
    }
  }
  
  printf("\n--- Memory System ---\n");
  printf("Store-to-Load Forwards: %lld\n", lsq_store_forwards);
  printf("Load-Store Violations: %lld\n", lsq_load_violations);
  printf("Address Conflicts: %lld\n", lsq_addr_conflicts);
  printf("Partial Forwards: %lld\n", lsq_partial_forwards);
  
  double forwarding_rate = safe_ratio(lsq_store_forwards, 
                                      lsq_store_forwards + lsq_addr_conflicts) * 100.0;
  printf("Forwarding Rate: %.1f%%\n", forwarding_rate);
  
  printf("\n--- Branch Prediction ---\n");
  printf("Prediction Accuracy: %.1f%%\n", safe_ratio(bp_correct, bp_lookups) * 100.0);
  printf("BTB Hit Rate: %.1f%%\n", safe_ratio(btb_hits, btb_hits + btb_misses) * 100.0);
  
  printf("\n--- Prefetching ---\n");
  printf("Prefetches Issued: %lld\n", prefetches_issued);
  printf("Useful Prefetches: %lld\n", prefetches_useful);
  printf("Prefetch Accuracy: %.1f%%\n", safe_ratio(prefetches_useful, prefetches_issued) * 100.0);
  
  printf("\n--- Pipeline Stalls ---\n");
  printf("Fetch IFQ Full: %lld\n", fetch_ifq_full);
  printf("Rename ROB Full: %lld\n", rename_rob_full);
  printf("Rename IQ Full: %lld\n", rename_iq_full);
  printf("Rename LSQ Full: %lld\n", rename_lsq_full);
  printf("Issue IQ Empty: %lld\n", issue_iq_empty);
  
  printf("\n--- Resource Utilization ---\n");
  for (int t = 0; t < num_hw_threads; t++) {
    if (tctx[t].active || sim_num_insn_tid[t] > 0) {
      printf("Thread %d Resource Utilization:\n", t);
      printf("  Average IFQ Occupancy: %.1f\n", perf_counters[t].avg_ifq_occupancy);
      printf("  Average ROB Occupancy: %.1f\n", perf_counters[t].avg_rob_occupancy);
      printf("  Average IQ Occupancy: %.1f\n", perf_counters[t].avg_iq_occupancy);
      printf("  Average LSQ Occupancy: %.1f\n", perf_counters[t].avg_lsq_occupancy);
    }
  }
  
  /* SMT Interference Analysis */
  printf("\n--- SMT Interference Analysis ---\n");
  if (num_hw_threads > 1) {
    double single_thread_ipc = safe_ratio(sim_num_insn_tid[0], cycles);
    double mt_ipc = safe_ratio(sim_num_insn, cycles);
    double throughput_gain = mt_ipc / (single_thread_ipc * num_hw_threads);
    
    printf("Single-thread IPC estimate: %.3f\n", single_thread_ipc);
    printf("Multi-thread total IPC: %.3f\n", mt_ipc);
    printf("Throughput efficiency: %.1f%%\n", throughput_gain * 100.0);
    
    /* Cache interference */
    double avg_il1_miss = 0, avg_dl1_miss = 0;
    for (int t = 0; t < num_hw_threads; t++) {
        avg_il1_miss += safe_ratio(il1_misses_tid[t], il1_hits_tid[t] + il1_misses_tid[t]);
        avg_dl1_miss += safe_ratio(dl1_misses_tid[t], dl1_hits_tid[t] + dl1_misses_tid[t]);
    }
    avg_il1_miss /= num_hw_threads;
    avg_dl1_miss /= num_hw_threads;
    
    printf("Average I-Cache miss rate: %.1f%%\n", avg_il1_miss * 100.0);
    printf("Average D-Cache miss rate: %.1f%%\n", avg_dl1_miss * 100.0);
  }
}
/* =====  M A I N   L O O P ========================================== */
void sim_main(void) {
  init_enhanced_simulator();
  init_advanced_features();
  while (1) {
    /* ---- fast‑forward window ---- */
    if (fastfwd && warmup < fastfwd){
      /* skip timing stats but still drive pipeline */
      fetch_stage();
      rename_stage();
      warmup++;
      continue;
    } else {    /* ---- timed region ---- */
      /* Process MSHR events first */
      if (mshr_events && cycles % 10 == 0) process_mshr_events();
      /* Handle precise exceptions */
      if (cycles % 100 == 0) handle_precise_exceptions();
      /* Update resource partitions every 1000 cycles */
      if (enable_dynamic_partitioning && cycles % 1000 == 0) {
          update_resource_partitions();
      }
      /* Update performance counters */
      if (cycles % 100 == 0) update_performance_counters();
     
      /* Pipeline stages */
      commit_stage();
      writeback_stage();
      check_branch_misprediction();
      check_load_store_violations();
      address_generation_stage();
      issue_stage();
      rename_stage();
      fetch_stage();


      /* Runahead execution for stalled threads */
      /* if (enable_runahead_execution) {
        for (int t = 0; t < num_hw_threads; t++) {
          if (tctx[t].active) {
             Check if thread is stalled on memory 
            int stalled_on_memory = 0;
            for (int i = lsq_head; i != lsq_tail; i = (i + 1) % LSQ_SIZE) {
              if (LSQ[i].tid == t && LSQ[i].is_load && 
                !LSQ[i].addr_ready && cycles - LSQ[i].addr_ready_cycle > 10) {
                stalled_on_memory = 1;
                break;
              }
            }
            
            if (stalled_on_memory && !runahead_state[t].runahead_mode) {
              enter_runahead_mode(t, tctx[t].pc);
            } else if (runahead_state[t].runahead_mode) {
              execute_runahead_instruction(t);
            }
          }
        }
      } */

      cycles++;
    }

    int any_active = 0;
    int any_pending = 0;
    for (int t = 0; t < num_hw_threads; ++t) {
      if (tctx[t].active) {
        any_active = 1;
      }
      if (!thread_empty(t)) {
        any_pending = 1;
      }
    }
      
    if (!any_active && !any_pending) {
      break;
    }
    
    if (sim_max_insn && sim_num_insn >= sim_max_insn) {
      break;
    }
    
    /* prevent infinite loop */
    if (cycles > sim_max_insn * 10 + 1000000) {
      fprintf(stderr, "Warning: Simulation appears stuck, terminating at cycle %lld\n", cycles);
      break;
    }
  }
  /* Final performance analysis */
  print_performance_analysis();
}

/* =====  S T A G E   S T U B S ====================================== */
static int fetch_tid_rr = 0; /* round‑robin pointer */
static void fetch_stage(void) {
  /* simple rr policy */
  bool in_timing = (fastfwd == 0) || (warmup >= fastfwd);
  int fetch_order[MAX_HW_THREAD];
  int fetch_count = 0;
  for (int t=0;t<num_hw_threads;++t){
    if (!tctx[t].active) continue;

    int adjusted_icount = tctx[t].icount;
    if (cycles < tctx[t].last_flush_cycle + 50) {
      adjusted_icount += 10;
    }

    int pos = fetch_count;
    for (int i=0;i<fetch_count; ++i){
      int other_adjusted = tctx[fetch_order[i]].icount;
      if (cycles < tctx[fetch_order[i]].last_flush_cycle + 50) {
        other_adjusted += 10;
      }
      if (adjusted_icount < other_adjusted) {
        pos = i;
        break;
      }
    }

    for (int i = fetch_count; i > pos; --i) {
      fetch_order[i] = fetch_order[i-1];
    }
    fetch_order[pos] = t;
    fetch_count++;
  }

  int fetched = 0;
  for (int i = 0; i < fetch_count && fetched < 4; ++i){
    int tid = fetch_order[i];
    if (((ifq_tail+1)%IFQ_SIZE)==ifq_head) {
      if (in_timing) fetch_ifq_full++;
      break;
    }
    md_inst_t inst;
    tick_t ready_time;

    /* TLB access for instruction fetch */
    md_addr_t physical_pc = tctx[tid].pc;
    if (itlb) {
      tlb_access_result_t tlb_result = tlb_access(itlb, tctx[tid].pc, tid, &physical_pc);
      if (tlb_result == TLB_PAGE_FAULT) continue; /* I-TLB miss stall */
    }

    /* I-Cache access */
    bool cache_hit = TRUE;
    if (il1_cache) {
      cache_access_result_t result = cache_access(il1_cache, tctx[tid].pc,0,tid, &ready_time);
      tctx[tid].icache_accesses++;

      if (result != CACHE_HIT) {
        tctx[tid].icache_misses++;
        if (ready_time > cycles + 10) {
          continue;
        }
        cache_hit = FALSE;
      }
    }

    mem_access(mem, Read, tctx[tid].pc, &inst, sizeof(md_inst_t));
          
    /* Branch Prediction */
    enum md_opcode op;
    MD_SET_OPCODE(op, inst);
    
    md_addr_t next_pc = tctx[tid].pc + sizeof(md_inst_t); /* Default next PC */

    int opcode = (inst >> 26) & 0x3F;
    int is_branch = (opcode >= 0x30 && opcode <= 0x3F);

    if (MD_OP_FLAGS(op) & F_CTRL) {
      md_addr_t pred_target;
      int pred_taken = predict_branch(tctx[tid].pc, &pred_target);
      
      if (pred_taken) next_pc = pred_target;
      /* For simplicity, assume perfect branch prediction during fetch */
      /*md_addr_t actual_target;
      int actual_taken = resolve_branch(inst, tctx[tid].pc, 
                                                &tctx[tid].regs, &actual_target);
      
      update_branch_predictor(tctx[tid].pc, actual_taken, actual_target);
      */
      tctx[tid].branches_executed++;

      /* Misprediction check */
      /* if (pred_taken != actual_taken || 
        (actual_taken && pred_target != actual_target)) {
        bp_mispred++;
        tctx[tid].branches_mispredicted++;
        
         Flush 
        flush_thread(tid);
        tctx[tid].pc = actual_taken ? actual_target : (tctx[tid].pc + 4);
        continue;
      } else {
        bp_correct++;
      }
      
      update PC 
      next_pc = actual_taken ? actual_target : next_pc; */
    }

    /* Store current PC in IFQ, update to next PC */
    IFQ[ifq_tail] = (struct ifq_entry){inst, tctx[tid].pc, tid};
    ifq_tail = (ifq_tail + 1) % IFQ_SIZE;
    tctx[tid].pc = next_pc;
    tctx[tid].icount++;
    fetched++;
    
    if (in_timing) fair_rounds++;
  } 
}

static inline int alpha_dest_reg(md_inst_t inst)
{
  return inst & 0x1F;   /* bits [4:0] */
}
static inline int alpha_src1(md_inst_t inst){ return (inst >> 21) & 0x1F; }
static inline int alpha_src2(md_inst_t inst){ return (inst >> 16) & 0x1F; }
static void rename_stage()  {
  int renamed = 0;
  while (renamed < sim_outorder_width) {
    if (ifq_head == ifq_tail) break; /* IFQ empty */

    if (((rob_tail_global+1)%ROB_SIZE)==rob_head_global) {
      rename_rob_full++;
      break; /* ROB full */
    }

    if (iq_cnt == IQ_SIZE) {
        rename_iq_full++;
        break;
    }

    /* pop IFQ */
    struct ifq_entry fe = IFQ[ifq_head];
    ifq_head = (ifq_head+1)%IFQ_SIZE;

    int tid = fe.tid;
    if (!tctx[tid].active) continue;

    /* enqueue ROB */
    int rid = rob_tail_global;
    struct rob_entry *re = &ROB[rob_tail_global];
    memset(re, 0, sizeof(*re));
    re->tid = tid;
    re->inst = fe.inst;
    re->PC = fe.PC;
    re->seq = ++tctx[tid].seq;
    re->new_phys = -1;
    re->old_phys = -1;

    /* Source physical regs (true‑dep check later) */
    enum md_opcode op;
    MD_SET_OPCODE(op, fe.inst);

    /* Check for system calls */
    /* if (op == CALL_PAL) {
      int func = (fe.inst >> 0) & 0x3F;
      if (func == 0x83) {  OSF_SYS_exit 
        detect_exception(rid, EXCEPTION_SYSTEM_CALL, 0);
      }
    } */

    int a1 = alpha_src1(fe.inst);
    int a2 = alpha_src2(fe.inst);
    int dest = alpha_dest_reg(fe.inst);
    re->src1 = (a1 == 31) ? -1 : tctx[tid].rename_map[a1];
    re->src2 = (a2 == 31) ? -1 : tctx[tid].rename_map[a2];

    re->is_load = is_load(op);
    re->is_store = is_store(op);

    int newp=-1;
    if (dest!=31 && !is_store(op)){
      newp = prf_alloc();
      if (newp <0) { /* rollback */
        ifq_head = (ifq_head-1 + IFQ_SIZE) % IFQ_SIZE;
        break;
      }
      re->new_phys=newp;
      re->old_phys = tctx[tid].rename_map[dest];
      tctx[tid].rename_map[dest]=newp;
      prf_ready[newp]=0;
    }

    /* Allocate IQ entry */
    int iq_allocated = 0;
    for (int i=0;i<IQ_SIZE;++i) {
      if (!IQ[i].ready && !IQ[i].issued) {
        struct iq_entry *q = &IQ[i];
        memset(q,0,sizeof(*q));
        q->rob_idx = rid;
        q->tid = tid;
        q->inst = fe.inst;
        q->src1 = re->src1;
        q->src2 = re->src2;
        q->dst = newp;
        q->is_load = is_load(op);
        q->is_store = is_store(op);
        q->ready = 1;
        iq_cnt++;
        iq_allocated = 1;
        break;
      }
    }

    if (!iq_allocated) {
      if (newp != -1) prf_free(newp);
      ifq_head = (ifq_head - 1 + IFQ_SIZE) % IFQ_SIZE;
      break;
    }

    /* Allocate LSQ entry for memory operations */
    if (is_load(op)||is_store(op)) {
      if (((lsq_tail+1)%LSQ_SIZE)==lsq_head){
        rename_lsq_full++;
        rob_tail_global = rid; /* rollback */
        if (iq_allocated) iq_cnt--;
        ifq_head = (ifq_head-1+IFQ_SIZE)&IFQ_SIZE;
        break;
      }

      struct lsq_entry *lsq = &LSQ[lsq_tail];
      memset(lsq, 0, sizeof(*lsq));
      lsq->rob_idx = rid;
      lsq->tid = tid;
      lsq->is_load = is_load(op);
      lsq->is_store = is_store(op);
      lsq->size = 4;
      lsq_tail = (lsq_tail+1)%LSQ_SIZE;
    }
    renamed++;
    rob_tail_global = (rob_tail_global + 1) % ROB_SIZE;
    tctx[tid].rob_tail = rob_tail_global;
  }
}
static void issue_stage()   { 
  int issued = 0;

  for (int idx = 0; idx < IQ_SIZE && issued < sim_outorder_width; ++idx){
    struct iq_entry *q = &IQ[idx];
    if (!q->ready||q->issued) continue; /* finished entry */

    /* Check source operand readiness */
    if ((q->src1 != -1 && !prf_ready[q->src1]) || (q->src2 != -1 && !prf_ready[q->src2])) continue; /* true dependency stall */
    struct rob_entry *re = &ROB[q->rob_idx];

    enum md_opcode op;
    MD_SET_OPCODE(op, q->inst);   
    /* Schedule address generation */
    if (q->is_load || q->is_store) {
      /* Find corresponding LSQ entry */
      int lsq_idx = -1;
      for (int i = lsq_head; i != lsq_tail; i = (i + 1) % LSQ_SIZE) {
        if (LSQ[i].rob_idx == q->rob_idx && LSQ[i].tid == q->tid) {
          lsq_idx = i;
          break;
        }
      }
      
      if (lsq_idx == -1) continue; /* LSQ entry not found */
      
      struct lsq_entry *lsq = &LSQ[lsq_idx];
      
      /* Calculate address */
      int base_reg = (q->inst >> 16) & 0x1F;
      short displacement = (short)(q->inst & 0xFFFF);
      
      md_addr_t base_value = 0;
      if (base_reg != 31) {
        int phys_reg = (q->src2 != -1) ? q->src2 : base_reg;
        if (phys_reg < MD_TOTAL_REGS) {
          base_value = tctx[q->tid].regs.regs_R[base_reg];
        }
      }
      md_addr_t vaddr = base_value + displacement;
      
      lsq->vaddr = vaddr;
      lsq->addr = vaddr; /* Simple virtual = physical for now */
      lsq->addr_ready = 1;
      lsq->addr_ready_cycle = cycles;
      
      if (q->is_load) {
        forward_result_t forward_result = check_store_forwarding(lsq_idx);
        if (forward_result == FORWARD_FULL) {
          q->done = cycles + 1;
        } else {
          enhanced_lsq_access(&LSQ[lsq_idx], lsq_idx);
          q->done = lsq->done;
        }
      } else {
        /* Store - mark data ready */
        lsq->data_ready = 1;
        lsq->data_ready_cycle = cycles;  
        q->done = cycles + 1;
      }
    } else {
      /* Non-memory instruction */
      // execute_alpha_instruction(q->inst, &tctx[q->tid].regs, ROB[q->rob_idx].PC);
      q->done = cycles + get_latency(op);
    }
    q->issued = 1;
    issued++; 
  }
    
  if (issued == 0 && iq_cnt > 0) {
      issue_iq_empty++;
  }
}
static void writeback_stage(){ 
  for (int idx = 0; idx < IQ_SIZE; ++idx){
    struct iq_entry *q = &IQ[idx];
    if (!q->issued) continue;
    if (cycles < q->done) continue;

    struct rob_entry *re = &ROB[q->rob_idx];
    re->ready = 1;
    re->done_cycle = cycles;
    if (q->dst!=-1 && q->dst < PRF_NUM) prf_ready[q->dst] = 1;
    
    memset(q, 0, sizeof(*q));
    if (iq_cnt > 0) iq_cnt--;
  }
}

static void commit_stage(void)
{
  int commits = 0;                                   /* # retired this cycle */
  while (commits < sim_outorder_width) {
    if (rob_head_global == rob_tail_global) break;
    struct rob_entry *re = &ROB[rob_head_global];
    if (!re->ready) break;

    enum md_opcode op;
    MD_SET_OPCODE(op, re->inst);
    
    if (op == CALL_PAL) {
      int func = (re->inst >> 0) & 0xFFFF;
      if (func == OSF_SYS_exit || func == 0x83) {
        printf("Thread %d exiting via system call at PC 0x%llx\n", 
               re->tid, re->PC);
        tctx[re->tid].active = 0;
      }
    }
    /* if (op == CALL_PAL || (re->inst & 0xFC000000) == 0x00000000) {
      int func = re->inst & 0x3FFFFFF;
      if (func == 0x83 || func == 1) {  OSF_SYS_exit or generic exit 
        printf("Thread %d exiting at cycle %lld, PC 0x%llx\n", 
                re->tid, cycles, re->PC);
        tctx[re->tid].active = 0;
      }
    } */

    /* Store commit -> in LSQ committed mark! */
    if (re->is_store) {
      for (int i = lsq_head; i!= lsq_tail; i = (i+1)%LSQ_SIZE){
        if (LSQ[i].rob_idx == rob_head_global){
          LSQ[i].committed = 1;
          break;
        }
      }
    }

    /* Load commit -> from LSQ, eliminate */
     if (re->is_load) {
      int found = -1;
      for (int i = lsq_head; i != lsq_tail; i = (i+1)%LSQ_SIZE) {
        if (LSQ[i].rob_idx == rob_head_global) {
          found = i;
          break;
        }
      }
      if (found != -1 && found == lsq_head) {
        lsq_head = (lsq_head + 1) % LSQ_SIZE;
      }
    }

    if (re->old_phys != -1) prf_free(re->old_phys);

    sim_num_insn++;
    sim_num_insn_tid[re->tid]++;
    commits++;
    if (MD_OP_FLAGS(op) & F_CTRL) {
      md_addr_t actual_target;
      int actual_taken = resolve_branch(re->inst, re->PC,&tctx[re->tid].regs, &actual_target);
      // If mispredicted, flush will happen in check_load_store_violations
    }
    rob_head_global = (rob_head_global + 1) % ROB_SIZE;
    tctx[re->tid].rob_head = rob_head_global;
    memset(re, 0, sizeof(*re));
  }

  /* Eliminate committed stores from LSQ */
  while (lsq_head != lsq_tail && LSQ[lsq_head].is_store && LSQ[lsq_head].committed) {
    lsq_head = (lsq_head + 1) % LSQ_SIZE;
  }
}
/* ===== T H R E A D ================================================== */
static void flush_thread(int tid){
  if (tid < 0 || tid >= num_hw_threads) return;

  tctx[tid].last_flush_cycle = cycles;
  tctx[tid].flush_count++;
  tctx[tid].speculation_depth = 0;

  /* Reset PC to architectural state */
  tctx[tid].pc = tctx[tid].speculative_pc;
  
  /* Comprehensive pipeline flush */

  /* Flush IFQ */
  int dst = ifq_head;
  for (int src = ifq_head; src != ifq_tail; src = (src + 1) % IFQ_SIZE){
    if (IFQ[src].tid == tid) continue;
    IFQ[dst] = IFQ[src];
    dst = (dst + 1) % IFQ_SIZE;
  }
  ifq_tail = dst;

  /* Flush IQ */
  for (int i=0;i<IQ_SIZE;++i){
    if (IQ[i].ready && IQ[i].tid == tid) {
        memset(&IQ[i], 0,sizeof(IQ[i]));
        iq_cnt--;
    }
  }

  /* Flush AGU */
  for (int i=0; i< AGU_SIZE; i++) {
    if (AGU[i].valid && AGU[i].tid == tid) {
      AGU[i].valid = 0;
    }
  }

  /* Flush LSQ */
  dst = lsq_head;
  for (int src = lsq_head; src != lsq_tail; src = (src + 1) % LSQ_SIZE) {
    if (LSQ[src].tid == tid) continue;
    LSQ[dst] = LSQ[src];
    dst = (dst + 1) % LSQ_SIZE;
  }
  lsq_tail = dst;

  /* Handle ROB entries */
  for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE){
    if (ROB[i].tid != tid) continue;
    if (!ROB[i].ready && ROB[i].new_phys != -1){
      prf_free(ROB[i].new_phys);
    }
    ROB[i].ready = 1; /* Mark for commit */
  }

  /* Restore rename map to architectural state */
  for (int r = 0; r < MD_TOTAL_REGS; r++) {
    tctx[tid].rename_map[r] = r;
  }
  /* Reset fetch counters */
  tctx[tid].icount = 0;
  printf("Thread %d flushed at cycle %lld (flush #%lld)\n", 
           tid, cycles, tctx[tid].flush_count);
}
/* Smart Thread Scheduling with Flush Awareness */
static int smart_thread_selection(int *fetch_order, int *fetch_count) {
  *fetch_count = 0;
  
  /* Priority-based thread selection */
  typedef struct {
    int tid;
    int priority;
    int icount;
  } thread_priority_t;
  
  thread_priority_t candidates[MAX_HW_THREAD];
  int candidate_count = 0;
  
  for (int t = 0; t < num_hw_threads; ++t) {
    if (!tctx[t].active) continue;
    
    candidates[candidate_count].tid = t;
    candidates[candidate_count].icount = tctx[t].icount;
    
    /* Calculate priority based on multiple factors */
    int priority = 100; /* Base priority */
    
    /* Penalize recently flushed threads */
    if (cycles - tctx[t].last_flush_cycle < 10) {
      priority -= 30;
    }
    
    /* Favor threads with lower instruction counts (ICOUNT policy) */
    priority -= tctx[t].icount;
      
      /* Bonus for threads with good branch prediction */
    if (tctx[t].branches_executed > 0) {
      int bp_accuracy = (tctx[t].branches_executed - 
                        tctx[t].branches_mispredicted) * 100 / 
                        tctx[t].branches_executed;
      if (bp_accuracy > 90) priority += 10;
      else if (bp_accuracy < 70) priority -= 10;
    }
      
    candidates[candidate_count].priority = priority;
    candidate_count++;
  }
  
  /* Sort by priority (descending) */
  for (int i = 0; i < candidate_count - 1; i++) {
    for (int j = i + 1; j < candidate_count; j++) {
      if (candidates[j].priority > candidates[i].priority) {
        thread_priority_t temp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = temp;
      }
    }
  }
  
  /* Fill fetch order */
  for (int i = 0; i < candidate_count; i++) {
      fetch_order[i] = candidates[i].tid;
  }
  *fetch_count = candidate_count;
  
  return (candidate_count > 0) ? fetch_order[0] : -1;
}