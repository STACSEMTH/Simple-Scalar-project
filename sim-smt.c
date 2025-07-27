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

#define BTB_SIZE 512
#define IFQ_SIZE 64 // 16 -> 64
#define IQ_SIZE 64 // 32 -> 64
#define LSQ_SIZE 64 // 32 -> 64
#define ROB_SIZE 256 // 128 -> 256
#define AGU_SIZE 8
#define MEM_DEP_TABLE_SIZE 256
#define SLAP_SIZE 1024
#define PREFETCH_QUEUE_SIZE 16
#define EXCEPTION_BUFFER_SIZE 8
#define LOCAL_PRED_SIZE 1024
#define LOCAL_HIST_SIZE 1024
#define GLOBAL_PRED_SIZE 4096
#define CHOICE_PRED_SIZE 4096
#define RAS_SIZE 32
#define STRIDE_TABLE_SIZE 64
#define GLOBAL_HIST_LEN 13
#define MAX_FETCH_PER_CYCLE sim_outorder_width     // Maximum fetches per cycle
#define MIN_FETCH_PER_THREAD 1    // Minimum fetches per thread
#define ADAPTIVE_FETCH_THRESHOLD 0.7 // Fetch if >70% of threads are active

typedef struct adaptive_fetch_control {
    int total_fetch_budget;         
    int thread_fetch_quota[MAX_HW_THREAD];  
    double thread_fetch_priority[MAX_HW_THREAD];  
    int consecutive_ifq_full[MAX_HW_THREAD];  
    tick_t last_quota_update;        
} adaptive_fetch_control_t;

static adaptive_fetch_control_t fetch_ctrl;
/* 2-bit saturating counter */
typedef enum {
  STRONGLY_NOT_TAKEN = 0,
  WEAKLY_NOT_TAKEN = 1,
  WEAKLY_TAKEN = 2,
  STRONGLY_TAKEN = 3
} branch_state_t;

/* Predictor Types */
typedef enum {
  PRED_LOCAL = 0,
  PRED_GSHARE = 1,
  PRED_TOURNAMENT = 2
} predictor_type_t;

/* Branch Predictor Structures */
typedef struct {
  branch_state_t state;
  unsigned local_history;
  md_addr_t tag;
  int valid;
  int confidence;
  tick_t last_update;
} local_predictor_entry_t;

typedef struct {
  branch_state_t state;
  int confidence;
} global_predictor_entry_t;

typedef struct {
  md_addr_t target;
  md_addr_t tag;
  int valid;
} btb_entry_t;

typedef struct {
  branch_state_t state;
  int bias; /* 0=favor local, 1=favor gshare */
} choice_entry_t;

/* Return Address Stack */
typedef struct {
  md_addr_t stack[RAS_SIZE];
  int top;
  int size;
} ras_t;

static btb_entry_t btb[BTB_SIZE];
static predictor_type_t current_predictor_type = PRED_TOURNAMENT;

/* Branch Predictor Statistics */
static counter_t local_correct = 0, local_wrong = 0;
static counter_t gshare_correct = 0, gshare_wrong = 0;
static counter_t tournament_correct = 0, tournament_wrong = 0;
static counter_t ras_hits = 0, ras_misses = 0;
static counter_t conditional_branches = 0;
static counter_t unconditional_branches = 0;
static counter_t function_calls = 0;
static counter_t function_returns = 0;
static counter_t bp_lookups = 0;
static counter_t bp_correct = 0;
static counter_t bp_mispred = 0;
static counter_t btb_hits = 0;
static counter_t btb_misses = 0;
/* Scheduling Policies */
typedef enum {
  SCHED_ROUND_ROBIN = 0,
  SCHED_ICOUNT = 1,
  SCHED_PERFORMANCE_FEEDBACK = 2,
  SCHED_ADAPTIVE = 3
} scheduling_policy_t;

/* Thread Performance Tracking */
typedef struct {
  /* Short-term metrics (sliding window) */
  double recent_ipc;
  double recent_branch_accuracy;
  double recent_cache_miss_rate;
  int recent_flush_count;
  
  /* Long-term metrics */
  double avg_ipc;
  double total_progress;
  counter_t total_cycles_active;
  
  /* Resource usage */
  int ifq_usage;
  int rob_usage;
  int iq_usage;
  int lsq_usage;
  double resource_efficiency;
  
  /* Fairness metrics */
  counter_t cycles_since_last_fetch;
  counter_t total_fetch_cycles;
  double fairness_score;
  
  /* Penalty tracking */
  int flush_penalty_remaining;
  int cache_miss_penalty;
  int resource_starvation_penalty;
  
  /* Priority and age */
  int base_priority;
  int dynamic_priority;
  int age_boost;
  
  /* Scheduling history */
  tick_t last_fetch_cycle;
  tick_t last_issue_cycle;
  tick_t last_commit_cycle;
} thread_performance_t;

/* Scheduling State */
typedef struct {
  scheduling_policy_t current_policy;
  int fetch_round_robin_ptr;
  int issue_round_robin_ptr;
  
  /* Performance feedback */
  double system_throughput;
  double system_fairness;
  double resource_utilization;
  
  /* Adaptation parameters */
  int adaptation_interval;
  int cycles_since_adaptation;
  
  /* Starvation prevention */
  int max_starvation_cycles;
  int fairness_boost_threshold;
} scheduler_state_t;

/* Global scheduler state */
static thread_performance_t thread_perf[MAX_HW_THREAD];
static scheduler_state_t scheduler;
/* For statistics - store as scaled integers */
static counter_t recent_ipc_scaled[MAX_HW_THREAD];
static counter_t resource_efficiency_scaled[MAX_HW_THREAD];
static counter_t fairness_score_scaled[MAX_HW_THREAD];
/* Configuration */
static scheduling_policy_t sched_policy = SCHED_ADAPTIVE;
static int enable_fairness_boost = 1;
static int enable_performance_feedback = 1;
static int starvation_threshold = 500; /* cycles */

/* Performance Window Configuration */
#define PERF_WINDOW_SIZE 1000
#define ADAPTATION_INTERVAL 10000
/* ===== I F Q ( I N S T R U C T I O N    F E T C H    Q U E U E) ========================= */
struct ifq_entry {
  md_inst_t inst;				/* inst register */
  md_addr_t PC;		/* current PC, predicted next PC */
  int tid;
};
/* Runahead Execution */
/* ===== I Q ( I N S T R U C T I O N     Q U E U E) ========================= */
struct iq_entry {
  int valid; /* 1 = valid, 0 = invalid */
  md_inst_t inst;
  int rob_idx, tid;
  int src1;
  int src2;
  int dst;
  unsigned is_load:1, is_store:1;
  tick_t done;
  char issued, ready;
};

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
  int valid; /* 1 = valid, 0 = invalid */
  unsigned size:4;
  unsigned is_load:1;
  unsigned is_store:1;
  unsigned addr_ready:1;  
  unsigned data_ready:1;  /* store data prepared */
  unsigned forwarded:1;    
  unsigned committed:1;   /* store commited */
  tick_t addr_ready_cycle;  
  tick_t data_ready_cycle;
  tick_t done;
};

typedef struct {
    int runahead_mode;
    md_addr_t runahead_pc;
    md_addr_t normal_pc;
    struct regs_t checkpoint_regs;
    tick_t runahead_start_cycle;
    int prefetches_generated;
} runahead_state_t;

/* Address Generation Unit */
typedef struct {
  md_addr_t addr;
  int rob_idx;
  int tid;
  tick_t ready_cycle;
  int valid;
} agu_entry_t;

static agu_entry_t AGU[AGU_SIZE];
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

/* IQ, LSQ, BTB, caches… share similar tid tagging */
/* ===== M E M O R Y     D E P E N D E N C E     P R E D I C T I O N ======== */
typedef struct {
  md_addr_t load_pc;
  md_addr_t store_pc;
  int confidence;
  tick_t last_update;
} mem_dep_entry_t;

static mem_dep_entry_t mem_dep_table[MEM_DEP_TABLE_SIZE];
static int enable_dynamic_partitioning = 1;
static int enable_stride_prefetcher = 1;
static int enable_runahead_execution = 0;
static int memory_dependency_prediction = 1;
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
/* ===== Advanced Prefetcher ===== */

typedef struct stride_entry {
    md_addr_t pc;
    md_addr_t last_addr;
    int stride;
    int confidence;
    int active;
} stride_entry_t;

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

struct thread_pipeline {
    /* IFQ - instruction fetch queue */
    struct ifq_entry IFQ[IFQ_SIZE];
    int ifq_head, ifq_tail;
    
    /* IQ - instruction queue */
    struct iq_entry IQ[IQ_SIZE];
    int iq_head, iq_tail;
    
    /* LSQ - load/store queue */
    struct lsq_entry LSQ[LSQ_SIZE];
    int lsq_head, lsq_tail;
    
    /* ROB - reorder buffer */
    struct rob_entry ROB[ROB_SIZE];
    int rob_head, rob_tail;
    
    /* AGU - Address Generation Unit */
    agu_entry_t AGU[AGU_SIZE];
    
    /* 스레드별 성능 카운터 */
    counter_t fetch_ifq_full;
    counter_t rename_rob_full;
    counter_t rename_iq_full;
    counter_t rename_lsq_full;
    counter_t issue_iq_empty;
    
    /* 스레드별 LSQ 통계 */
    counter_t lsq_store_forwards;
    counter_t lsq_load_violations;
    counter_t lsq_addr_conflicts;
    counter_t lsq_partial_forwards;

    mem_dep_entry_t mem_dep_table[MEM_DEP_TABLE_SIZE];
    struct slap_entry slap[SLAP_SIZE];
    stride_entry_t stride_table[STRIDE_TABLE_SIZE];
    prefetch_request_t prefetch_queue[PREFETCH_QUEUE_SIZE];
    int prefetch_head, prefetch_tail;
    counter_t prefetches_issued, prefetches_useful, prefetches_late;
    exception_entry_t exception_buffer[EXCEPTION_BUFFER_SIZE];
    int exception_head, exception_tail;
};

struct thread_branch_predictor {
    local_predictor_entry_t local_predictor[LOCAL_PRED_SIZE];
    unsigned local_history_table[LOCAL_HIST_SIZE];
    global_predictor_entry_t global_predictor[GLOBAL_PRED_SIZE];
    choice_entry_t choice_predictor[CHOICE_PRED_SIZE];
    unsigned global_history;
    ras_t return_address_stack;
    predictor_type_t current_predictor_type;
    
    /* 스레드별 브랜치 통계 */
    counter_t bp_lookups, bp_correct, bp_mispred;
    counter_t btb_hits, btb_misses;
    counter_t local_correct, local_wrong;
    counter_t gshare_correct, gshare_wrong;
    counter_t tournament_correct, tournament_wrong;
    counter_t ras_hits, ras_misses;
    counter_t conditional_branches, unconditional_branches;
    counter_t function_calls, function_returns;
};
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
static counter_t dcache_hits_actual[MAX_HW_THREAD] = {0};  
static counter_t dcache_forwarding[MAX_HW_THREAD] = {0};    
static counter_t dcache_accesses_total[MAX_HW_THREAD] = {0};
typedef struct resource_tracker {
    double ifq_occupancy_sum;
    double rob_occupancy_sum;
    double iq_occupancy_sum;
    double lsq_occupancy_sum;
    counter_t sample_count;
    
    int ifq_samples[100];
    int rob_samples[100];
    int iq_samples[100];
    int lsq_samples[100];
    int sample_index;
} resource_tracker_t;

static resource_tracker_t resource_tracker[MAX_HW_THREAD];

typedef struct stall_stats {
    counter_t fetch_stalls;
    counter_t rename_stalls;
    counter_t issue_stalls;
    counter_t active_cycles;  
} stall_stats_t;
static stall_stats_t thread_stall_stats[MAX_HW_THREAD];
/* ===== T H R E A D   C O N T E X T ================================== */
struct thread_ctx {
  md_addr_t pc;
  struct regs_t regs;             /* architectural registers */
  int rename_map[MD_TOTAL_REGS];  /* arch→phys map */
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
  struct thread_branch_predictor bp;
  struct thread_pipeline pipeline;
  thread_performance_t perf;
  runahead_state_t runahead;
  performance_counters_t perf_counters;
};
static struct thread_ctx tctx[MAX_HW_THREAD];
static int num_hw_threads = 1;
static int smart_thread_selection(int *fetch_order, int *fetch_count);


static void update_memory_dependence_predictor(int tid, md_addr_t load_pc, md_addr_t store_pc, 
  int violation_occurred) {
  if (!memory_dependency_prediction) return;      
  struct thread_pipeline *tp = &tctx[tid].pipeline;                                         
  unsigned idx = (load_pc ^ store_pc) % MEM_DEP_TABLE_SIZE;
  
  mem_dep_entry_t *entry = &tp->mem_dep_table[idx];
  
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

static inline int is_load(enum md_opcode op){
    /* Alpha: primary opcode 0x08~0x0F = LDQ/LDSx … */
    return (MD_OP_FLAGS(op) & F_LOAD) != 0;
}
static inline int is_store(enum md_opcode op){
    return (MD_OP_FLAGS(op) & F_STORE) != 0;   /* STQ/STx */
}

/* ===== F O R W A R D I N G ======================================= */
static bool can_issue_load_safely(int tid, struct iq_entry *load_iq, int load_lsq_idx) {
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  struct lsq_entry *load = &tp->LSQ[load_lsq_idx];
  
  if (!load->addr_ready) return false;
  
  int unknown_stores = 0;
  int total_stores = 0;
  
  // Check older stores with RELAXED policy
  for (int i = tp->lsq_head; i != load_lsq_idx; i = (i + 1) % LSQ_SIZE) {
    struct lsq_entry *older_store = &tp->LSQ[i];
    
    if (older_store->is_store) {
      total_stores++;
      
      if (!older_store->addr_ready) {
        unknown_stores++;
        
        // CRITICAL FIX: Try to calculate store address early if possible
        struct rob_entry *store_re = &tp->ROB[older_store->rob_idx];
        if (store_re && !older_store->addr_ready) {
          // Try early address calculation for simple addressing modes
          int base_reg = (store_re->inst >> 16) & 0x1F;
          short displacement = (short)(store_re->inst & 0xFFFF);
          
          // If base register is ready, calculate address immediately
          if (base_reg == 31 || prf_ready[tctx[tid].rename_map[base_reg]]) {
            md_addr_t base_value = (base_reg == 31) ? 0 : tctx[tid].regs.regs_R[base_reg];
            md_addr_t store_addr = base_value + displacement;
            
            // Early address resolution
            older_store->addr = store_addr;
            older_store->vaddr = store_addr;
            older_store->addr_ready = 1;
            older_store->addr_ready_cycle = cycles;
            unknown_stores--;
            
            printf("EARLY ADDR CALC: Store@0x%llx resolved early at cycle %lld\n", 
                   store_addr, cycles);
          }
        }
      }
      
      // If this store has known address and overlaps, check data readiness
      if (older_store->addr_ready && 
          addr_overlap(older_store->addr, older_store->size, load->addr, load->size)) {
        
        if (!older_store->data_ready) {
          printf("DATA DEPENDENCY: Load@0x%llx waits for Store@0x%llx data at cycle %lld\n", 
                 load->addr, older_store->addr, cycles);
          return false; // Known conflict - must wait
        }
      }
    }
  }
  
  // RELAXED POLICY: Allow load if most stores are resolved OR very few unknown stores
  if (total_stores == 0) {
    return true; // No stores to worry about
  }
  
  double unknown_ratio = (double)unknown_stores / total_stores;
  
  // Allow speculative execution if:
  // 1. Less than 25% of stores have unknown addresses, OR
  // 2. Only 1-2 stores have unknown addresses (regardless of ratio)
  if (unknown_ratio <= 0.25 || unknown_stores <= 2) {
    if (unknown_stores > 0) {
      printf("SPECULATIVE LOAD: Load@0x%llx proceeds with %d/%d unknown stores at cycle %lld\n",
             load->addr, unknown_stores, total_stores, cycles);
    }
    return true; // Allow speculative execution
  }
  
  // Too many unknown stores - conservative stall
  printf("CONSERVATIVE STALL: Load@0x%llx waits for %d/%d stores at cycle %lld\n",
         load->addr, unknown_stores, total_stores, cycles);
  return false;
}
static forward_result_t check_store_forwarding(int tid, int load_lsq_idx) {
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  struct lsq_entry *load = &tp->LSQ[load_lsq_idx];
  
  if (!load->is_load || !load->addr_ready) return FORWARD_NONE;
  if (load->forwarded) return FORWARD_FULL;
  // Search for forwarding candidates - more conservative range
  int search_range = MIN(16, LSQ_SIZE/2);
  for (int i = 0; i < search_range; i++) {
    int store_idx = (load_lsq_idx - 1 - i + LSQ_SIZE) % LSQ_SIZE;
    if (store_idx == tp->lsq_head) break;
    struct lsq_entry *store = &tp->LSQ[store_idx];
    
    if (!store->is_store || !store->addr_ready) continue;

    if (addr_overlap(store->addr, store->size, load->addr, load->size)) {
      
      // Perfect match for full forwarding
      if (store->addr == load->addr && 
          store->size == load->size && 
          store->data_ready) {
          
        load->data = store->data;
        load->forwarded = 1;
        load->done = cycles + 1; // Fast forwarding
        
        // CRITICAL: Signal ROB entry completion immediately
        struct rob_entry *load_re = &tp->ROB[load->rob_idx];
        load_re->ready = 1;
        load_re->done_cycle = cycles + 1;
        
        tp->lsq_store_forwards++;
        lsq_store_forwards++;
        
        printf("FORWARD SUCCESS: Load@0x%llx from Store@0x%llx at cycle %lld\n",
               load->addr, store->addr, cycles);
        return FORWARD_FULL;
      }
      
      // Partial forwarding case
      else if (store->data_ready) {
        md_addr_t overlap_start = MAX(store->addr, load->addr);
        md_addr_t overlap_end = MIN(store->addr + store->size, 
                                  load->addr + load->size);
        int overlap_size = overlap_end - overlap_start;
        
        if (overlap_size > 0 && overlap_size < load->size) {
          load->done = cycles + 25; // Partial forward penalty
          tp->lsq_partial_forwards++;
          lsq_partial_forwards++;
          return FORWARD_PARTIAL;
        }
      }
      
      // Address conflict but data not ready
      else {
        printf("FORWARD CONFLICT: Load@0x%llx conflicts with Store@0x%llx (data not ready) at cycle %lld\n",
               load->addr, store->addr, cycles);
        load->done = cycles + 35; // Conflict penalty
        tp->lsq_addr_conflicts++;
        lsq_addr_conflicts++;
        return FORWARD_CONFLICT;
      }
    }
  }
  
  // No forwarding - proceed to cache
  return FORWARD_NONE;
}
static void enhanced_flush_thread(int tid);
static void check_load_store_violations(int tid) {
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  
  for (int i = tp->rob_head; i != tp->rob_tail; i = (i + 1) % ROB_SIZE) {
    struct rob_entry *re = &tp->ROB[i];
    
    if (!re->ready || !re->is_store || re->tid != tid) continue;
    
    int store_lsq_idx = -1;
    for (int j = tp->lsq_head; j != tp->lsq_tail; j = (j + 1) % LSQ_SIZE) {
      if (tp->LSQ[j].rob_idx == i && tp->LSQ[j].tid == tid) {
        store_lsq_idx = j;
        break;
      }
    }
    
    if (store_lsq_idx == -1 || !tp->LSQ[store_lsq_idx].addr_ready) continue;
    struct lsq_entry *store = &tp->LSQ[store_lsq_idx];
    

    bool violation_found = false;
    for (int j = (store_lsq_idx + 1) % LSQ_SIZE; 
        j != tp->lsq_tail && !violation_found; 
        j = (j + 1) % LSQ_SIZE) {
      struct lsq_entry *load = &tp->LSQ[j];
      
      if (!load->is_load || !load->addr_ready || load->tid != tid) continue;
      if (load->forwarded) continue; 
      
      if (store->addr == load->addr && store->size == load->size) {
        if (!store->committed) continue;
        
        tp->lsq_load_violations++;
        lsq_load_violations++;
        
        struct rob_entry *load_re = &tp->ROB[load->rob_idx];
        update_memory_dependence_predictor(tid, load_re->PC, re->PC, 1);
        
        printf("LSQ VIOLATION: Load PC=0x%llx vs Store PC=0x%llx addr=0x%llx at cycle %lld\n",
               load_re->PC, re->PC, load->addr, cycles);
        
        enhanced_flush_thread(tid);
        tctx[tid].pc = load_re->PC; /* Load 명령어로 되돌림 */
        violation_found = true;
      }
    }
    
    if (violation_found) return; /* 한 번에 하나의 violation만 처리 */
  }
}
/* =====  B R A C N C H ============================================== */
/* Predeclaration */
static int get_branch_type(md_inst_t inst);
static void adapt_scheduling_policy();
static void dynamic_resource_allocation();
static void analyze_scheduling_performance();

static void init_branch_predictor() {
  for (int i = 0; i < BTB_SIZE; i++) {
    btb[i].valid = 0;
    btb[i].tag = 0;
    btb[i].target = 0;
  }
  memset(mem_dep_table, 0, sizeof(mem_dep_table));
  bp_lookups = 0;
  bp_correct = 0;
  bp_mispred = 0;
  btb_hits = 0;
  btb_misses = 0;
  conditional_branches = 0;
  unconditional_branches = 0;
  function_calls = 0;
  function_returns = 0;
  local_correct = 0;
  local_wrong = 0;
  gshare_correct = 0;
  gshare_wrong = 0;
  tournament_correct = 0;
  tournament_wrong = 0;
  ras_hits = 0;
  ras_misses = 0;
  current_predictor_type = PRED_TOURNAMENT;
  printf("Enhanced branch predictor initialized:\n");
  printf("  Tournament predictor with local + gshare\n");
  printf("  BTB: %d entries\n", BTB_SIZE);
  printf("  Memory dependence predictor: %d entries\n", MEM_DEP_TABLE_SIZE);
}

/* Local Predictor Functions */
static int local_predict_for_thread(int tid, md_addr_t pc) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    unsigned pc_idx = (pc >> 2) % LOCAL_PRED_SIZE;
    unsigned hist_idx = (pc >> 2) % LOCAL_HIST_SIZE;
    unsigned local_hist = bp->local_history_table[hist_idx] & ((1 << 10) - 1);
    unsigned pattern_idx = ((pc >> 2) ^ local_hist) % LOCAL_PRED_SIZE;
    
    return (bp->local_predictor[pattern_idx].state >= WEAKLY_TAKEN) ? 1 : 0;
}

static void local_update_for_thread(int tid, md_addr_t pc, int taken) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    unsigned hist_idx = (pc >> 2) % LOCAL_HIST_SIZE;
    unsigned local_hist = bp->local_history_table[hist_idx];
    unsigned pattern_idx = ((pc >> 2) ^ local_hist) % LOCAL_PRED_SIZE;
    /* Update predictor state first */
    if (taken) {
        if (bp->local_predictor[pattern_idx].state < STRONGLY_TAKEN) 
            bp->local_predictor[pattern_idx].state++;
    } else {
        if (bp->local_predictor[pattern_idx].state > STRONGLY_NOT_TAKEN) 
            bp->local_predictor[pattern_idx].state--;
    }
    bp->local_predictor[pattern_idx].last_update = cycles;
    /* Then update history */
    bp->local_history_table[hist_idx] = 
        ((bp->local_history_table[hist_idx] << 1) | (taken ? 1 : 0)) & ((1 << 10) - 1);
}

/* Gshare Predictor Functions */
static int gshare_predict_for_thread(int tid, md_addr_t pc) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    unsigned gshare_idx = ((pc >> 2) ^ bp->global_history) % GLOBAL_PRED_SIZE;
    return (bp->global_predictor[gshare_idx].state >= WEAKLY_TAKEN) ? 1 : 0;
}

static void gshare_update_for_thread(int tid, md_addr_t pc, int taken) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    unsigned gshare_idx = ((pc >> 2) ^ bp->global_history) % GLOBAL_PRED_SIZE;
    
    if (taken) {
        if (bp->global_predictor[gshare_idx].state < STRONGLY_TAKEN) 
            bp->global_predictor[gshare_idx].state++;
    } else {
        if (bp->global_predictor[gshare_idx].state > STRONGLY_NOT_TAKEN) 
            bp->global_predictor[gshare_idx].state--;
    }
}
/* Return Address Stack Functions */
static void ras_push_for_thread(int tid, md_addr_t return_addr) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    bp->return_address_stack.stack[bp->return_address_stack.top] = return_addr;
    bp->return_address_stack.top = (bp->return_address_stack.top + 1) % RAS_SIZE;
    if (bp->return_address_stack.size < RAS_SIZE) {
        bp->return_address_stack.size++;
    }
}

static int ras_predict_for_thread(int tid, md_addr_t pc, md_addr_t *pred_target) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    if (bp->return_address_stack.size > 0) {
        bp->return_address_stack.top = (bp->return_address_stack.top - 1 + RAS_SIZE) % RAS_SIZE;
        bp->return_address_stack.size--;
        *pred_target = bp->return_address_stack.stack[bp->return_address_stack.top];
        return 1;
    }
    *pred_target = pc + 4;
    return 0;
}
/* Tournament Predictor Functions */
static int tournament_predict_for_thread(int tid, md_addr_t pc, md_addr_t *pred_target) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    unsigned choice_idx = (pc >> 2) % CHOICE_PRED_SIZE;
    
    int local_pred = local_predict_for_thread(tid, pc);
    int gshare_pred = gshare_predict_for_thread(tid, pc);
    
    /* Use choice predictor properly */
    int use_gshare = (bp->choice_predictor[choice_idx].state >= WEAKLY_TAKEN);
    int final_pred = use_gshare ? gshare_pred : local_pred;
    
    /* BTB lookup */
    unsigned btb_idx = (pc >> 2) % BTB_SIZE;
    if (btb[btb_idx].valid && btb[btb_idx].tag == pc) {
        *pred_target = btb[btb_idx].target;
        bp->btb_hits++;
        btb_hits++;
    } else {
        *pred_target = pc + 4;
        bp->btb_misses++;
        btb_misses++;
    }
    
    return final_pred;
}

/* Enhanced Branch Type Detection */
static int get_branch_type(md_inst_t inst) {
  int opcode = (inst >> 26) & 0x3F;
  
  switch (opcode) {
    case 0x30: return 3; /* BR - unconditional */
    case 0x34: return 2; /* BSR - subroutine call */
    case 0x1A: /* JMP format */
      {
        int hint = (inst >> 14) & 0x3;
        if (hint == 0x2) return 4; /* RET - return */
        else if (hint == 0x1) return 2; /* JSR - call */
        else return 3; /* JMP - unconditional */
      }
    case 0x38: case 0x39: case 0x3A: case 0x3B:
    case 0x3C: case 0x3D: case 0x3E: case 0x3F:
    case 0x31: case 0x32: case 0x33: case 0x35:
    case 0x36: case 0x37:
      return 1; /* Conditional branch */
    default:
      return 0; /* Not a branch */
  }
}

static void update_branch_predictor_for_thread(int tid, md_addr_t pc, int taken, 
    md_addr_t actual_target, md_inst_t inst) {
    struct thread_branch_predictor *bp = &tctx[tid].bp;
    int branch_type = get_branch_type(inst);
    
    if (branch_type == 1) { /* Conditional branch */
      int local_pred = local_predict_for_thread(tid, pc);
      int gshare_pred = gshare_predict_for_thread(tid, pc);
      
      /* Update choice predictor */
      unsigned choice_idx = (pc >> 2) % CHOICE_PRED_SIZE;
      if (local_pred != gshare_pred) {
        if (local_pred == taken && gshare_pred != taken) {
          if (bp->choice_predictor[choice_idx].state > STRONGLY_NOT_TAKEN)
            bp->choice_predictor[choice_idx].state--;
          bp->local_correct++;
          local_correct++;
          bp->gshare_wrong++;
          gshare_wrong++;
        } else if (gshare_pred == taken && local_pred != taken) {
          if (bp->choice_predictor[choice_idx].state < STRONGLY_TAKEN)
            bp->choice_predictor[choice_idx].state++;
          bp->gshare_correct++;
          gshare_correct++;
          bp->local_wrong++;
          local_wrong++;
        } else {
          /* Both predictors agree */
          if (local_pred == taken) {
            bp->local_correct++;
            local_correct++;
            bp->gshare_correct++;
            gshare_correct++;
          } else {
            bp->local_wrong++;
            local_wrong++;
            bp->gshare_wrong++;
            gshare_wrong++;
          }
        }
      } else {
        /* Both predictors agree */
        if (local_pred == taken) {
          bp->local_correct++;
          local_correct++;
          bp->gshare_correct++;
          gshare_correct++;
        } else {
          bp->local_wrong++;
          bp->gshare_wrong++;
          local_wrong++;
          gshare_wrong++;
        }
      }
      
      /* Update both predictors */
      local_update_for_thread(tid, pc, taken);
      gshare_update_for_thread(tid, pc, taken);
      
      /* Update global history */
      bp->global_history = ((bp->global_history << 1) | (taken ? 1 : 0)) & 
                          ((1 << GLOBAL_HIST_LEN) - 1);
    }
    
    /* Update BTB for all control instructions */
    if (branch_type > 0) {
        unsigned btb_idx = (pc >> 2) % BTB_SIZE;
        btb[btb_idx].tag = pc;
        btb[btb_idx].target = actual_target;
        btb[btb_idx].valid = 1;
    }
}
/* Enhanced Branch Resolution */
static int resolve_branch(md_inst_t inst, md_addr_t pc, struct regs_t *regs, md_addr_t *target) {
  int opcode = (inst >> 26) & 0x3F;
  int ra = (inst >> 21) & 0x1F;  /* source register */
  
  if (opcode == 0x1A) { /* JMP format */
    int rb = (inst >> 16) & 0x1F;
    int hint = (inst >> 14) & 0x3;
    
    md_addr_t base_addr = (rb == 31) ? 0 : regs->regs_R[rb];
    int displacement = (inst & 0x3FFF) << 2;
    if (displacement & 0x8000) displacement |= 0xFFFF0000; /* sign extend 14 bits */
    
    *target = (base_addr + displacement) & ~0x3; /* word align */
    
    if (hint == 0x2) return 4; /* RET */
    else if (hint == 0x1) return 2; /* JSR */
    else return 3; /* JMP */
  } else if ((opcode >= 0x30 && opcode <= 0x3F)) {
    /* 21-bit displacement, sign extended */
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
}
static void update_global_branch_stats(int tid, md_addr_t pc, int taken, md_addr_t actual_target, md_inst_t inst) {
  int branch_type = get_branch_type(inst);
  
  /* 전역 통계 업데이트 */
  bp_lookups++;
  
  switch (branch_type) {
    case 1: /* Conditional branch */
      conditional_branches++;
      break;
    case 2: /* Function call */
      function_calls++;
      break;
    case 3: /* Unconditional branch */
      unconditional_branches++;
      break;
    case 4: /* Function return */
      function_returns++;
      break;
  }
  
  /* 스레드별 통계도 업데이트 */
  tctx[tid].bp.bp_lookups++;
  if (branch_type == 1) {
    tctx[tid].bp.conditional_branches++;
  } else if (branch_type == 2) {
    tctx[tid].bp.function_calls++;
  } else if (branch_type == 3) {
    tctx[tid].bp.unconditional_branches++;
  } else if (branch_type == 4) {
    tctx[tid].bp.function_returns++;
  }
}
static int predict_branch_for_thread(int tid, md_addr_t pc, md_addr_t *pred_target, md_inst_t inst);
static void check_branch_misprediction(void) {
  // Check committed branches for mispredictions
  /* for (int i = rob_head_global; i != rob_tail_global; i = (i + 1) % ROB_SIZE) {
    struct rob_entry *re = &ROB[i];
    if (!re->ready) continue;
    
    enum md_opcode op;
    MD_SET_OPCODE(op, re->inst);
    
    if (MD_OP_FLAGS(op) & F_CTRL) {
      md_addr_t actual_target;
      int actual_taken = resolve_branch(re->inst, re->PC, &tctx[re->tid].regs, &actual_target);
      
      // Get prediction
      md_addr_t pred_target;
      int pred_taken = predict_branch_for_thread(re->tid, re->PC, &pred_target, re->inst);
      
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
      update_branch_predictor_for_thread(re->tid, re->PC, actual_taken, actual_target, re->inst);
    }
  } */
  return;
}
static void update_stall_statistics(int tid) {
    if (tid < 0 || tid >= num_hw_threads || !tctx[tid].active) return;
    
    thread_stall_stats[tid].active_cycles++;
}
static double safe_ratio(counter_t numerator, counter_t denominator);
static void init_adaptive_fetch_control() {
  fetch_ctrl.total_fetch_budget = MAX_FETCH_PER_CYCLE;
  fetch_ctrl.last_quota_update = 0;
  
  for (int t = 0; t < MAX_HW_THREAD; t++) {
    fetch_ctrl.thread_fetch_quota[t] = MAX_FETCH_PER_CYCLE / num_hw_threads;
    fetch_ctrl.thread_fetch_priority[t] = 1.0;
    fetch_ctrl.consecutive_ifq_full[t] = 0;
  }
}
static void update_fetch_priorities() {
  if (cycles - fetch_ctrl.last_quota_update < 100) return;  // 100 cycle마다 업데이트
  
  fetch_ctrl.last_quota_update = cycles;
  
  // 각 스레드의 성능 지표를 기반으로 priority 계산
  for (int t = 0; t < num_hw_threads; t++) {
    if (!tctx[t].active) {
      fetch_ctrl.thread_fetch_priority[t] = 0.0;
      continue;
    }
    
    struct thread_pipeline *tp = &tctx[t].pipeline;
    
    // IFQ utilization 계산
    int ifq_occupancy = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
    double ifq_util = (double)ifq_occupancy / IFQ_SIZE;
    
    // IPC 기반 priority
    double ipc = safe_ratio(sim_num_insn_tid[t], cycles);
    
    // Priority 계산: IPC가 높고 IFQ utilization이 낮으면 높은 priority
    fetch_ctrl.thread_fetch_priority[t] = ipc * (1.0 - ifq_util);
    
    // IFQ full penalty
    if (ifq_occupancy >= IFQ_SIZE - 4) {
      fetch_ctrl.consecutive_ifq_full[t]++;
      fetch_ctrl.thread_fetch_priority[t] *= 0.5;  // Penalty
    } else {
      fetch_ctrl.consecutive_ifq_full[t] = 0;
    }
      
    // Starvation prevention
    if (cycles - thread_perf[t].last_fetch_cycle > 200) {
      fetch_ctrl.thread_fetch_priority[t] *= 2.0;  // Boost starved threads
    }
  }
  
  // Priority 정규화
  double total_priority = 0.0;
  for (int t = 0; t < num_hw_threads; t++) {
    if (tctx[t].active) {
      total_priority += fetch_ctrl.thread_fetch_priority[t];
    }
  }
  
  if (total_priority > 0) {
    for (int t = 0; t < num_hw_threads; t++) {
      if (tctx[t].active) {
        fetch_ctrl.thread_fetch_priority[t] /= total_priority;
      }
    }
  }
}

// Dynamic fetch quota 할당
static void allocate_fetch_quotas() {
  int active_threads = 0;
  int total_ifq_free = 0;
  
  for (int t = 0; t < num_hw_threads; t++) {
    if (tctx[t].active) {
      struct thread_pipeline *tp = &tctx[t].pipeline;
      int occupancy = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
      int free_slots = IFQ_SIZE - occupancy;
      total_ifq_free += free_slots;
      active_threads++;
    }
  }
  
  if (active_threads == 0) return;
  
  /* Dynamic budget based on actual free space */
  fetch_ctrl.total_fetch_budget = MIN(MAX_FETCH_PER_CYCLE, total_ifq_free / 2);
  
  /* Ensure minimum fetch for each thread */
  int base_quota = MAX(1, fetch_ctrl.total_fetch_budget / active_threads);
  int remaining_budget = fetch_ctrl.total_fetch_budget;
  
  for (int t = 0; t < num_hw_threads; t++) {
    if (tctx[t].active) {
      /* Base allocation */
      fetch_ctrl.thread_fetch_quota[t] = base_quota;
      remaining_budget -= base_quota;
      
      /* Extra allocation based on IFQ occupancy */
      struct thread_pipeline *tp = &tctx[t].pipeline;
      int occupancy = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
      double occupancy_ratio = (double)occupancy / IFQ_SIZE;
      
      /* Give more quota to threads with low IFQ occupancy */
      if (occupancy_ratio < 0.5 && remaining_budget > 0) {
        int extra = MIN(2, remaining_budget);
        fetch_ctrl.thread_fetch_quota[t] += extra;
        remaining_budget -= extra;
      }
      
      /* Throttle threads with high IFQ occupancy */
      if (occupancy_ratio > 0.75) {
        fetch_ctrl.thread_fetch_quota[t] = MAX(1, fetch_ctrl.thread_fetch_quota[t] / 2);
      }
    }
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
  counter_t *hits_counter = (tlb == dtlb) ? &dtlb_hits_tid[thread_id] : &itlb_hits_tid[thread_id];
  counter_t *misses_counter = (tlb == dtlb) ? &dtlb_misses_tid[thread_id] : &itlb_misses_tid[thread_id];
  
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
      (*hits_counter)++;
      return TLB_HIT;
    }
  }
    
  /* TLB miss - page table walk */
  tlb->misses++;
  /* Update per-thread counters based on TLB type */
  (*misses_counter)++;
  
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
static void init_tlb_system(void) {
  dtlb = tlb_create(TLB_SIZE, 4);
  itlb = tlb_create(TLB_SIZE, 4);
}
static double safe_ratio(counter_t numerator, counter_t denominator) {
    return (denominator > 0) ? (double)numerator / denominator : 0.0;
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

/* ===== Precise Exception Handling ===== */
static void detect_exception(int tid, int rob_idx, exception_type_t type, 
    md_addr_t fault_addr) {
    struct thread_pipeline *tp = &tctx[tid].pipeline;
    if (((exception_tail + 1) % EXCEPTION_BUFFER_SIZE) == exception_head) {
      return; /* Exception buffer full */
    }
    
    exception_entry_t *exc = &tp->exception_buffer[exception_tail];
    
    exc->type = type;
    exc->pc = tp->ROB[rob_idx].PC;
    exc->fault_addr = fault_addr;
    exc->thread_id = tid;
    exc->rob_idx = rob_idx;
    exc->detection_cycle = cycles;
    
    tp->exception_tail = (tp->exception_tail + 1) % EXCEPTION_BUFFER_SIZE;
    exceptions_detected++;
}
static void handle_syscall_exit(int tid){
  if (tid < 0 || tid >= num_hw_threads) return;
  
  printf("Thread %d received exit system call at cycle %lld\n", tid, cycles);
  tctx[tid].active = 0;
  
  flush_thread(tid);
  
  printf("Thread %d final stats: %lld instructions, %lld flushes\n", 
         tid, sim_num_insn_tid[tid], tctx[tid].flush_count);
}
static void handle_precise_exceptions(int tid) {
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  while (tp->exception_head != tp->exception_tail) {
    exception_entry_t *exc = &tp->exception_buffer[exception_head];
    
    /* Check if this is the oldest instruction in the ROB */
    if (exc->rob_idx != tp->rob_head) {
      break; /* Wait for precise exception point */
    }
    
    /* Handle the exception */
    switch (exc->type) {
      case EXCEPTION_PAGE_FAULT:
        printf("Page fault at PC 0x%llx, addr 0x%llx, thread %d\n",
                exc->pc, exc->fault_addr, tid);
        break;
          
      case EXCEPTION_SYSTEM_CALL:
        printf("Thread %d received exit system call at cycle %lld\n", tid, cycles);
        tctx[tid].active = 0;
        flush_thread(tid);
        break;
          
      default:
        printf("Exception type %d at PC 0x%llx, thread %d\n",
                exc->type, exc->pc, tid);
        break;
    }
        
    tp->exception_head = (tp->exception_head + 1) % EXCEPTION_BUFFER_SIZE;
    exceptions_handled++;
    precise_exceptions++;
  }
}

static int predict_branch_for_thread(int tid, md_addr_t pc, md_addr_t *pred_target, md_inst_t inst) {  
  struct thread_branch_predictor *bp = &tctx[tid].bp;
  int opcode = (inst >> 26) & 0x3F;
  bp->bp_lookups++;
  bp_lookups++;
  /* Default fallthrough */
  *pred_target = pc + 4;
  
  /* JMP format */
  if (opcode == 0x1A) {
    int hint = (inst >> 14) & 0x3;
    
    /* BTB lookup for target */
    unsigned btb_idx = (pc >> 2) % BTB_SIZE;
    if (btb[btb_idx].valid && btb[btb_idx].tag == pc) {
      *pred_target = btb[btb_idx].target;
      bp->btb_hits++;
      btb_hits++;
    } else {
      bp->btb_misses++;
      btb_misses++;
      /* Conservative prediction for unknown targets */
      if (hint == 0x2) { /* RET */
        if (ras_predict_for_thread(tid, pc, pred_target)) {
          bp->ras_hits++;
          ras_hits++;
        } else {
          bp->ras_misses++;
          ras_misses++;
        }
      }
    }
    
    if (hint == 0x1) { /* JSR - call */
      ras_push_for_thread(tid, pc + 4);
      bp->function_calls++;
      function_calls++;
    } else if (hint == 0x2) { /* RET */
      bp->function_returns++;
      function_returns++;
    }
    
    return 1; /* Always taken */
  }
  
  /* Branch format */
  else if (opcode >= 0x30 && opcode <= 0x3F) {
    /* Calculate target for all branches */
    int displacement = (int)((inst & 0x1FFFFF) << 2);
    if (displacement & 0x400000) displacement |= 0xFF800000; 
    md_addr_t branch_target = pc + 4 + displacement;
    
    /* BTB lookup */
    unsigned btb_idx = (pc >> 2) % BTB_SIZE;
    if (btb[btb_idx].valid && btb[btb_idx].tag == pc) {
      branch_target = btb[btb_idx].target;
      bp->btb_hits++;
      btb_hits++;
    } else {
      /* Install calculated target in BTB */
      btb[btb_idx].tag = pc;
      btb[btb_idx].target = branch_target;
      btb[btb_idx].valid = 1;
      bp->btb_misses++;
      btb_misses++;
    }
    
    /* Unconditional branches */
    if (opcode == 0x30 || opcode == 0x34) {
      *pred_target = branch_target;
      if (opcode == 0x34) { /* BSR */
        ras_push_for_thread(tid, pc + 4);
        bp->function_calls++;
        function_calls++;
      }
      bp->unconditional_branches++;
      unconditional_branches++;
      return 1;
    }
    
    if (opcode >= 0x38 && opcode <= 0x3F) {
      bp->conditional_branches++;
      conditional_branches++;
      
      /* Tournament prediction */
      int pred_taken = tournament_predict_for_thread(tid, pc, pred_target);
      if (pred_taken) {
        *pred_target = branch_target;  /* Use calculated/BTB target */
      }
      return pred_taken;
    }
  }
  
  return 0; /* Not a branch */
}
static void init_thread_branch_predictor(int tid) {
  struct thread_branch_predictor *bp = &tctx[tid].bp;
  /* Clear BTB completely */
  for (int i = 0; i < BTB_SIZE; i++) {
    btb[i].valid = 0;
    btb[i].tag = 0;
    btb[i].target = 0;
  }
  /* Clear Local predictor */
  for (int i = 0; i < LOCAL_PRED_SIZE; i++) {
      bp->local_predictor[i].state = WEAKLY_NOT_TAKEN;
      bp->local_predictor[i].local_history = 0;
      bp->local_predictor[i].confidence = 0;
      bp->local_predictor[i].last_update = 0;
  }
  
  memset(bp->local_history_table, 0, sizeof(bp->local_history_table));
  
  /* Clear Global history */
  for (int i = 0; i < GLOBAL_PRED_SIZE; i++) {
    bp->global_predictor[i].state = WEAKLY_NOT_TAKEN;
    bp->global_predictor[i].confidence = 0;
  }
  
  for (int i = 0; i < CHOICE_PRED_SIZE; i++) {
    bp->choice_predictor[i].state = WEAKLY_NOT_TAKEN;
    bp->choice_predictor[i].bias = 0;
  }
  
  bp->global_history = 0;
  bp->current_predictor_type = PRED_TOURNAMENT;
  memset(&bp->return_address_stack, 0, sizeof(bp->return_address_stack));
  
  /* Clear statistics */
  bp->bp_lookups = bp->bp_correct = bp->bp_mispred = 0;
  bp->btb_hits = bp->btb_misses = 0;
  bp->local_correct = bp->local_wrong = 0;
  bp->gshare_correct = bp->gshare_wrong = 0;
  bp->tournament_correct = bp->tournament_wrong = 0;
  bp->ras_hits = bp->ras_misses = 0;
  bp->conditional_branches = bp->unconditional_branches = 0;
  bp->function_calls = bp->function_returns = 0;
}
static inline int get_latency(enum md_opcode op)
{
  /* ALU = 1cy, LD/ST = 2cy, MUL = 4cy, DIV = 12cy, FP‑div = 16cy */
  int opc = (op >> 26) & 0x3F;         /* Alpha primary opcode */
  if (opc == 0x10) return 4;      /* MULx */
  if (opc == 0x11) return 12;     /* DIVx */
  if ((opc >> 3) == 0x04) return 2; /* LDx/STx group */
  return 1; /* default ALU */
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

static void update_performance_counters(int tid) {
  if (!tctx[tid].active) return;
  
  performance_counters_t *pc = &tctx[tid].perf_counters;
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  
  pc->cycles_executed = cycles;
  pc->instructions_committed = sim_num_insn_tid[tid];
  
  /* Calculate resource occupancy */
  int ifq_count = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
  int rob_count = (tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE;
  int iq_count = (tp->iq_tail - tp->iq_head + IQ_SIZE) % IQ_SIZE;
  int lsq_count = (tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE;
  
  /* Update running averages */
  double alpha = 0.1;
  pc->avg_ifq_occupancy = (1 - alpha) * pc->avg_ifq_occupancy + alpha * ifq_count;
  pc->avg_rob_occupancy = (1 - alpha) * pc->avg_rob_occupancy + alpha * rob_count;
  pc->avg_iq_occupancy = (1 - alpha) * pc->avg_iq_occupancy + alpha * iq_count;
  pc->avg_lsq_occupancy = (1 - alpha) * pc->avg_lsq_occupancy + alpha * lsq_count;
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
      line->thread_id = -1; /* No thread assigned initially */
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
/* ===== ENHANCED ADDRESS GENERATION AND LSQ ======================= */
static cache_access_result_t cache_access(cache_t *cache, md_addr_t addr, 
                                         int is_write, int thread_id, 
                                         tick_t *ready_time);
  
static void enhanced_lsq_access(struct lsq_entry *lsq, int lsq_idx, int tid);
/* Enhanced LSQ with proper address generation timing */
static void address_generation_stage() {
  for (int tid = 0; tid < num_hw_threads; tid++) {
    if (!tctx[tid].active) continue; /* Skip inactive threads */
    struct thread_pipeline *tp = &tctx[tid].pipeline;
    /* Process Address Generation Unit */
    for (int i = 0; i < AGU_SIZE; i++) {
      agu_entry_t *agu = &tp->AGU[i];
      if (!agu->valid || cycles < agu->ready_cycle) continue;
      
      /* Find corresponding LSQ entry */
      for (int j = tp->lsq_head; j != tp->lsq_tail; j = (j + 1) % LSQ_SIZE) {
        if (tp->LSQ[j].rob_idx == agu->rob_idx && tp->LSQ[j].tid == agu->tid) {
          tp->LSQ[j].vaddr = agu->addr;
          tp->LSQ[j].addr = agu->addr;
          tp->LSQ[j].addr_ready = 1;
          tp->LSQ[j].addr_ready_cycle = cycles;
          
          /* IMMEDIATE memory access for loads */
          if (tp->LSQ[j].is_load) {
            tick_t cache_ready_time;
            cache_access_result_t result = cache_access(dl1_cache, tp->LSQ[j].addr, 
                0, tid, &cache_ready_time);
            if (result == CACHE_HIT) {
              tp->LSQ[j].data_ready = 1;
              tp->LSQ[j].done = cache_ready_time;

              /* IMMEDIATE ROB completion */
              if (tp->LSQ[j].rob_idx >= 0 && tp->LSQ[j].rob_idx < ROB_SIZE) {
                struct rob_entry *rob = &tp->ROB[tp->LSQ[j].rob_idx];
                rob->ready = 1;
                rob->done_cycle = cache_ready_time;
              }
              
              printf("AGU CACHE HIT: Load@0x%llx immediate completion at cycle %lld\n", 
                     tp->LSQ[j].addr, cycles);
            } else {
              tp->LSQ[j].done = cache_ready_time;
              printf("AGU CACHE MISS: Load@0x%llx, ready at cycle %lld\n", 
                     tp->LSQ[j].addr, cache_ready_time);
            }
          }
          break;
        }
      }
      
      /* Clear AGU entry */
      agu->valid = 0;
    }
  }
}
/* ===== MSHR EVENT QUEUE SYSTEM ===================================== */
typedef struct mshr_event {
  tick_t ready_cycle;
  md_addr_t addr;
  int cache_level;  /* 1=L1, 2=L2 */
  int is_writeback;
  int thread_id;
  int lsq_idx; /* LSQ index if applicable */
  int rob_idx; /* ROB index if applicable */
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
  struct thread_pipeline *tp = &tctx[thread_id].pipeline;
  md_addr_t line_addr = addr / cache->config.line_size;

  for (int i = tp->lsq_head; i != tp->lsq_tail; i = (i + 1) % LSQ_SIZE) {
    if (tp->LSQ[i].tid == thread_id && 
      (tp->LSQ[i].addr / cache->config.line_size) == line_addr &&
      tp->LSQ[i].done > cycles) {
      tp->LSQ[i].done = cycles; /* Ready now */
      tp->LSQ[i].data_ready = 1; /* Data is ready */

      if (tp->LSQ[i].rob_idx >= 0 && tp->LSQ[i].rob_idx < ROB_SIZE) {
        struct rob_entry *rob = &tp->ROB[tp->LSQ[i].rob_idx];
        rob->ready = 1;
        rob->done_cycle = cycles;
        
        printf("MSHR COMPLETION: LSQ[%d]->ROB[%d] ready at cycle %lld\n", 
               i, tp->LSQ[i].rob_idx, cycles);
      }
    }
  }
}

static void handle_writeback_completion(cache_t *cache, md_addr_t addr, int thread_id) {
  /* Writeback completed - victim line can be reused */
  
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
    
  md_addr_t line_addr = addr / cache->config.line_size;
  int set_idx = line_addr % cache->num_sets;
  md_addr_t tag = line_addr / cache->num_sets;
  
  cache_set_t *set = &cache->sets[set_idx];
  
  /* Search for hit */
  for (int i = 0; i < cache->config.assoc; i++) {
    cache_line_t *line = &set->lines[i];
    if (line->valid && line->tag == tag) {
      /* Cache hit */
      cache_update_lru(cache, set_idx, line);
      line->thread_id = thread_id;
      line->last_access = cycles;
      if (is_write) line->dirty = 1;
      
      cache->hits++;
      if (cache == il1_cache) il1_hits_tid[thread_id]++;
      if (cache == dl1_cache) dl1_hits_tid[thread_id]++;
      if (cache == dl2_cache) dl2_hits_tid[thread_id]++;
      
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
        cache->hits++;
        if (cache == dl2_cache) dl2_hits_tid[thread_id]++;
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
  if (cache == dl1_cache) dl1_misses_tid[thread_id]++;
  if (cache == dl2_cache) dl2_misses_tid[thread_id]++;

  /* Handle replacement and potential writeback */
  cache_line_t *victim = cache_find_victim(cache, set_idx);
    if (victim->valid && victim->dirty) {
      cache->writebacks++;
      /* Schedule writeback event */
      md_addr_t wb_addr = (victim->tag * cache->num_sets + set_idx) * cache->config.line_size;
      tick_t wb_completion = cycles + cache->config.latency;
      if (cache->next_level) wb_completion += cache->next_level->config.latency;        
      int wb_tid = (victim->thread_id >= 0 && victim->thread_id < MAX_HW_THREAD) ? victim->thread_id : 0;
      event_queue_insert(wb_completion, wb_addr, 
                          (cache == dl1_cache) ? 1 : 2, 1, wb_tid);
  }
    
  /* Install new line */
  victim->tag = tag;
  victim->valid = 1;
  victim->dirty = is_write;
  victim->thread_id = thread_id;
  victim->last_access = cycles;
  cache_update_lru(cache, set_idx, victim);

  /* Calculate miss penalty */
  int miss_latency = cache->config.latency;
  if (cache->next_level) {
    tick_t next_ready;
    cache_access(cache->next_level, addr, is_write, thread_id, &next_ready);
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
  /* L1 Instruction Cache: 64KB, 32B line, 4-way, 1 cycle */
  cache_config_t il1_config = {64*1024, 32, 4, 1, "LRU"};
  il1_cache = cache_create("IL1", il1_config, 8);
  
  /* L1 Data Cache: 64KB, 32B line, 4-way, 1 cycle */
  cache_config_t dl1_config = {64*1024, 32, 4, 1, "LRU"};
  dl1_cache = cache_create("DL1", dl1_config, 16);
  
  /* L2 Unified Cache: 512KB, 64B line, 8-way, 6 cycles */
  cache_config_t dl2_config = {512*1024, 64, 8, 6, "LRU"};
  dl2_cache = cache_create("DL2", dl2_config, 32);
  
  /* Set up hierarchy */
  il1_cache->next_level = dl2_cache;
  dl1_cache->next_level = dl2_cache;

  memset(il1_hits_tid, 0, sizeof(il1_hits_tid));
  memset(il1_misses_tid, 0, sizeof(il1_misses_tid));
  memset(dl1_hits_tid, 0, sizeof(dl1_hits_tid));
  memset(dl1_misses_tid, 0, sizeof(dl1_misses_tid));
  memset(dl2_hits_tid, 0, sizeof(dl2_hits_tid));
  memset(dl2_misses_tid, 0, sizeof(dl2_misses_tid));
  
  printf("Cache hierarchy initialized:\n");
  printf("  IL1: %dKB, %d-way, %dB line, %d MSHR\n", 
         il1_config.size/1024, il1_config.assoc, il1_config.line_size, 8);
  printf("  DL1: %dKB, %d-way, %dB line, %d MSHR\n", 
         dl1_config.size/1024, dl1_config.assoc, dl1_config.line_size, 16);
  printf("  L2:  %dKB, %d-way, %dB line, %d MSHR\n", 
         dl2_config.size/1024, dl2_config.assoc, dl2_config.line_size, 32);
}
/* Function prototype to avoid implicit declaration */
static void cache_invalidate(cache_t *c, md_addr_t addr, int tid) {
  md_addr_t line_addr = addr / c->config.line_size;
  int set = line_addr % c->num_sets;
  md_addr_t tag = line_addr / c->num_sets;
  cache_set_t *s = &c->sets[set];

  for (int w = 0; w < c->config.assoc; w++) {
    cache_line_t *l = &s->lines[w];
    if (l->valid && l->tag == tag && l->thread_id == tid) {
      l->valid = 0; 
      l->dirty = 0; 
      l->thread_id = -1;
      return;
    }
  }
}
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
          cache_invalidate(dl1_cache, addr, t);
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
    default:
      break; /* Unsupported transaction */
  }
  
  entry->last_access = cycles;
}
static void stride_prefetcher_access(int tid, md_addr_t pc, md_addr_t addr) {
  if (!enable_stride_prefetcher) return;  
  struct thread_pipeline *tp = &tctx[tid].pipeline;
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
      if (((tp->prefetch_tail + 1) % PREFETCH_QUEUE_SIZE) != tp->prefetch_head) {
        prefetch_request_t *req = &tp->prefetch_queue[tp->prefetch_tail];
        req->addr = prefetch_addr;
        req->thread_id = tid;
        req->issue_time = cycles;
        req->useful = 0;
        
        tp->prefetch_tail = (tp->prefetch_tail + 1) % PREFETCH_QUEUE_SIZE;
        tp->prefetches_issued++;
        
        /* Trigger cache access */
        tick_t ready_time;
        cache_access(dl1_cache, prefetch_addr, 0, tid, &ready_time);
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
static void enhanced_lsq_access(struct lsq_entry *lsq, int lsq_idx, int tid) {
  if (!lsq || !lsq->addr_ready || tid < 0 || tid >= num_hw_threads) {
    return;
  }
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  /* TLB access */
  md_addr_t physical_addr;
  if (dtlb) {
    tlb_access_result_t tlb_result = tlb_access(dtlb, lsq->vaddr, lsq->tid, &physical_addr);
    if (tlb_result == TLB_MISS) {
      lsq->done = cycles + 25; /* TLB miss penalty */
      return;
    }
   lsq->addr = physical_addr; /* Use physical address */
  }
   

  tick_t cache_ready_time = cycles + 1; /* 기본값 */
  cache_access_result_t result = CACHE_HIT;

  if (lsq->is_load) {
    dcache_accesses_total[tid]++;
    tctx[lsq->tid].dcache_accesses++;
    /* Access stride prefetcher */
    struct rob_entry *re = &tp->ROB[lsq->rob_idx];
    stride_prefetcher_access(tid, re->PC, lsq->addr);
    
    /* Check store forwarding first */
    forward_result_t forward_result = check_store_forwarding(tid, lsq_idx);
    
    if (forward_result == FORWARD_FULL) {
      dcache_forwarding[tid]++;  
      lsq->done = cycles + 1;
      return; /* Forwarding handled completion */
    }
    
    if (forward_result == FORWARD_CONFLICT) {
      lsq->done = cycles + 30; /* Conflict resolution delay */
      return;
    }
    
    /* Handle coherence */
    handle_coherence_transaction(lsq->addr, BUS_READ, lsq->tid);
    
    /* Access cache hierarchy */
    if (dl1_cache) {
      result = cache_access(dl1_cache, lsq->addr, 0, lsq->tid, &cache_ready_time);
      cache_ready_time = MIN(cache_ready_time, cycles + 20);
    } 
    if (result != CACHE_HIT) {
      tctx[lsq->tid].dcache_misses++;
    } else {
      dcache_hits_actual[tid]++;
    }
    lsq->done = cache_ready_time;
    
  } else { /* store */
    dcache_accesses_total[tid]++;
    tctx[lsq->tid].dcache_accesses++;
    /* Handle coherence */
    handle_coherence_transaction(lsq->addr, BUS_WRITE, lsq->tid);
    if (dl1_cache) {
      result = cache_access(dl1_cache, physical_addr, 1, lsq->tid, &cache_ready_time);
      cache_ready_time = MIN(cache_ready_time, cycles + 10);
    }
    if (result != CACHE_HIT) {
      tctx[lsq->tid].dcache_misses++;
    } else {
      dcache_hits_actual[tid]++;
    }
    lsq->done = cache_ready_time;
    lsq->data_ready = 1;
    lsq->data_ready_cycle = cycles;
    struct rob_entry *store_re = &tp->ROB[lsq->rob_idx];
    store_re->done_cycle = cache_ready_time;
    store_re->ready      = 1;
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
          struct thread_pipeline *tp = &tctx[t].pipeline;
          for (int j = tp->lsq_head; j != tp->lsq_tail; j = (j + 1) % LSQ_SIZE) {
            if (tp->LSQ[j].tid == t && 
                (tp->LSQ[j].addr / cache->config.line_size) == line_addr &&
                tp->LSQ[j].done > cycles) {
              tp->LSQ[j].done = cycles + cache->config.latency;
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

static void init_thread_pipeline(int tid) {
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  
  /* 파이프라인 구조체들 초기화 */
  memset(tp->IFQ, 0, sizeof(tp->IFQ));
  memset(tp->IQ, 0, sizeof(tp->IQ));
  memset(tp->LSQ, 0, sizeof(tp->LSQ));
  memset(tp->ROB, 0, sizeof(tp->ROB));
  memset(tp->AGU, 0, sizeof(tp->AGU));
  
  tp->ifq_head = tp->ifq_tail = 0;
  tp->iq_head = tp->iq_tail = 0;
  tp->lsq_head = tp->lsq_tail = 0;
  tp->rob_head = tp->rob_tail = 0;
  
  /* 통계 카운터들 초기화 */
  tp->fetch_ifq_full = 0;
  tp->rename_rob_full = tp->rename_iq_full = tp->rename_lsq_full = 0;
  tp->issue_iq_empty = 0;
  tp->lsq_store_forwards = tp->lsq_load_violations = 0;
  tp->lsq_addr_conflicts = tp->lsq_partial_forwards = 0;
  
  /* 메모리 시스템 초기화 */
  memset(tp->mem_dep_table, 0, sizeof(tp->mem_dep_table));
  memset(tp->slap, 0, sizeof(tp->slap));
  memset(tp->stride_table, 0, sizeof(tp->stride_table));
  memset(tp->prefetch_queue, 0, sizeof(tp->prefetch_queue));
  tp->prefetch_head = tp->prefetch_tail = 0;
  tp->prefetches_issued = tp->prefetches_useful = tp->prefetches_late = 0;
  
  /* 예외 처리 버퍼 초기화 */
  memset(tp->exception_buffer, 0, sizeof(tp->exception_buffer));
  tp->exception_head = tp->exception_tail = 0;
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

static void enter_runahead_mode(int tid, md_addr_t stall_pc) {
    if (!enable_runahead_execution) return;
    
    runahead_state_t *ra = &tctx[tid].runahead;
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
    runahead_state_t *ra = &tctx[tid].runahead;
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
    runahead_state_t *ra = &tctx[tid].runahead;
    if (!ra->runahead_mode) return;
    
    /* Restore checkpoint */
    tctx[tid].regs = ra->checkpoint_regs;
    tctx[tid].pc = ra->normal_pc;
    
    printf("Thread %d exiting runahead mode, generated %d prefetches\n", 
           tid, ra->prefetches_generated);
    
    ra->runahead_mode = 0;
}
/* ===== ENHANCED THREAD SCHEDULING ===================================== */
/* Update thread performance metrics */
static void update_thread_performance(int tid) {
  if (tid < 0 || tid >= num_hw_threads || !tctx[tid].active) return;
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  thread_performance_t *perf = &thread_perf[tid];
  
  /* Calculate recent IPC (sliding window) */
  if (cycles > PERF_WINDOW_SIZE) {
    counter_t recent_instructions = sim_num_insn_tid[tid] - 
        (sim_num_insn_tid[tid] * (cycles - PERF_WINDOW_SIZE) / cycles);
    perf->recent_ipc = (double)recent_instructions / PERF_WINDOW_SIZE;
  } else {
    perf->recent_ipc = safe_ratio(sim_num_insn_tid[tid], cycles);
  }
  
  /* Calculate recent branch accuracy */
  if (tctx[tid].branches_executed > 0) {
    perf->recent_branch_accuracy = 
        (double)(tctx[tid].branches_executed - tctx[tid].branches_mispredicted) / 
        tctx[tid].branches_executed;
  }
  
  /* Calculate recent cache miss rate */
  counter_t total_cache_accesses = tctx[tid].dcache_accesses;
  if (total_cache_accesses > 0) {
    perf->recent_cache_miss_rate = 
        (double)tctx[tid].dcache_misses / total_cache_accesses;
  }
  
  /* Update resource usage */
  perf->ifq_usage = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
  perf->rob_usage = (tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE;
  perf->iq_usage = (tp->iq_tail - tp->iq_head + IQ_SIZE) % IQ_SIZE;
  perf->lsq_usage = (tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE;

  
  /* Count resources used by this thread */
  for (int i = tp->ifq_head; i != tp->ifq_tail; i = (i + 1) % IFQ_SIZE) {
    if (tp->IFQ[i].tid == tid) perf->ifq_usage++;
  }
  
  for (int i = tp->rob_head; i != tp->rob_tail; i = (i + 1) % ROB_SIZE) {
    if (tp->ROB[i].tid == tid) perf->rob_usage++;
  }
  
  for (int i = 0; i < IQ_SIZE; i++) {
    if (tp->IQ[i].ready && tp->IQ[i].tid == tid) perf->iq_usage++;
  }
  
  for (int i = tp->lsq_head; i != tp->lsq_tail; i = (i + 1) % LSQ_SIZE) {
    if (tp->LSQ[i].tid == tid) perf->lsq_usage++;
  }
  
  /* Calculate resource efficiency */
  int total_resources = perf->ifq_usage + perf->rob_usage + 
                        perf->iq_usage + perf->lsq_usage;
  if (total_resources > 0 && sim_num_insn_tid[tid] > 0) {
    perf->resource_efficiency = perf->recent_ipc * 100.0 / total_resources;
  } else {
    perf->resource_efficiency = 0.0;
  }
  
  /* Update starvation tracking */
  perf->cycles_since_last_fetch = cycles - perf->last_fetch_cycle;
  
  /* Update long-term averages */
  double alpha = 0.1; /* Exponential smoothing factor */
  perf->avg_ipc = (1 - alpha) * perf->avg_ipc + alpha * perf->recent_ipc;
  
  /* Update penalties */
  if (perf->flush_penalty_remaining > 0) {
    perf->flush_penalty_remaining--;
  }
  
  if (perf->cache_miss_penalty > 0) {
    perf->cache_miss_penalty--;
  }
  
  if (perf->cycles_since_last_fetch > scheduler.fairness_boost_threshold) {
    perf->resource_starvation_penalty++;
  } else {
    perf->resource_starvation_penalty = MAX(perf->resource_starvation_penalty - 1, 0);
  }
}

/* Calculate thread priority */
static int calculate_thread_priority(int tid) {
  if (tid < 0 || tid >= num_hw_threads || !tctx[tid].active) return 0;
  
  thread_performance_t *perf = &thread_perf[tid];
  int priority = perf->base_priority;
    
  switch (scheduler.current_policy) {
    case SCHED_ROUND_ROBIN:
      /* Simple round-robin, all threads equal priority */
      return 100;
        
    case SCHED_ICOUNT:
      /* ICOUNT policy - favor threads with fewer instructions */
      priority = 200 - MIN(tctx[tid].icount, 100);
      break;
      
    case SCHED_PERFORMANCE_FEEDBACK:
      /* Performance-based priority */
      
      /* IPC bonus/penalty */
      if (perf->recent_ipc > perf->avg_ipc * 1.2) {
        priority += 20; /* High-performing thread */
      } else if (perf->recent_ipc < perf->avg_ipc * 0.8) {
        priority -= 10; /* Low-performing thread */
      }
      
      /* Branch prediction bonus/penalty */
      if (perf->recent_branch_accuracy > 0.9) {
        priority += 15;
      } else if (perf->recent_branch_accuracy < 0.7) {
        priority -= 15;
      }
      
      /* Cache performance bonus/penalty */
      if (perf->recent_cache_miss_rate < 0.1) {
        priority += 10;
      } else if (perf->recent_cache_miss_rate > 0.3) {
        priority -= 20;
      }
      
      /* Resource efficiency bonus */
      if (perf->resource_efficiency > 0.5) {
        priority += 10;
      }
      break;
    
    case SCHED_ADAPTIVE:
      /* Adaptive policy combining multiple factors */
      
      /* Base ICOUNT component */
      priority += (200 - MIN(tctx[tid].icount, 100)) / 4;
      
      /* Performance feedback component */
      priority += (int)(perf->recent_ipc * 50);
      priority += (int)(perf->recent_branch_accuracy * 20);
      priority -= (int)(perf->recent_cache_miss_rate * 30);
      
      /* Resource efficiency component */
      priority += (int)(perf->resource_efficiency * 15);
      break;
  }
    
    /* Apply penalties */
    
    /* Flush penalty - recently flushed threads get lower priority */
    if (cycles - tctx[tid].last_flush_cycle < 50) {
      priority -= 40;
      perf->flush_penalty_remaining = 50;
    }
    
    /* Progressive flush penalty */
    if (tctx[tid].flush_count > 100) {
      priority -= MIN((int)(tctx[tid].flush_count / 50), 30);
    }
    
    /* Cache miss penalty */
    if (perf->recent_cache_miss_rate > 0.5) {
      priority -= 25;
    }
    
    /* Resource starvation penalty */
    priority -= perf->resource_starvation_penalty;
    
    /* Apply fairness boost */
  if (enable_fairness_boost) {
    /* Age-based priority boost */
    if (perf->cycles_since_last_fetch > scheduler.fairness_boost_threshold) {
      int age_boost = MIN(perf->cycles_since_last_fetch / 50, 50);
      priority += age_boost;
      perf->age_boost = age_boost;
    }
    
    /* Starvation prevention */
    if (perf->cycles_since_last_fetch > scheduler.max_starvation_cycles) {
      priority += 100; /* High priority to prevent starvation */
    }
    
    /* Fairness score adjustment */
    if (perf->fairness_score < 0.5) {
      priority += 30; /* Boost unfairly treated threads */
    }
  }
  
  perf->dynamic_priority = priority;
  return MAX(priority, 1); /* Ensure positive priority */
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
  /* Enhanced branch predictor stats */
  stat_reg_counter(sdb, "conditional_branches", "conditional branches", 
                    &conditional_branches, 0, NULL);
  stat_reg_counter(sdb, "unconditional_branches", "unconditional branches", 
                    &unconditional_branches, 0, NULL);
  stat_reg_counter(sdb, "function_calls", "function calls", 
                    &function_calls, 0, NULL);
  stat_reg_counter(sdb, "function_returns", "function returns", 
                    &function_returns, 0, NULL);
  stat_reg_counter(sdb, "ras_hits", "RAS hits", &ras_hits, 0, NULL);
  stat_reg_counter(sdb, "ras_misses", "RAS misses", &ras_misses, 0, NULL);
  
  if (current_predictor_type == PRED_TOURNAMENT) {
    stat_reg_counter(sdb, "local_correct", "local predictor correct", 
                      &local_correct, 0, NULL);
    stat_reg_counter(sdb, "local_wrong", "local predictor wrong", 
                      &local_wrong, 0, NULL);
    stat_reg_counter(sdb, "gshare_correct", "gshare predictor correct", 
                      &gshare_correct, 0, NULL);
    stat_reg_counter(sdb, "gshare_wrong", "gshare predictor wrong", 
                      &gshare_wrong, 0, NULL);
    
    stat_reg_formula(sdb, "local_accuracy", "local predictor accuracy",
                      "local_correct / (local_correct + local_wrong)", NULL);
    stat_reg_formula(sdb, "gshare_accuracy", "gshare predictor accuracy",
                      "gshare_correct / (gshare_correct + gshare_wrong)", NULL);
  }
  
  stat_reg_formula(sdb, "ras_hit_rate", "RAS hit rate",
                    "ras_hits / (ras_hits + ras_misses)", NULL);
  
  /* Scheduling performance stats */
  for (int t = 0; t < num_hw_threads; t++) {
    char name[64], desc[128];
    
    sprintf(name, "dynamic_priority_t%d", t);
    sprintf(desc, "Dynamic priority, thread %d", t);
    stat_reg_double(sdb, name, desc, (counter_t*)&thread_perf[t].dynamic_priority, 0, NULL);
    
    sprintf(name, "resource_efficiency_t%d", t);
    sprintf(desc, "Resource efficiency, thread %d", t);
    stat_reg_double(sdb, name, desc, (counter_t*)&thread_perf[t].resource_efficiency, 0, NULL);
    
    sprintf(name, "fairness_score_t%d", t);
    sprintf(desc, "Fairness score, thread %d", t);
    stat_reg_double(sdb, name, desc, (counter_t*)&thread_perf[t].fairness_score, 0, NULL);
  }
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
  for (int t = 0; t < num_hw_threads; t++) {
        char name[64], desc[128];
        struct thread_pipeline *tp = &tctx[t].pipeline;
        
        sprintf(name, "fetch_ifq_full_t%d", t);
        sprintf(desc, "fetch stalls (IFQ full), thread %d", t);
        stat_reg_counter(sdb, name, desc, &tp->fetch_ifq_full, 0, NULL);
        
        sprintf(name, "rename_rob_full_t%d", t);
        sprintf(desc, "rename stalls (ROB full), thread %d", t);
        stat_reg_counter(sdb, name, desc, &tp->rename_rob_full, 0, NULL);
        
        sprintf(name, "lsq_store_forwards_t%d", t);
        sprintf(desc, "store-to-load forwards, thread %d", t);
        stat_reg_counter(sdb, name, desc, &tp->lsq_store_forwards, 0, NULL);
        
        /* 스레드별 브랜치 예측 통계 */
        sprintf(name, "bp_lookups_t%d", t);
        sprintf(desc, "branch predictor lookups, thread %d", t);
        stat_reg_counter(sdb, name, desc, &tctx[t].bp.bp_lookups, 0, NULL);
        
        sprintf(name, "bp_correct_t%d", t);
        sprintf(desc, "correct branch predictions, thread %d", t);
        stat_reg_counter(sdb, name, desc, &tctx[t].bp.bp_correct, 0, NULL);
        
        sprintf(name, "bp_accuracy_t%d", t);
        sprintf(desc, "branch prediction accuracy, thread %d", t);
        sprintf(name, "bp_correct_t%d / (bp_correct_t%d + bp_mispred_t%d)", t, t, t);
        stat_reg_formula(sdb, name, desc, name, NULL);
        sprintf(name, "bp_mispred_t%d", t);

        sprintf(desc, "branch mispredictions, thread %d", t);
        stat_reg_counter(sdb, name, desc, &tctx[t].bp.bp_mispred, 0, NULL);
        
        sprintf(name, "recent_ipc_t%d", t);
        sprintf(desc, "recent IPC, thread %d", t);
        stat_reg_double(sdb, name, desc, (counter_t*)&thread_perf[t].recent_ipc, 0, NULL);
    }
    
  char formula[256];
  int i;

  /* 빈 문자열로 초기화 */
  formula[0] = '\0';

  /* num_hw_threads 개수만큼 “fetch_ifq_full_t0 + …” 를 붙여 나감 */
  for (i = 0; i < num_hw_threads; i++) {
    char buf[32];
    snprintf(buf, sizeof(buf), "fetch_ifq_full_t%d", i);
    strcat(formula, buf);
    if (i + 1 < num_hw_threads)
        strcat(formula, " + ");
  }

  stat_reg_formula(sdb,
    "total_fetch_stalls",
    "total fetch stalls across all threads",
    "fetch_ifq_full",
    NULL);
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
  mem = mem_create("mem");
  
  /* Initialize thread contexts and pipelines */
  for (int t = 0; t < MAX_HW_THREAD; t++) {
    struct thread_pipeline *tp = &tctx[t].pipeline;
    
    /* Pipeline 구조체 초기화 */
    memset(tp->IFQ, 0, sizeof(tp->IFQ));
    memset(tp->IQ, 0, sizeof(tp->IQ));
    memset(tp->LSQ, 0, sizeof(tp->LSQ));
    memset(tp->ROB, 0, sizeof(tp->ROB));
    memset(tp->AGU, 0, sizeof(tp->AGU));
    
    tp->ifq_head = tp->ifq_tail = 0;
    tp->iq_head = tp->iq_tail = 0;
    tp->lsq_head = tp->lsq_tail = 0;
    tp->rob_head = tp->rob_tail = 0;
    
    /* 통계 카운터 초기화 */
    tp->fetch_ifq_full = 0;
    tp->rename_rob_full = tp->rename_iq_full = tp->rename_lsq_full = 0;
    tp->issue_iq_empty = 0;
    tp->lsq_store_forwards = tp->lsq_load_violations = 0;
    tp->lsq_addr_conflicts = tp->lsq_partial_forwards = 0;
    
    /* 메모리 시스템 초기화 */
    memset(tp->mem_dep_table, 0, sizeof(tp->mem_dep_table));
    memset(tp->slap, 0, sizeof(tp->slap));
    memset(tp->stride_table, 0, sizeof(tp->stride_table));
    memset(tp->prefetch_queue, 0, sizeof(tp->prefetch_queue));
    tp->prefetch_head = tp->prefetch_tail = 0;
    tp->prefetches_issued = tp->prefetches_useful = tp->prefetches_late = 0;
    
    /* 예외 처리 버퍼 초기화 */
    memset(tp->exception_buffer, 0, sizeof(tp->exception_buffer));
    tp->exception_head = tp->exception_tail = 0;
    
    /* Thread context 초기화 */
    tctx[t].active = 0;
    tctx[t].speculation_depth = 0;
    tctx[t].last_flush_cycle = 0;
    tctx[t].flush_count = 0;
    tctx[t].branches_executed = 0;
    tctx[t].branches_mispredicted = 0;
    tctx[t].icache_accesses = 0;
    tctx[t].icache_misses = 0;
    tctx[t].dcache_accesses = 0;
    tctx[t].dcache_misses = 0;

    /* Branch predictor 초기화 */
    struct thread_branch_predictor *bp = &tctx[t].bp;
    
    /* Local predictor 초기화 */
    for (int i = 0; i < LOCAL_PRED_SIZE; i++) {
        bp->local_predictor[i].state = WEAKLY_NOT_TAKEN;
        bp->local_predictor[i].local_history = 0;
        bp->local_predictor[i].confidence = 0;
        bp->local_predictor[i].last_update = 0;
    }
    
    memset(bp->local_history_table, 0, sizeof(bp->local_history_table));
    
    /* Global predictor 초기화 */
    for (int i = 0; i < GLOBAL_PRED_SIZE; i++) {
        bp->global_predictor[i].state = WEAKLY_NOT_TAKEN;
        bp->global_predictor[i].confidence = 0;
    }
    
    /* Choice predictor 초기화 */
    for (int i = 0; i < CHOICE_PRED_SIZE; i++) {
        bp->choice_predictor[i].state = WEAKLY_NOT_TAKEN;
        bp->choice_predictor[i].bias = 0;
    }
    
    bp->global_history = 0;
    bp->current_predictor_type = PRED_TOURNAMENT;
    memset(&bp->return_address_stack, 0, sizeof(bp->return_address_stack));
    
    /* 브랜치 예측 통계 초기화 */
    bp->bp_lookups = bp->bp_correct = bp->bp_mispred = 0;
    bp->btb_hits = bp->btb_misses = 0;
    bp->local_correct = bp->local_wrong = 0;
    bp->gshare_correct = bp->gshare_wrong = 0;
    bp->tournament_correct = bp->tournament_wrong = 0;
    bp->ras_hits = bp->ras_misses = 0;
    bp->conditional_branches = bp->unconditional_branches = 0;
    bp->function_calls = bp->function_returns = 0;
    
    /* Performance tracking 초기화 */
    thread_perf[t].recent_ipc = 0.0;
    thread_perf[t].base_priority = 100;
    thread_perf[t].flush_penalty_remaining = 0;
    thread_perf[t].cache_miss_penalty = 0;
    thread_perf[t].resource_starvation_penalty = 0;
    thread_perf[t].cycles_since_last_fetch = 0;
    thread_perf[t].total_progress = 0;
    thread_perf[t].total_cycles_active = 0;
    thread_perf[t].last_fetch_cycle = 0;
    thread_perf[t].total_fetch_cycles = 0;
    thread_perf[t].fairness_score = 1.0;
    
    /* Performance counters 초기화 */
    memset(&tctx[t].perf_counters, 0, sizeof(performance_counters_t));
    
    /* Runahead state 초기화 */
    memset(&tctx[t].runahead, 0, sizeof(runahead_state_t));
    
    /* Resource partitions 초기화 */
    resource_partitions[t].fetch_slots = IFQ_SIZE / num_hw_threads;
    resource_partitions[t].rename_slots = ROB_SIZE / num_hw_threads;
    resource_partitions[t].issue_slots = IQ_SIZE / num_hw_threads;
    resource_partitions[t].lsq_slots = LSQ_SIZE / num_hw_threads;

    
  }
  for (int i = 0; i < MD_TOTAL_REGS; i++) {
    prf_ready[i] = 1;
  }
  /* PRF free list 초기화 */
  int pos = 0;
  for (int i = MD_TOTAL_REGS; i < PRF_NUM; ++i) {
    free_list[pos++] = i;
    prf_ready[i] = 0;
  }
  for (int i = 0; i < MD_TOTAL_REGS; ++i) prf_ready[i] = 1;
  free_head = 0;
  free_tail = pos;
 
  /* Cache hierarchy 초기화 */
  init_cache_hierarchy();
  
  /* TLB system 초기화 */
  init_tlb_system();
  
  /* Event queue 초기화 */
  event_queue_init();
  
  /* Scheduler 초기화 */
  scheduler.current_policy = SCHED_ADAPTIVE;
  scheduler.fetch_round_robin_ptr = 0;
  scheduler.adaptation_interval = ADAPTATION_INTERVAL;
  scheduler.cycles_since_adaptation = 0;
  scheduler.max_starvation_cycles = starvation_threshold;
  scheduler.fairness_boost_threshold = starvation_threshold / 2;
  
  /* Coherence table 초기화 */
  memset(coherence_table, 0, sizeof(coherence_table));
  
  /* Global branch predictor structures 초기화 */
  memset(btb, 0, sizeof(btb));
  for (int i = 0; i < BTB_SIZE; i++) {
    btb[i].valid = 0;
    btb[i].tag = 0;
    btb[i].target = 0;
  }
  
  /* Memory dependence predictor 초기화 */
  memset(mem_dep_table, 0, sizeof(mem_dep_table));
  
  /* Global statistics 초기화 */
  cycles = 0;
  sim_num_insn = 0;
  memset(sim_num_insn_tid, 0, sizeof(sim_num_insn_tid));
  
  /* LSQ statistics 초기화 */
  lsq_store_forwards = 0;
  lsq_load_violations = 0;
  lsq_addr_conflicts = 0;
  lsq_partial_forwards = 0;
  
  /* Branch predictor statistics 초기화 */
  bp_lookups = 0;
  bp_correct = 0;
  bp_mispred = 0;
  btb_hits = 0;
  btb_misses = 0;
  local_correct = 0;
  local_wrong = 0;
  gshare_correct = 0;
  gshare_wrong = 0;
  tournament_correct = 0;
  tournament_wrong = 0;
  ras_hits = 0;
  ras_misses = 0;
  conditional_branches = 0;
  unconditional_branches = 0;
  function_calls = 0;
  function_returns = 0;
  
  /* Pipeline stall statistics 초기화 */
  fetch_ifq_full = 0;
  rename_rob_full = 0;
  rename_iq_full = 0;
  rename_lsq_full = 0;
  issue_iq_empty = 0;
  
  /* Cache coherence statistics 초기화 */
  bus_transactions = 0;
  coherence_misses = 0;
  invalidations = 0;
  
  /* Prefetching statistics 초기화 */
  prefetches_issued = 0;
  prefetches_useful = 0;
  prefetches_late = 0;
  
  /* Exception statistics 초기화 */
  exceptions_detected = 0;
  exceptions_handled = 0;
  precise_exceptions = 0;
  
  /* Per-thread cache/TLB statistics 초기화 */
  memset(il1_hits_tid, 0, sizeof(il1_hits_tid));
  memset(il1_misses_tid, 0, sizeof(il1_misses_tid));
  memset(dl1_hits_tid, 0, sizeof(dl1_hits_tid));
  memset(dl1_misses_tid, 0, sizeof(dl1_misses_tid));
  memset(dl2_hits_tid, 0, sizeof(dl2_hits_tid));
  memset(dl2_misses_tid, 0, sizeof(dl2_misses_tid));
  memset(itlb_hits_tid, 0, sizeof(itlb_hits_tid));
  memset(itlb_misses_tid, 0, sizeof(itlb_misses_tid));
  memset(dtlb_hits_tid, 0, sizeof(dtlb_hits_tid));
  memset(dtlb_misses_tid, 0, sizeof(dtlb_misses_tid));
  init_adaptive_fetch_control();
  printf("SMT simulator initialized with %d hardware threads\n", num_hw_threads);
  printf("Enhanced features: dynamic partitioning=%s, stride prefetch=%s, memory dep pred=%s\n",
         enable_dynamic_partitioning ? "ON" : "OFF",
         enable_stride_prefetcher ? "ON" : "OFF", 
         memory_dependency_prediction ? "ON" : "OFF");
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
    tctx[tid].pipeline.rob_head = tctx[tid].pipeline.rob_tail = 0;
    tctx[tid].seq = tctx[tid].icount = 0;
  }
}
void sim_uninit(void) {}
void sim_aux_stats(FILE *stream) {}
void sim_aux_config(FILE *stream) {
  fprintf(stream, "threads %d\n", num_hw_threads);
}
static int thread_empty(int tid) {
  if (tid < 0 || tid >= num_hw_threads) return 1;
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  int entries_found = 0;
  
  /* IFQ check */
  int ifq_count = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
  entries_found += ifq_count;
  
  /* IQ check */
  int iq_count = (tp->iq_tail - tp->iq_head + IQ_SIZE) % IQ_SIZE;
  entries_found += iq_count;
  
  /* ROB check */
  int rob_count = (tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE;
  entries_found += rob_count;
  
  /* LSQ check */
  int lsq_count = (tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE;
  entries_found += lsq_count;
  
  return (entries_found == 0);
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
      counter_t active_cycles = MAX(1, thread_stall_stats[t].active_cycles);
            printf("  IPC: %.3f (based on %lld active cycles)\n", 
                   safe_ratio(sim_num_insn_tid[t], active_cycles), active_cycles);
            
      if (tctx[t].branches_executed > 0) {
        double bp_accuracy = safe_ratio((tctx[t].branches_executed - tctx[t].branches_mispredicted), tctx[t].branches_executed) * 100.0;
        if (bp_accuracy < 0) bp_accuracy = 0;
        if (bp_accuracy > 100) bp_accuracy = 100;
        printf("  Branch Accuracy: %.1f%%\n", bp_accuracy);
      } else {
        printf("  Branch Accuracy: N/A\n");
      }
      
      printf("  Flush Count: %lld\n", tctx[t].flush_count);
      
      /* 캐시 통계 */
      double actual_hit_rate = safe_ratio(dcache_hits_actual[t], dcache_accesses_total[t]) * 100.0;
            double forwarding_rate = safe_ratio(dcache_forwarding[t], dcache_accesses_total[t]) * 100.0;
            double total_hit_rate = safe_ratio(dcache_hits_actual[t] + dcache_forwarding[t], 
                                             dcache_accesses_total[t]) * 100.0;
            
            printf("  D-Cache Actual Hit Rate: %.1f%% (%lld/%lld)\n", 
                   actual_hit_rate, dcache_hits_actual[t], dcache_accesses_total[t]);
            printf("  D-Cache Forwarding Rate: %.1f%% (%lld/%lld)\n", 
                   forwarding_rate, dcache_forwarding[t], dcache_accesses_total[t]);
            printf("  D-Cache Total Hit Rate: %.1f%%\n", total_hit_rate);
            
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
  printf("Prediction Accuracy: %.1f%%\n", safe_ratio(bp_correct, bp_correct + bp_mispred) * 100.0);
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
/* Performance Analysis Function */
static void analyze_branch_predictor_performance() {
    printf("\n=== Enhanced Branch Predictor Analysis ===\n");
    
    double overall_accuracy = safe_ratio(bp_correct, bp_correct + bp_mispred) * 100.0;
    printf("Overall Accuracy: %.2f%% (%lld/%lld)\n", 
           overall_accuracy, bp_correct, bp_correct + bp_mispred);
    
    if (conditional_branches > 0) {
        printf("Conditional Branches: %lld\n", conditional_branches);
        if (current_predictor_type == PRED_TOURNAMENT) {
            double local_acc = safe_ratio(local_correct, local_correct + local_wrong) * 100.0;
            double gshare_acc = safe_ratio(gshare_correct, gshare_correct + gshare_wrong) * 100.0;
            printf("  Local Predictor Accuracy: %.2f%%\n", local_acc);
            printf("  Gshare Predictor Accuracy: %.2f%%\n", gshare_acc);
        }
    }
    
    printf("Unconditional Branches: %lld\n", unconditional_branches);
    printf("Function Calls: %lld\n", function_calls);
    printf("Function Returns: %lld\n", function_returns);
    
    if (function_returns > 0) {
        double ras_accuracy = safe_ratio(ras_hits, ras_hits + ras_misses) * 100.0;
        printf("RAS Accuracy: %.2f%% (%lld/%lld)\n", 
               ras_accuracy, ras_hits, ras_hits + ras_misses);
    }
    
    double btb_hit_rate = safe_ratio(btb_hits, btb_hits + btb_misses) * 100.0;
    printf("BTB Hit Rate: %.2f%% (%lld/%lld)\n", 
           btb_hit_rate, btb_hits, btb_hits + btb_misses);
}
static void update_resource_utilization(int tid);
static bool rename_stalling[MAX_HW_THREAD] = {FALSE};
static bool commit_blocked[MAX_HW_THREAD] = {false};
/* =====  M A I N   L O O P ========================================== */
void sim_main(void) {
  counter_t stall_cycles = 0;
  counter_t max_stall_cycles = 10000;
  counter_t last_committed_insn = 0;
  counter_t last_cycle_check = 0;
  while (1) {
    /* ---- fast‑forward window ---- */
    if (fastfwd && warmup < fastfwd){
      /* skip timing stats but still drive pipeline */
      fetch_stage();
      rename_stage();
      warmup++;
      continue;
    } else {    /* ---- timed region ---- */
      for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
          update_resource_utilization(t);
          update_stall_statistics(t);
        }
      }
      /* Process MSHR events first */
      if (mshr_events) process_mshr_events();
      /* Handle precise exceptions & Update performance counters */
      if (cycles % 100 == 0) {
        for (int t = 0; t < num_hw_threads; t++) {
          if (tctx[t].active) {
            handle_precise_exceptions(t);
            update_performance_counters(t);
          }
        }
      }
      /* Update resource partitions every 1000 cycles */
      if (cycles % 1000 == 0) {
        /* Enhanced scheduling adaptations */
        adapt_scheduling_policy();
        dynamic_resource_allocation();
      }

      /* Pipeline stages */
      fetch_stage();
      rename_stage();
      issue_stage();
      address_generation_stage();
      writeback_stage();
      if (mshr_events) process_mshr_events();
      for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
          check_load_store_violations(t);
        }
      }
      check_branch_misprediction();
      commit_stage();
      if (mshr_events) process_mshr_events();

      /* Runahead execution for stalled threads */
      if (enable_runahead_execution) {
        for (int t = 0; t < num_hw_threads; t++) {
          if (tctx[t].active) {
            /* Check if thread is stalled on memory */ 
            struct thread_pipeline *tp = &tctx[t].pipeline;
            int stalled_on_memory = 0;
            for (int i = tp->lsq_head; i != tp->lsq_tail; i = (i + 1) % LSQ_SIZE) {
              if (tp->LSQ[i].tid == t && tp->LSQ[i].is_load && 
                !tp->LSQ[i].addr_ready && cycles - tp->LSQ[i].addr_ready_cycle > 10) {
                stalled_on_memory = 1;
                break;
              }
            }
            
            if (stalled_on_memory && !tctx[t].runahead.runahead_mode) {
              enter_runahead_mode(t, tctx[t].pc);
            } else if (tctx[t].runahead.runahead_mode) {
              execute_runahead_instruction(t);
            }
          }
        }
      } 

      cycles++;
    }

    int any_active = 0;
    int any_pending = 0;
    int total_lsq_full = 0;
    int total_rob_full = 0;

    for (int t = 0; t < num_hw_threads; ++t) {
      if (tctx[t].active) {
        any_active = 1;
        struct thread_pipeline *tp = &tctx[t].pipeline;
        
        int lsq_occ = (tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE;
        int rob_occ = (tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE;
        
        if (lsq_occ > LSQ_SIZE * 0.9) total_lsq_full++;
        if (rob_occ > ROB_SIZE * 0.9) total_rob_full++;
      }
      if (!thread_empty(t)) {
        any_pending = 1;
      }
    }
    
    /* ═══ PROGRESS MONITORING ═══ */
    
    if (cycles % 1000 == 0) {
      printf("Cycle %lld: Instructions committed: %lld, IPC: %.3f\n", 
             cycles, sim_num_insn, (double)sim_num_insn / cycles);
      
      for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
          printf("  Thread %d: %lld insns, rename_stall=%s, commit_block=%s\n", 
                 t, sim_num_insn_tid[t], 
                 rename_stalling[t] ? "YES" : "NO",
                 commit_blocked[t] ? "YES" : "NO");
        }
      }
    }
    
    /* ═══ TERMINATION CONDITIONS ═══ */
    
    /* Update resource utilization */
    for (int t = 0; t < num_hw_threads; t++) {
      if (tctx[t].active) {
        update_resource_utilization(t);
        update_stall_statistics(t);
      }
    }
    
    /* Check for completion */    
    for (int t = 0; t < num_hw_threads; t++) {
      if (tctx[t].active) any_active = 1;
      if (!thread_empty(t)) any_pending = 1;
    }
    
    if (!any_active && !any_pending) {
      printf("Simulation completed at cycle %lld\n", cycles);
      break;
    }
    
    if (sim_max_insn && sim_num_insn >= sim_max_insn) {
      printf("Instruction limit reached at cycle %lld\n", cycles);
      break;
    }
    
    /* Enhanced deadlock detection */
    if (cycles - last_cycle_check >= 50) {
      if (sim_num_insn > last_committed_insn) {
        last_committed_insn = sim_num_insn;
        stall_cycles = 0;
      } else {
        stall_cycles += (cycles - last_cycle_check);
      }
      last_cycle_check = cycles;
    }
    
    if (stall_cycles > max_stall_cycles) {
      printf("DEADLOCK DETECTED at cycle %lld (no progress for %lld cycles)\n", 
             cycles, max_stall_cycles);
      
      /* Diagnostic information */
      for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
          struct thread_pipeline *tp = &tctx[t].pipeline;
          printf("  Thread %d: ROB=%d/%d, IQ=%d/%d, LSQ=%d/%d, rename_stall=%s\n",
                 t, 
                 (tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE, ROB_SIZE,
                 (tp->iq_tail - tp->iq_head + IQ_SIZE) % IQ_SIZE, IQ_SIZE,
                 (tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE, LSQ_SIZE,
                 rename_stalling[t] ? "YES" : "NO");
        }
      }
      
      /* Emergency recovery */
      for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
          enhanced_flush_thread(t);
        }
      }
      
      stall_cycles = 0;
      max_stall_cycles *= 2;
      
      if (max_stall_cycles > 80000) {
        printf("Multiple deadlocks detected, terminating simulation\n");
        break;
      }
    }
    
    /* Periodic optimizations */
    if (cycles % 1000 == 0) {
      adapt_scheduling_policy();
      dynamic_resource_allocation();
    }
  }
  
  /* Final performance analysis */
  print_performance_analysis();
  analyze_branch_predictor_performance();
  analyze_scheduling_performance();
  report_fetch_stats();
}
static inline bool can_accept_fetch(int tid)
{
    struct thread_pipeline *tp = &tctx[tid].pipeline;
    int rob_free = ROB_SIZE - ((tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE);
    int iq_free  = IQ_SIZE  - ((tp->iq_tail - tp->iq_head + IQ_SIZE) % IQ_SIZE);
    int lsq_free = LSQ_SIZE - ((tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE);
    return (rob_free >= 4 && iq_free >= 2 && lsq_free >= 2);
}
/* =====  S T A G E   S T U B S ====================================== */
static long long fetch_stall_rename[MAX_HW_THREAD];
static long long fetch_stall_rob_full[MAX_HW_THREAD];
static long long fetch_stall_iq_full[MAX_HW_THREAD];
static long long fetch_stall_ifq_full[MAX_HW_THREAD];
static void fetch_stage() {
  bool in_timing = (fastfwd == 0) || (warmup >= fastfwd);
  /* Simple round-robin with back-pressure awareness */
  static int fetch_rr_ptr = 0; /* round‑robin pointer */

  int total_fetched = 0;
  
  /* Multiple fetch rounds to fill budget */
  for (int round = 0; round < num_hw_threads && total_fetched < sim_outorder_width; round++) {
    int tid = (fetch_rr_ptr + round) % num_hw_threads;
    if (!tctx[tid].active) continue;
    
    struct thread_pipeline *tp = &tctx[tid].pipeline;
      
    /* 1. Rename OR Commit blocked? - STOP FETCH IMMEDIATELY */
    if (rename_stalling[tid] || commit_blocked[tid]) {
      fetch_stall_rename[tid]++;
      printf("FETCH BLOCKED: Thread %d due to rename stall at cycle %lld\n", tid, cycles);
      continue;
    }
    /* 2. ROB full check - EXPLICIT FULL CONDITION */
    if ( (tp->rob_tail + 1) % ROB_SIZE == tp->rob_head ) {
      fetch_stall_rob_full[tid]++;
      continue;
    }
    /* 3. IQ full check - EXPLICIT FULL CONDITION */
    if ( (tp->iq_tail  + 1) % IQ_SIZE  == tp->iq_head ) {
      fetch_stall_iq_full[tid]++;
      continue;
    }
    /* 4. IFQ full check - EXPLICIT FULL CONDITION */
    int ifq_next = (tp->ifq_tail + 1) % IFQ_SIZE;
    if (ifq_next == tp->ifq_head) {
      fetch_stall_ifq_full[tid]++;
      continue;
    }

    /* 5. Speculation depth limit */
    if (tctx[tid].speculation_depth > 16) {
      continue; 
    }
    
    /* ═══ FETCH INSTRUCTION ═══ */
    md_inst_t inst;
    tick_t ready_time;
    
    /* TLB access for instruction fetch */
    md_addr_t physical_pc = tctx[tid].pc;
    if (itlb) {
      tlb_access_result_t tlb_result = tlb_access(itlb, tctx[tid].pc, tid, &physical_pc);
      if (tlb_result == TLB_PAGE_FAULT) continue;
    }
    
    /* I-Cache access */
    if (il1_cache) {
      cache_access_result_t result = cache_access(il1_cache, physical_pc, 0, tid, &ready_time);
      tctx[tid].icache_accesses++;
      
      if (result != CACHE_HIT) {
        tctx[tid].icache_misses++;
      }
    }
      
    /* Memory access */
    if (mem_access(mem, Read, tctx[tid].pc, &inst, sizeof(md_inst_t)) != md_fault_none) {
      printf("Warning: Memory access fault at PC 0x%llx for thread %d\n", tctx[tid].pc, tid);
      tctx[tid].active = 0;
      continue;
    }
      
    /* Branch Prediction */
    enum md_opcode op;
    MD_SET_OPCODE(op, inst);
    
    md_addr_t next_pc = tctx[tid].pc + sizeof(md_inst_t);
    
    if (MD_OP_FLAGS(op) & F_CTRL) {
      md_addr_t pred_target = next_pc;  /* Default fallthrough */
      int pred_taken = predict_branch_for_thread(tid, tctx[tid].pc, &pred_target, inst);
      
      /* Update BTB with prediction */
      unsigned btb_idx = (tctx[tid].pc >> 2) % BTB_SIZE;
      if (!btb[btb_idx].valid || btb[btb_idx].tag != tctx[tid].pc) {
        btb[btb_idx].tag = tctx[tid].pc;
        btb[btb_idx].target = pred_target;
        btb[btb_idx].valid = 1;
      }
      
      if (in_timing) {
        tctx[tid].branches_executed++;
        tctx[tid].bp.bp_lookups++;
        bp_lookups++;
      }
      
      /* Update PC based on prediction */
      next_pc = pred_taken ? pred_target : next_pc;
      
      /* Only increment speculation depth for taken branches */
      if (pred_taken) {
        tctx[tid].speculation_depth++;
      }
    }
      
    /* Store instruction in IFQ */
    tp->IFQ[tp->ifq_tail] = (struct ifq_entry){inst, tctx[tid].pc, tid};
    tp->ifq_tail = (tp->ifq_tail + 1) % IFQ_SIZE;
    
    /* Update PC for next fetch */
    tctx[tid].pc = next_pc;
    tctx[tid].speculative_pc = next_pc;
    
    /* Update counters */
    tctx[tid].icount++;
    total_fetched++;
    if (in_timing) {
      thread_perf[tid].last_fetch_cycle = cycles;
      thread_perf[tid].cycles_since_last_fetch = 0;
    }
    
    printf("FETCH SUCCESS: Thread %d, PC=0x%llx->0x%llx, inst=0x%x\n", 
           tid, tctx[tid].pc - 4, next_pc, inst);
  }
  
  fetch_rr_ptr = (fetch_rr_ptr + 1) % num_hw_threads;
}

static inline int alpha_dest_reg(md_inst_t inst)
{
  return inst & 0x1F;   /* bits [4:0] */
}
static inline int alpha_src1(md_inst_t inst){ return (inst >> 21) & 0x1F; }
static inline int alpha_src2(md_inst_t inst){ return (inst >> 16) & 0x1F; }
static void rename_stage()  {
  int renamed = 0;
   
  for (int tid = 0; tid < num_hw_threads && renamed < sim_outorder_width; tid++) {
    if (commit_blocked[tid]) {
      rename_stalling[tid] = true;
      continue;
    }
    if (!tctx[tid].active) { 
      rename_stalling[tid] = false;
      continue;
    }
    struct thread_pipeline *tp = &tctx[tid].pipeline;
    /* 1. IFQ empty check */
    if (tp->ifq_head == tp->ifq_tail) {
      rename_stalling[tid] = false;
      continue;
    }
    
    /* 2. ROB full check */
    int rob_next = (tp->rob_tail + 1) % ROB_SIZE;
    if (rob_next == tp->rob_head) {
      tp->rename_rob_full++;
      rename_stalling[tid] = true;
      continue;
    }
    
    /* 3. IQ full check */
    int next_tail = (tp->iq_tail + 1) % IQ_SIZE;
    if (next_tail == tp->iq_head) {
      tp->rename_iq_full++;
      rename_stalling[tid] = true;
      continue;
    }

    /* 4. PEEK IFQ to check LSQ requirement (before dequeue) */
    struct ifq_entry peek_fe = tp->IFQ[tp->ifq_head];
    enum md_opcode peek_op;
    MD_SET_OPCODE(peek_op, peek_fe.inst);
    bool needs_lsq = is_load(peek_op) || is_store(peek_op);
    
    if (needs_lsq) {
      int lsq_next = (tp->lsq_tail + 1) % LSQ_SIZE;
      if (lsq_next == tp->lsq_head) {
        tp->rename_lsq_full++;
        rename_stalling[tid] = true;
        continue;
      }
    }
    /* 5. Dequeue IFQ */
    rename_stalling[tid] = false;
    struct ifq_entry fe = tp->IFQ[tp->ifq_head];
    tp->ifq_head = (tp->ifq_head + 1) % IFQ_SIZE;

    /* 6. Allocate ROB entry */
    int rid = tp->rob_tail;
    struct rob_entry *re = &tp->ROB[rid];
    memset(re, 0, sizeof(*re));
    re->tid = tid;
    re->inst = fe.inst;
    re->PC = fe.PC;
    re->seq = ++tctx[tid].seq;
    
    enum md_opcode op;
    MD_SET_OPCODE(op, fe.inst);

    /* 7. Register renaming */
    int a1 = alpha_src1(fe.inst);
    int a2 = alpha_src2(fe.inst);
    int dest = alpha_dest_reg(fe.inst);
    
    re->src1 = (a1 == 31) ? -1 : tctx[tid].rename_map[a1];
    re->src2 = (a2 == 31) ? -1 : tctx[tid].rename_map[a2];
    re->is_load = is_load(op);
    re->is_store = is_store(op);

    /* 8. Physical register allocation */
    int newp = -1;
    if (dest != 31 && !is_store(op)) {
      newp = prf_alloc();
      if (newp < 0) {
        tp->ifq_head = (tp->ifq_head - 1 + IFQ_SIZE) % IFQ_SIZE;
        continue;
      }
      re->new_phys = newp;
      re->old_phys = tctx[tid].rename_map[dest];
      tctx[tid].rename_map[dest] = newp;
      prf_ready[newp] = 0;
    }

    /* 9. Allocate IQ entry */
    bool iq_allocated = false;
    int allocated_iq_idx = -1;
    for (int i = 0; i < IQ_SIZE; i++) {
      struct iq_entry *q = &tp->IQ[i];
      if (!q->valid) {
        memset(q, 0, sizeof(struct iq_entry));
        q->valid = 1;
        q->rob_idx = rid;
        q->tid = tid;
        q->inst = fe.inst;
        q->src1 = re->src1;
        q->src2 = re->src2;
        q->dst = newp;
        q->is_load = is_load(op);
        q->is_store = is_store(op);
        q->ready = ((q->src1 < 0 || prf_ready[q->src1]) &&
            (q->src2 < 0 || prf_ready[q->src2]));
        q->issued = 0;
        tp->iq_tail = next_tail;
        iq_allocated = true;
        allocated_iq_idx = i;
        break;
      }
    }
    
    if (!iq_allocated) {
      /* Rollback allocations */
      if (newp != -1) {
        prf_free(newp);
        tctx[tid].rename_map[dest] = re->old_phys;
      }
      tp->ifq_head = (tp->ifq_head - 1 + IFQ_SIZE) % IFQ_SIZE;
      tp->rename_iq_full++;
      continue;
    }
    /* 10. Allocate LSQ entry if needed */
    if (needs_lsq) {
      struct lsq_entry *lsq = &tp->LSQ[tp->lsq_tail];
      memset(lsq, 0, sizeof(*lsq));
      lsq->rob_idx = rid;
      lsq->tid = tid;
      lsq->is_load = is_load(op);
      lsq->is_store = is_store(op);
      lsq->size = 4;
      lsq->valid = 1;
      tp->lsq_tail = (tp->lsq_tail + 1) % LSQ_SIZE;
    }
    /* 11. Advance ROB tail */
    tp->rob_tail = (tp->rob_tail + 1) % ROB_SIZE;
    renamed++;

    printf("RENAME SUCCESS: Thread %d, PC=0x%llx, ROB[%d], IQ[%d]\n", 
           tid, fe.PC, rid, allocated_iq_idx);
  }
}
static void issue_stage()   { 
  int issued = 0;
  /* Round-robin through threads for better fairness */
  static int issue_rr_ptr = 0;
  int max_per_thread = MAX(1, sim_outorder_width / num_hw_threads);
  for (int round = 0; round < num_hw_threads && issued < sim_outorder_width; round++) {
    int tid = (issue_rr_ptr + round) % num_hw_threads;
    if (!tctx[tid].active) continue; 

    struct thread_pipeline *tp = &tctx[tid].pipeline;
    int thread_issued = 0;
    while (tp->iq_head != tp->iq_tail &&
           issued  < sim_outorder_width &&
           thread_issued < max_per_thread){
      struct iq_entry *q = &tp->IQ[tp->iq_head];
      if (!q->valid) {
        tp->iq_head = (tp->iq_head + 1) % IQ_SIZE;
        continue;
      }

      if (q->issued) {
        memset(q,0,sizeof(*q));
        tp->iq_head = (tp->iq_head + 1) % IQ_SIZE;
        continue;
      }
      /* Check source operand readiness */
      if ((q->src1>=0 && !prf_ready[q->src1]) ||
      (q->src2>=0 && !prf_ready[q->src2])) {
        break;
      }

      enum md_opcode op;
      MD_SET_OPCODE(op, q->inst);   

      /* Schedule address generation */
      if (q->is_load || q->is_store) {
        /* Find corresponding LSQ entry */
        int lsq_idx = -1;
        for (int i = tp->lsq_head; i != tp->lsq_tail; i = (i + 1) % LSQ_SIZE) {
          if (tp->LSQ[i].rob_idx == q->rob_idx && tp->LSQ[i].tid == tid) {
            lsq_idx = i;
            break;
          }
        }
        
        if (lsq_idx == -1) {
          continue;
        }

        struct lsq_entry *lsq = &tp->LSQ[lsq_idx];
        if (!lsq->addr_ready) {
            /* Calculate address */
          int base_reg = (q->inst >> 16) & 0x1F;
          short displacement = (short)(q->inst & 0xFFFF);
          
          md_addr_t base_value = 0;
          if (base_reg != 31 && base_reg < MD_TOTAL_REGS) {
            base_value = tctx[tid].regs.regs_R[base_reg];
          }
          md_addr_t vaddr = base_value + displacement;
          
          lsq->vaddr = vaddr;
          lsq->addr = vaddr; /* Default to virtual address */
          lsq->addr_ready = 1;
          lsq->addr_ready_cycle = cycles;

          printf("ISSUE ADDR CALC: %s@0x%llx calculated at issue at cycle %lld\n", 
                 q->is_load ? "Load" : "Store", vaddr, cycles);
        }
        md_addr_t physical_addr = lsq->addr;
        if (dtlb) {
          tlb_access_result_t tlb_result = tlb_access(dtlb, lsq->addr, q->tid, &physical_addr);
          if (tlb_result == TLB_MISS) {
            q->done = cycles + 15; /* TLB miss stall */
            issued++;
            thread_issued++;
            continue; /* TLB miss, cannot issue yet */
          }
        }
        lsq->addr = physical_addr;
        
        if (q->is_load) {
          if (!can_issue_load_safely(tid, q, lsq_idx)) {
            continue; // Stall this load
          }
          forward_result_t forward_result = check_store_forwarding(tid, lsq_idx);
          if (forward_result == FORWARD_FULL) {
            q->done = cycles + 1;
            lsq->done = cycles + 1; // Stall this load
            lsq->data_ready = 1;
            tp->ROB[q->rob_idx].ready      = 1;   
            tp->ROB[q->rob_idx].done_cycle = q->done;
          } else if (forward_result == FORWARD_CONFLICT) {
            q->done = cycles + 35; // Stall this load
            lsq->done = q->done;
          } else {
            /* Handle coherence*/
            handle_coherence_transaction(physical_addr, BUS_READ, tid);

            /* Access cache hierarchy */
            tick_t cache_ready_time;
            cache_access_result_t cache_result = cache_access(dl1_cache, physical_addr, 0, tid, &cache_ready_time);
            tctx[tid].dcache_accesses++;
            if (cache_result != CACHE_HIT) {
              tctx[tid].dcache_misses++;
            }

            /* Stride prefetecher */
            stride_prefetcher_access(tid, tp->ROB[q->rob_idx].PC, physical_addr);
            q->done = cache_ready_time;
            lsq->done = cache_ready_time;
          }
        } else {
          /* Store - mark data ready */
          handle_coherence_transaction(physical_addr, BUS_WRITE, tid);

          tick_t cache_ready_time;
          cache_access(dl1_cache, physical_addr, 1, tid, &cache_ready_time);
          tctx[tid].dcache_accesses++;

          lsq->done = cache_ready_time;
          lsq->data_ready = 1;
          lsq->data_ready_cycle = cycles;
          q->done = cache_ready_time;

          struct rob_entry *store_re = &tp->ROB[q->rob_idx];
          store_re->done_cycle = cache_ready_time;
          store_re->ready = 1; /* Store is ready to commit */
        }
      } else {
        /* Non-memory instruction */
        int latency = get_latency(op);
        q->done = cycles + latency;
      }
      memset(q, 0, sizeof(*q));
      tp->iq_head = (tp->iq_head + 1) % IQ_SIZE;

      issued++;
      thread_issued++;
    }
  }
  if (issued == 0) {
    int total_ready_entries = 0;
    for (int t = 0; t < num_hw_threads; t++) {
      if (tctx[t].active) {
        struct thread_pipeline *tp = &tctx[t].pipeline;
        for (int i = 0; i < IQ_SIZE; i++) {
          if (tp->IQ[i].valid && !tp->IQ[i].issued &&
      ((tp->IQ[i].src1<0 || prf_ready[tp->IQ[i].src1]) &&
      (tp->IQ[i].src2<0 || prf_ready[tp->IQ[i].src2]))) {
            total_ready_entries++;
          }
        }
      }
    }
    if (total_ready_entries > 0) {
      issue_iq_empty++;
    }
  }
  issue_rr_ptr = (issue_rr_ptr + 1) % num_hw_threads;
}
static void writeback_stage(){ 
  for (int tid = 0; tid < num_hw_threads; tid++) {
    if (!tctx[tid].active) continue;
    
    struct thread_pipeline *tp = &tctx[tid].pipeline;
    
    for (int idx = 0; idx < IQ_SIZE; idx++) {
      struct iq_entry *q = &tp->IQ[idx];
      if (!q->issued || cycles < q->done) continue;
      
      if (q->rob_idx >= 0 && q->rob_idx < ROB_SIZE) {
        struct rob_entry *re = &tp->ROB[q->rob_idx];
        if (!re->ready) {
          printf("WB→ROB[%d] ready at cycle %lld\n",
+                 q->rob_idx, cycles);
        }
        re->ready = 1;
        re->done_cycle = cycles;

        if (q->is_load || q->is_store) {
          for (int i = tp->lsq_head; i != tp->lsq_tail; i = (i + 1) % LSQ_SIZE) {
            if (tp->LSQ[i].rob_idx == q->rob_idx && tp->LSQ[i].tid == tid) {
              tp->LSQ[i].done        = cycles;
              tp->LSQ[i].data_ready  = 1;
              printf("WB→LSQ[%d] committed at cycle %lld (ROB[%d])\n",
                     i, cycles, q->rob_idx);
              tp->LSQ[i].committed  = 1;
              break;
            }
          }
          // tp->ROB[q->rob_idx].ready      = 1;
          ///tp->ROB[q->rob_idx].done_cycle = cycles;
        }
      }
      
      if (q->dst >= 0 && q->dst < PRF_NUM) {
        prf_ready[q->dst] = 1;
        for (int j = 0; j < IQ_SIZE; j++) {
        struct iq_entry *w = &tp->IQ[j];
        if (!w->valid || w->issued || w->ready) continue;
        if ((w->src1 == q->dst || w->src2 == q->dst) &&
            (w->src1 < 0 || prf_ready[w->src1]) &&
            (w->src2 < 0 || prf_ready[w->src2]))
            w->ready = 1;
          }
      }
      
      memset(q, 0, sizeof(*q));
    }
  }
}

static int memory_stall_count[MAX_HW_THREAD] = {0};

static void commit_stage(void) {
  int commits = 0;  /* total commits this cycle */
  for (int t = 0; t < num_hw_threads && commits < sim_outorder_width; ++t) {

    if (!tctx[t].active) continue;

    struct thread_pipeline *tp = &tctx[t].pipeline;
    commit_blocked[t] = false;  /* Reset commit block signal */
    while (tp->lsq_head != tp->lsq_tail &&
           tp->LSQ[tp->lsq_head].committed)
    {
      /* Clean up committed LSQ entries */
      printf("PRE-COMMIT LSQ[%d] cleanup (rob_idx=%d)\n",
             tp->lsq_head, tp->LSQ[tp->lsq_head].rob_idx);
      memset(&tp->LSQ[tp->lsq_head], 0, sizeof(tp->LSQ[tp->lsq_head]));
      tp->lsq_head = (tp->lsq_head + 1) % LSQ_SIZE;
    }
    int thread_commits         = 0;
    int max_per_thread         = MAX(1, sim_outorder_width / num_hw_threads);
    
    /* === O(1) ROB HEAD PROCESSING === */
    while (tp->rob_head != tp->rob_tail &&
           thread_commits < max_per_thread &&
           commits        < sim_outorder_width)
    {
      struct rob_entry *re = &tp->ROB[tp->rob_head];
      enum md_opcode   op;  
      MD_SET_OPCODE(op, re->inst);

      /* ---- O(1) READINESS CHECK ---- */
      if (!re->ready) {
        if (!(MD_OP_FLAGS(op) & (F_LOAD | F_STORE))) {
          /* Non-memory instruction: check execution completion */
          if (re->done_cycle && cycles >= re->done_cycle)
              re->ready = 1;
        } else {
          /* Memory instruction: check ONLY LSQ HEAD (O(1)) */
          if (tp->lsq_head != tp->lsq_tail) {
            struct lsq_entry *le = &tp->LSQ[tp->lsq_head];
            if (le->rob_idx == tp->rob_head && le->tid == t &&
                le->addr_ready &&
                (cycles >= le->done || le->data_ready))
            {
              re->ready = 1;     
              /* LSQ committed */
            }
          }
        }
      }

      /* If not ready, stop processing and signal stall */
      if (!re->ready) {
        commit_blocked[t] = true;  /* Signal back-pressure */
        printf("COMMIT BLOCKED: Thread %d, ROB[%d] not ready at cycle %lld\n", 
               t, tp->rob_head, cycles);
        break;
      }

      /* ---- COMMIT PROCESSING ---- */

      /* System call handling */
      if (op == CALL_PAL) {
        int func = re->inst & 0xFFFF;
        if (func == OSF_SYS_exit || func == 0x83)
          handle_syscall_exit(t);
      }

      /* Branch resolution */
      if (MD_OP_FLAGS(op) & F_CTRL) {
        md_addr_t actual_target, pred_target;
        int actual_taken =
            resolve_branch(re->inst, re->PC, &tctx[t].regs, &actual_target);
        int pred_taken =
            predict_branch_for_thread(t, re->PC, &pred_target, re->inst);

        int mispred = (pred_taken != actual_taken) ||
                      (actual_taken &&
                       llabs((long long)(pred_target - actual_target)) > 4);

        update_branch_predictor_for_thread(t, re->PC, actual_taken, actual_target, re->inst);
        update_global_branch_stats(t, re->PC, actual_taken, actual_target, re->inst);
        
        if (mispred) {
          tctx[t].bp.bp_mispred++;  
          bp_mispred++;

          if (re->old_phys != -1) prf_free(re->old_phys);
          
          unsigned btb_idx      = (re->PC >> 2) % BTB_SIZE;
          btb[btb_idx].tag      = re->PC;
          btb[btb_idx].target   = actual_target;
          btb[btb_idx].valid    = 1;
          
          enhanced_flush_thread(t);
          tctx[t].pc = actual_taken ? actual_target : (re->PC + 4);
        } else {
          tctx[t].bp.bp_correct++;  
          bp_correct++;
        }
      }

      /* Execute instruction */
      if (!re->is_load && !re->is_store)
          execute_alpha_instruction(re->inst, &tctx[t].regs, re->PC);

      /* Free old physical register */
      if (re->old_phys != -1) prf_free(re->old_phys);

      /* Mark LSQ entry as committed (single assignment) */
      if ((re->is_load || re->is_store) &&
          tp->lsq_head != tp->lsq_tail)
      {
        struct lsq_entry *le = &tp->LSQ[tp->lsq_head];
        if (le->rob_idx == tp->rob_head && le->tid == t)
            le->committed = 1;  
      }

      /* Update statistics and advance pointers */
      sim_num_insn++;          
      sim_num_insn_tid[t]++;
      if (tctx[t].speculation_depth > 0)
          tctx[t].speculation_depth--;

      tp->rob_head = (tp->rob_head + 1) % ROB_SIZE;
      memset(re, 0, sizeof(*re));

      commits++;  
      thread_commits++;
      printf("COMMIT SUCCESS: Thread %d, insn %lld at cycle %lld\n", 
             t, sim_num_insn_tid[t], cycles);
      /* ---- O(1) LSQ HEAD CLEANUP ---- */
      if (tp->lsq_head != tp->lsq_tail &&
          tp->LSQ[tp->lsq_head].committed)
      {
        memset(&tp->LSQ[tp->lsq_head], 0, sizeof(tp->LSQ[tp->lsq_head]));
        tp->lsq_head = (tp->lsq_head + 1) % LSQ_SIZE;
        printf("LSQ HEAD ADVANCE: Thread %d, new head=%d\n", t, tp->lsq_head);
      }
      if (thread_commits > 0) commit_blocked[t] = false;
    } /* while ROB */
  }   /* for each thread */
  if (commits == 0) {
    printf("NO COMMITS at cycle %lld\n", cycles);
  }
}
/* ===== T H R E A D ================================================== */
static void flush_thread(int tid){
  if (tid < 0 || tid >= num_hw_threads) return;

  tctx[tid].last_flush_cycle = cycles;
  tctx[tid].flush_count++;
  tctx[tid].speculation_depth = 0;
  
  /* Comprehensive pipeline flush */
  struct thread_pipeline *tp = &tctx[tid].pipeline;

  /* Flush IFQ */
  memset(tp->IFQ, 0, sizeof(tp->IFQ));
  tp->ifq_head = tp->ifq_tail = 0;

  /* Flush IQ */
  for (int i = 0; i < IQ_SIZE; ++i) {
    if (tp->IQ[i].ready && tp->IQ[i].tid == tid) {
      memset(&tp->IQ[i], 0, sizeof(tp->IQ[i]));
    }
  }

  /* Handle ROB entries before flushing */
  for (int i = tp->rob_head; i != tp->rob_tail; i = (i + 1) % ROB_SIZE) {
    if (tp->ROB[i].tid == tid) {
      if (!tp->ROB[i].ready && tp->ROB[i].new_phys != -1) {
        prf_free(tp->ROB[i].new_phys);
      }
      memset(&tp->ROB[i], 0, sizeof(tp->ROB[i]));
    }
  }
  
  /* Flush ROB */
  tp->rob_head = tp->rob_tail = 0;

  /* Flush AGU */
  for (int i = 0; i < AGU_SIZE; i++) {
    if (tp->AGU[i].valid && tp->AGU[i].tid == tid) {
      tp->AGU[i].valid = 0;
    }
  }

  /* Flush LSQ */
  memset(tp->LSQ, 0, sizeof(tp->LSQ));
  tp->lsq_head = tp->lsq_tail = 0;

  /* Restore rename map to architectural state */
  for (int r = 0; r < MD_TOTAL_REGS; r++) {
    tctx[tid].rename_map[r] = r;
    prf_ready[r] = (r < MD_TOTAL_REGS) ? 1 : 0;
  }
  
  /* Reset fetch counters */
  tctx[tid].icount = 0;
  printf("Thread %d flushed at cycle %lld (flush #%lld)\n", 
           tid, cycles, tctx[tid].flush_count);
}

/* Smart Thread Scheduling with Flush Awareness */
static int smart_thread_selection(int *fetch_order, int *fetch_count) {
  *fetch_count = 0;
  
  /* Update all thread performance metrics */
  for (int t = 0; t < num_hw_threads; t++) {
    if (tctx[t].active) {
      update_thread_performance(t);
    }
  }
    
  /* Collect eligible threads with priorities */
  typedef struct {
    int tid;
    int priority;
    int starvation_cycles;
    double fairness_boost;
  } thread_candidate_t;
  
  thread_candidate_t candidates[MAX_HW_THREAD];
  int candidate_count = 0;
  
  for (int t = 0; t < num_hw_threads; t++) {
    if (!tctx[t].active) continue;
    
    candidates[candidate_count].tid = t;
    candidates[candidate_count].priority = calculate_thread_priority(t);
    candidates[candidate_count].starvation_cycles = thread_perf[t].cycles_since_last_fetch;
    if (candidates[candidate_count].starvation_cycles > 100) {
      candidates[candidate_count].fairness_boost = 
      MIN(candidates[candidate_count].starvation_cycles / 50.0, 5.0);
      candidates[candidate_count].priority += (int)(candidates[candidate_count].fairness_boost * 20);
    } else {
      candidates[candidate_count].fairness_boost = 0.0;
    }
    
    candidate_count++;
  }
  
  if (candidate_count == 0) return -1;
  
  /* Sort candidates by priority (descending) */
  for (int i = 0; i < candidate_count - 1; i++) {
    for (int j = i + 1; j < candidate_count; j++) {
      int swap = 0;
      
      /* Primary sort: priority */
      if (candidates[j].starvation_cycles > 500 && 
          candidates[i].starvation_cycles <= 500) {
        swap = 1;
      }
      /* Secondary sort: starvation prevention */
      else if (candidates[j].starvation_cycles <= 500 && 
          candidates[i].starvation_cycles <= 500) {
        if (candidates[j].starvation_cycles > candidates[i].starvation_cycles) {
          swap = 1;
        }
      }
      
      if (swap) {
        thread_candidate_t temp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = temp;
      }
    }
  }
  
  /* Apply scheduling policy specific adjustments */
  if (scheduler.current_policy == SCHED_ROUND_ROBIN) {
      /* Override with round-robin */
    int start_tid = scheduler.fetch_round_robin_ptr;
    candidate_count = 0;
    
    for (int i = 0; i < num_hw_threads; i++) {
      int tid = (start_tid + i) % num_hw_threads;
      if (tctx[tid].active) {
        candidates[candidate_count].tid = tid;
        candidate_count++;
      }
    }
    scheduler.fetch_round_robin_ptr = (scheduler.fetch_round_robin_ptr + 1) % num_hw_threads;
  }
  
  /* Fill fetch order */
  for (int i = 0; i < candidate_count; i++) {
    fetch_order[i] = candidates[i].tid;
  }
  *fetch_count = candidate_count;
  
  /* Update scheduler statistics */
  if (candidate_count > 0) {
    int selected_tid = fetch_order[0];
    thread_perf[selected_tid].last_fetch_cycle = cycles;
    thread_perf[selected_tid].total_fetch_cycles++;
    thread_perf[selected_tid].cycles_since_last_fetch = 0;
  }
  
  return (candidate_count > 0) ? fetch_order[0] : -1;
}

/* Adaptive scheduling policy adjustment */
static void adapt_scheduling_policy() {
  if (!enable_performance_feedback) return;
  
  scheduler.cycles_since_adaptation++;
  if (scheduler.cycles_since_adaptation < scheduler.adaptation_interval) {
    return;
  }
  
  scheduler.cycles_since_adaptation = 0;
  
  /* Calculate system metrics */
  double total_ipc = safe_ratio(sim_num_insn, cycles);
  double fairness_variance = 0.0;
  double avg_thread_ipc = 0.0;
  int active_threads = 0;
  
  for (int t = 0; t < num_hw_threads; t++) {
    if (tctx[t].active) {
      double thread_ipc = safe_ratio(sim_num_insn_tid[t], cycles);
      avg_thread_ipc += thread_ipc;
      active_threads++;
    }
  }
  
  if (active_threads > 0) {
    avg_thread_ipc /= active_threads;
    
    /* Calculate fairness (variance in IPC) */
    for (int t = 0; t < num_hw_threads; t++) {
      if (tctx[t].active) {
        double thread_ipc = safe_ratio(sim_num_insn_tid[t], cycles);
        double diff = thread_ipc - avg_thread_ipc;
        fairness_variance += diff * diff;
      }
    }
    fairness_variance = sqrt(fairness_variance / active_threads);
  }
  
  scheduler.system_throughput = total_ipc;
  scheduler.system_fairness = 1.0 / (1.0 + fairness_variance);
  
  /* Adapt policy based on metrics */
  scheduling_policy_t new_policy = scheduler.current_policy;
  
  if (fairness_variance > 0.3 && scheduler.system_fairness < 0.7) {
    /* High unfairness - switch to fairness-oriented policy */
    if (scheduler.current_policy != SCHED_ROUND_ROBIN) {
      new_policy = SCHED_ROUND_ROBIN;
      printf("Cycle %lld: Switching to Round Robin (fairness=%.3f)\n", 
              cycles, scheduler.system_fairness);
    }
  } else if (total_ipc < avg_thread_ipc * active_threads * 0.8) {
    /* Low throughput - switch to performance-oriented policy */
    if (scheduler.current_policy != SCHED_PERFORMANCE_FEEDBACK) {
      new_policy = SCHED_PERFORMANCE_FEEDBACK;
      printf("Cycle %lld: Switching to Performance Feedback (IPC=%.3f)\n", 
              cycles, total_ipc);
    }
  } else {
      /* Balanced performance - use adaptive policy */
    if (scheduler.current_policy != SCHED_ADAPTIVE) {
      new_policy = SCHED_ADAPTIVE;
      printf("Cycle %lld: Switching to Adaptive policy\n", cycles);
    }
  }
  
  scheduler.current_policy = new_policy;
}
static void enhanced_flush_thread(int tid) {
  if (tid < 0 || tid >= num_hw_threads) return;
  
  printf("PIPELINE FLUSH: Thread %d at cycle %lld (violation recovery)\n", tid, cycles);
  
  thread_performance_t *perf = &thread_perf[tid];
  struct thread_pipeline *tp = &tctx[tid].pipeline;
  
  perf->recent_flush_count++;
  perf->flush_penalty_remaining = 50;
  
  tctx[tid].speculation_depth = 0;
  tctx[tid].last_flush_cycle = cycles;
  tctx[tid].flush_count++;

  // Clear all pipeline stages
  memset(tp->IFQ, 0, sizeof(tp->IFQ));
  tp->ifq_head = tp->ifq_tail = 0;

  for (int i = 0; i < IQ_SIZE; i++) {
    if (tp->IQ[i].tid == tid) {
      if (tp->IQ[i].dst >= 0 && tp->IQ[i].dst < PRF_NUM) {
        prf_free(tp->IQ[i].dst);
      }
      memset(&tp->IQ[i], 0, sizeof(tp->IQ[i]));
    }
  }

  // Clear ROB with register cleanup
  for (int i = tp->rob_head; i != tp->rob_tail; i = (i + 1) % ROB_SIZE) {
    struct rob_entry *re = &tp->ROB[i];
    if (re->tid == tid) {
      if (re->new_phys != -1) {
        prf_free(re->new_phys);
      }
      memset(re, 0, sizeof(*re));
    }
  }
  tp->rob_head = tp->rob_tail = 0;

  // Clear LSQ completely
  memset(tp->LSQ, 0, sizeof(tp->LSQ));
  tp->lsq_head = tp->lsq_tail = 0;

  // Clear AGU
  for (int i = 0; i < AGU_SIZE; i++) {
    if (tp->AGU[i].tid == tid) {
      memset(&tp->AGU[i], 0, sizeof(tp->AGU[i]));
    }
  }

  // Reset rename map to architectural state
  for (int r = 0; r < MD_TOTAL_REGS; r++) {
    tctx[tid].rename_map[r] = r;
  }
  
  // Reset architectural registers state
  for (int r = 0; r < MD_TOTAL_REGS; r++) {
    prf_ready[r] = 1;
  }

  tctx[tid].icount = 0;
  tctx[tid].seq = 0;
  
  printf("  → Pipeline completely flushed and reset\n");
}
/* Resource allocation based on thread performance */
static void dynamic_resource_allocation() {
  if (!enable_dynamic_partitioning) {
        /* 고정 균등 분할 */
        for (int t = 0; t < num_hw_threads; t++) {
            resource_partitions[t].fetch_slots = IFQ_SIZE / num_hw_threads;
            resource_partitions[t].rename_slots = ROB_SIZE / num_hw_threads;
            resource_partitions[t].issue_slots = IQ_SIZE / num_hw_threads;
            resource_partitions[t].lsq_slots = LSQ_SIZE / num_hw_threads;
        }
        return;
    }
    
    /* 동적 할당 로직 */
    double total_weight = 0.0;
    double thread_weights[MAX_HW_THREAD];
    
    for (int t = 0; t < num_hw_threads; t++) {
        if (tctx[t].active) {
            /* 성능 기반 가중치 계산 */
            double recent_ipc = safe_ratio(sim_num_insn_tid[t], 
                                         MAX(1, thread_stall_stats[t].active_cycles));
            double utilization = (resource_tracker[t].sample_count > 0) ?
                (resource_tracker[t].ifq_occupancy_sum + 
                 resource_tracker[t].rob_occupancy_sum +
                 resource_tracker[t].iq_occupancy_sum + 
                 resource_tracker[t].lsq_occupancy_sum) / 
                (4.0 * resource_tracker[t].sample_count) : 0.5;
            
            /* 가중치: IPC가 높고 자원 활용도가 높은 스레드에게 더 많이 할당 */
            thread_weights[t] = recent_ipc * (0.5 + utilization) + 0.1; /* 최소 가중치 */
            total_weight += thread_weights[t];
        } else {
            thread_weights[t] = 0.0;
        }
    }
    
    if (total_weight > 0) {
        for (int t = 0; t < num_hw_threads; t++) {
            if (tctx[t].active) {
                double proportion = thread_weights[t] / total_weight;
                
                /* 최소 보장 + 비례 할당 */
                int min_slots = 2;
                resource_partitions[t].fetch_slots = 
                    MIN(IFQ_SIZE, MAX(min_slots, (int)(IFQ_SIZE * proportion)));
                resource_partitions[t].rename_slots = 
                    MIN(ROB_SIZE, MAX(min_slots * 4, (int)(ROB_SIZE * proportion)));
                resource_partitions[t].issue_slots = 
                    MIN(IQ_SIZE, MAX(min_slots, (int)(IQ_SIZE * proportion)));
                resource_partitions[t].lsq_slots = 
                    MIN(LSQ_SIZE, MAX(min_slots, (int)(LSQ_SIZE * proportion)));
            }
        }
        
        printf("Dynamic allocation at cycle %lld: weights=[%.2f,%.2f]\n", 
               cycles, thread_weights[0], thread_weights[1]);
    }
}
static void update_resource_utilization(int tid) {
    if (tid < 0 || tid >= num_hw_threads) return;
    
    struct thread_pipeline *tp = &tctx[tid].pipeline;
    resource_tracker_t *rt = &resource_tracker[tid];
    
    /* 현재 점유율 계산 */
    int ifq_count = (tp->ifq_tail - tp->ifq_head + IFQ_SIZE) % IFQ_SIZE;
    int rob_count = (tp->rob_tail - tp->rob_head + ROB_SIZE) % ROB_SIZE;
    int iq_count = (tp->iq_tail - tp->iq_head + IQ_SIZE) % IQ_SIZE;
    int lsq_count = (tp->lsq_tail - tp->lsq_head + LSQ_SIZE) % LSQ_SIZE;
    
    /* 누적 합계 업데이트 */
    rt->ifq_occupancy_sum += ifq_count;
    rt->rob_occupancy_sum += rob_count;
    rt->iq_occupancy_sum += iq_count;
    rt->lsq_occupancy_sum += lsq_count;
    rt->sample_count++;
    
    /* 슬라이딩 윈도우 업데이트 */
    rt->ifq_samples[rt->sample_index] = ifq_count;
    rt->rob_samples[rt->sample_index] = rob_count;
    rt->iq_samples[rt->sample_index] = iq_count;
    rt->lsq_samples[rt->sample_index] = lsq_count;
    rt->sample_index = (rt->sample_index + 1) % 100;
    
    /* performance_counters 업데이트 */
    if (rt->sample_count > 0) {
        perf_counters[tid].avg_ifq_occupancy = rt->ifq_occupancy_sum / rt->sample_count;
        perf_counters[tid].avg_rob_occupancy = rt->rob_occupancy_sum / rt->sample_count;
        perf_counters[tid].avg_iq_occupancy = rt->iq_occupancy_sum / rt->sample_count;
        perf_counters[tid].avg_lsq_occupancy = rt->lsq_occupancy_sum / rt->sample_count;
    }
}
/* Performance analysis and reporting */
static void analyze_scheduling_performance() {
  double total_ipc = safe_ratio(sim_num_insn, cycles);
  double avg_thread_ipc = 0.0;
  int active_threads = 0;

  for (int t = 0; t < num_hw_threads; t++) {
    if (sim_num_insn_tid[t] > 0) {
      double thread_ipc = safe_ratio(sim_num_insn_tid[t], cycles);
      avg_thread_ipc += thread_ipc;
      active_threads++;
      
      // Fairness score 업데이트
      thread_perf[t].fairness_score = (avg_thread_ipc > 0) ? 
          thread_ipc / avg_thread_ipc : 1.0;
    }
  }

  if (active_threads > 0) {
    avg_thread_ipc /= active_threads;
    
    // Fairness variance 계산
    double fairness_variance = 0.0;
    for (int t = 0; t < num_hw_threads; t++) {
      if (sim_num_insn_tid[t] > 0) {
        double thread_ipc = safe_ratio(sim_num_insn_tid[t], cycles);
        double diff = thread_ipc - avg_thread_ipc;
        fairness_variance += diff * diff;
      }
    }
    fairness_variance = sqrt(fairness_variance / active_threads);
    
    scheduler.system_throughput = total_ipc;
    scheduler.system_fairness = (fairness_variance > 0) ? 
        1.0 / (1.0 + fairness_variance) : 1.0;
  }
  printf("\n=== Enhanced Thread Scheduling Analysis ===\n");
  printf("Current Policy: %s\n", 
          scheduler.current_policy == SCHED_ADAPTIVE ? "Adaptive" :
          scheduler.current_policy == SCHED_PERFORMANCE_FEEDBACK ? "Performance Feedback" :
          scheduler.current_policy == SCHED_ICOUNT ? "ICOUNT" : "Round Robin");
  
  printf("System Throughput: %.3f IPC\n", scheduler.system_throughput);
  printf("System Fairness: %.3f\n", scheduler.system_fairness);
  printf("Cycles: %lld\n", cycles);
  printf("Total Instructions: %lld\n", sim_num_insn);
  printf("Active Threads: %d\n", num_hw_threads);
  
  printf("\n--- Per-Thread Scheduling Metrics ---\n");
  for (int t = 0; t < num_hw_threads; t++) {
    if (sim_num_insn_tid[t] > 0) {
      thread_performance_t *perf = &thread_perf[t];
      
      printf("Thread %d:\n", t);
      printf("  Recent IPC: %.3f (avg: %.3f)\n", 
              perf->recent_ipc, perf->avg_ipc);
      printf("  Dynamic Priority: %d (base: %d)\n", 
              perf->dynamic_priority, perf->base_priority);
      double branch_acc = (tctx[t].branches_executed > 0) ? 
            safe_ratio((tctx[t].branches_executed - tctx[t].branches_mispredicted), tctx[t].branches_executed) * 100.0 : 0;
      if (branch_acc < 0) branch_acc = 0;
      printf("  Branch Accuracy: %.1f%%\n", branch_acc);
      printf("  Cache Miss Rate: %.1f%%\n", safe_ratio(tctx[t].dcache_misses, tctx[t].dcache_accesses) * 100.0);
      printf("  Resource Efficiency: %.3f\n", perf->resource_efficiency);
      printf("  Fairness Score: %.3f\n", perf->fairness_score);
      printf("  Fetch Cycles: %lld\n", perf->total_fetch_cycles);
      printf("  Active Penalties: flush=%d, cache=%d, starvation=%d\n",
              perf->flush_penalty_remaining, perf->cache_miss_penalty,
              perf->resource_starvation_penalty);
      
      /* Resource allocation */
      printf("  Resource Allocation: IFQ=%d, ROB=%d, IQ=%d, LSQ=%d\n",
              resource_partitions[t].fetch_slots,
              resource_partitions[t].rename_slots,
              resource_partitions[t].issue_slots,
              resource_partitions[t].lsq_slots);
    }
  }
    
  /* Scheduling effectiveness analysis */
  double total_flushes = 0;
  for (int t = 0; t < num_hw_threads; t++) {
    total_flushes += tctx[t].flush_count;
  }
  
  printf("\n--- Scheduling Effectiveness ---\n");
  printf("Average flushes per thread: %.1f\n", total_flushes / num_hw_threads);
  printf("Pipeline utilization: %.1f%%\n", 
          safe_ratio(sim_num_insn, cycles * sim_outorder_width) * 100.0);
  
  /* Resource contention analysis */
  double avg_fetch_stalls = safe_ratio(fetch_ifq_full, cycles) * 100.0;
  double avg_rename_stalls = safe_ratio(rename_rob_full + rename_iq_full + rename_lsq_full, cycles) * 100.0;
  double avg_issue_stalls = safe_ratio(issue_iq_empty, cycles) * 100.0;
  
  printf("Resource contention:\n");
  printf("  Fetch stalls: %.1f%% of cycles\n", avg_fetch_stalls);
  printf("  Rename stalls: %.1f%% of cycles\n", avg_rename_stalls);
  printf("  Issue stalls: %.1f%% of cycles\n", avg_issue_stalls);
}
void report_fetch_stats(void){
    printf("\n==== FETCH STALL STATISTICS ====\n");
    printf("  tid | rename | rob_full | iq_full | ifq_full\n");
    printf("------+--------+----------+---------+---------\n");
    for (int tid = 0; tid < num_hw_threads; tid++) {
        printf("  %2d  | %6lld | %8lld | %7lld | %8lld\n",
            tid,
            fetch_stall_rename[tid],
            fetch_stall_rob_full[tid],
            fetch_stall_iq_full[tid],
            fetch_stall_ifq_full[tid]);
    }
    printf("================================\n");
}