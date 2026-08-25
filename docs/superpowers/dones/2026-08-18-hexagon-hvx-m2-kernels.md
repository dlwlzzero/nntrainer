# M2: DSP 실행기와 HVX 커널 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** qwen3-0.6b용 op 9종 HVX 커널과 op-list 해석 실행기를 구현하고 hexagon-sim(x86)에서 C 레퍼런스 대비 전부 통과시킨다.

**Architecture:** DSP 측(`nntrainer/tensor/hexagon/htp/`)에 워커 풀 + op 디스패치 실행기를 추가하고, op별 HVX 커널을 QuRT 시뮬레이터(runelf.pbn + run_main_on_hexagon_sim)에서 스칼라 C 레퍼런스와 golden 비교한다. FastRPC 글루(executor.c)는 마지막에 실행기와 연결하되, M1 더미 경로(`n_ops==0`)를 유지해 기존 디바이스 테스트를 깨지 않는다.

**Tech Stack:** hexagon-clang 8.7.08 (HVX v75, 128B), Hexagon SDK 6.0.0.2 (QuRT sim, run_main_on_hexagon), ggml-hexagon 차용분(MIT), 스펙: `docs/superpowers/specs/hexagon-hvx/02-hvx-kernels.md`

## Global Constraints

- **승인 게이트 (사용자 메모리, 최우선):** 에이전트는 빌드/테스트를 **직접 실행하지 않는다**. 각 Task의 검증 단계는 "명령어 제시 → 정지 → 사용자가 실행한 출력 수신 → 판정"이며, 출력을 받기 전 통과/실패 판정·완료 보고 금지. 사용자 승인 후에만 커밋하고 다음 Task로 진행한다. (이 게이트 때문에 표준 TDD의 "실패 확인 실행" 단계는 생략하고, 테스트+구현을 함께 제시한 뒤 1회 검증으로 간다.)
- **커밋 형식:** 제목(`[htp] ...`) + 바디 + `Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>` + `Co-authored-by: Claude Fable 5 <noreply@anthropic.com>`
- **SDK 환경 (사용자가 셸에서 설정):** `source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.0.0.2/setup_sdk_env.source` — 이후 `HEXAGON_SDK_ROOT`, `DEFAULT_HEXAGON_TOOLS_ROOT` 사용 가능
- **시뮬레이터 타깃은 v75:** SDK 6.0.0.2의 QuRT sim 이미지가 v75까지만 제공됨. 디바이스 skel은 기존대로 v79(`build_skel.sh`).
- **정렬 규칙:** 모든 텐서 오프셋·행 stride 128B 정렬. 커널에 비정렬 처리 없음. 검증은 전부 init(validate)에서, forward 경로에 검사 없음.
- **데이터 타입:** op 사이 fp16(`__fp16`), int8은 matmul 내부만, 누산 int32(vrmpy)/fp32(널리 민감 구간). per-token 동적 양자화는 스칼라 C로 구현(비용 O(M·K) ≈ 전체 0.1%, 스펙상 최적화 대상 아님).
- **워커 수:** `qurt_hvx_get_units()` 런타임 조회. 하드코딩 금지.
- **차용 코드:** ggml-hexagon(MIT) 파일은 원 저작권 헤더 유지 + SPDX MIT + 출처 주석 추가, `NOTICE`에 항목 추가. 출처: `/home/j2z0/Project/llama.cpp/ggml/src/ggml-hexagon/htp/`
- **ABI:** `NNTR_HTP_ABI_VERSION` 1→2 범프(op 디스크립터 추가). M1 디바이스 테스트(`hexagon_rpc_test.cpp`)는 `n_ops==0` 더미 경로 유지로 그대로 PASS해야 한다.
- **plan/spec 문서는 커밋하지 않는다** (`docs/superpowers/`는 gitignore 대상).
- 빌드 산출물은 `build_hexagon/` (gitignored).

---

## 공통 설계 (모든 Task의 단일 기준)

### 와이어 포맷 v2 (`nntr_htp_common.h`)

```c
#define NNTR_HTP_ABI_VERSION 2u

enum nntr_htp_buf_id {
  NNTR_HTP_BUF_WEIGHTS = 0,
  NNTR_HTP_BUF_KV = 1,
  NNTR_HTP_BUF_ACT = 2,
  NNTR_HTP_BUF_TOKENS = 3, /* forward() token_ids 인자, 매 호출 갱신 */
  NNTR_HTP_BUF_LOGITS = 4, /* forward() logits 인자, 매 호출 갱신 */
  NNTR_HTP_BUF_COUNT = 5
};

enum nntr_htp_op_kind {
  NNTR_HTP_OP_EMBED = 0,
  NNTR_HTP_OP_RMSNORM = 1,
  NNTR_HTP_OP_MATMUL_W8A8 = 2,
  NNTR_HTP_OP_ROPE = 3,
  NNTR_HTP_OP_ATTN = 4,
  NNTR_HTP_OP_SILU_MUL = 5,
  NNTR_HTP_OP_ADD = 6,
  NNTR_HTP_OP_MATMUL_LOGITS = 7,
  NNTR_HTP_OP_KIND_COUNT = 8
};

#define NNTR_HTP_FLAG_PER_HEAD 0x1u /* RMSNORM: head_dim 단위 QK-Norm */

struct nntr_htp_tensor_ref {
  uint32_t buf;    /* enum nntr_htp_buf_id */
  uint32_t offset; /* bytes, 128B 정렬 */
};                 /* 8B */

struct nntr_htp_op_desc {
  uint32_t kind, flags, layer;
  uint32_t m, k, n; /* m==0 → 실행 시 n_tokens로 치환 */
  struct nntr_htp_tensor_ref in0, in1, in2, out;
  uint32_t param0, param1; /* op별 (fp32 비트패턴 등) */
};                         /* 64B, size check 필수 */

struct nntr_htp_oplist_header {
  uint32_t magic, version, n_ops, reserved; /* v1 접두 유지 (버전 체크 호환) */
  uint32_t n_layers, n_heads, n_kv_heads, head_dim;
  uint32_t hidden, ffn, vocab, max_seq;
  uint32_t max_chunk;
  uint32_t reserved2[3];
}; /* 64B, size check 필수 */
```

### op 시맨틱스 표

| kind | in0 | in1 | in2 | out | m/k/n | flags·param |
|---|---|---|---|---|---|---|
| EMBED | TOKENS int32[m] | W int8[vocab][k] | scale fp32[vocab] | fp16[m][k] | m=0, k=hidden | |
| RMSNORM | x fp16[m][n] | gamma fp16[n] (PER_HEAD면 [head_dim]) | – | fp16[m][n] | m=0 | PER_HEAD, param0=eps(f32비트) |
| MATMUL_W8A8 | X fp16[m][k] | W int8[n][k] | scale fp32[n] | Y fp16[m][n] | m=0 | |
| ROPE | q fp16[m][n_heads·128] 제자리 | k fp16[m][n_kv_heads·128] 제자리 | cos/sin fp16[max_seq][128] | =in0 (관례) | m=0 | 실행 ctx의 pos 사용 |
| ATTN | q fp16[m][n_heads·128] | k fp16[m][n_kv_heads·128] | v fp16[m][n_kv_heads·128] | fp16[m][n_heads·128] | m=0 | layer→KV 주소, param0=1/√head_dim(f32비트) |
| SILU_MUL | gate fp16[m][n] | up fp16[m][n] | – | fp16[m][n] | m=0, n=ffn | |
| ADD | a fp16[m][n] | b fp16[m][n] | – | fp16[m][n] | m=0 | |
| MATMUL_LOGITS | X fp16[m][k]의 마지막 행 | W int8[n][k] | scale fp32[n] | LOGITS fp32[n] | m=1, k=hidden, n=vocab | |

- **KV 레이아웃:** K 영역 `[n_layers][n_kv_heads][max_seq][head_dim]` fp16, 이어서 V 영역 동일 크기. `k_base(layer,h) = ((layer*n_kv_heads + h)*max_seq)*head_dim*2`, V는 `+ n_layers*n_kv_heads*max_seq*head_dim*2`. head_dim=128 → 행 256B 정렬 유지.
- **ROPE 테이블:** WEIGHTS 내 fp16 `[max_seq][128]`: 행 = cos[64] ‖ sin[64] (rotate-half 쌍 (i, i+64), qwen3 rope_theta=1e6). M3 lowering이 생성하고, M2 테스트는 자체 생성.
- **GQA 매핑:** q 헤드 h는 kv 헤드 `h / (n_heads/n_kv_heads)` 사용.

### 실행 컨텍스트와 op 진입점 (`ops/htp_ops.h`)

```c
struct htp_exec_ctx {
  uint8_t *buf[NNTR_HTP_BUF_COUNT];
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  const struct nntr_htp_oplist_header *cfg;
  uint32_t n_tokens, pos;
  struct wp_pool *pool;
  int8_t *xq;        /* per-token 양자화 스크래치 [max_chunk][k_max] */
  float *xq_scale;   /* [max_chunk] */
  float *attn_scratch; /* [n_workers][max_seq] fp32 (scores) */
  uint8_t *vtcm;     /* Task 7부터, 워커별 분할 */
  uint32_t vtcm_size;
};
typedef void (*htp_op_fn)(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d);
```

- 각 op 함수가 내부에서 `wp_run`으로 병렬 구간을 돌리고 리턴 = 배리어. 실행기는 op을 순차 호출만 한다.
- 헬퍼: `htp_ref_ptr(c,r) = c->buf[r.buf] + r.offset`, `htp_m(c,d) = d->m ? d->m : c->n_tokens`.

### 테스트 하네스 규약

- 시뮬레이터 테스트는 `libnntr_sim_test.so` 하나에 전부 링크, `main(argc,argv)`가 `argv[1]` 이름으로 디스패치. 출력 접두 `SIM_TEST`, 성공 시 `SIM_TEST <name> PASS` + return 0.
- 레퍼런스는 `test/hexagon/sim/ref_ops.{h,c}` — 스칼라 C, fp32 연산(양자화 정수 연산은 커널과 동일 산식으로 bit-exact).
- 결정적 PRNG (테스트 공용, `sim_test_util.h`):

```c
static uint32_t rng_state = 12345u;
static inline float frand(void) { /* [-1, 1) */
  rng_state = rng_state * 1664525u + 1013904223u;
  return (((rng_state >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
}
static inline int cmp_f(const char *tag, const float *ref, const float *got,
                        uint32_t n, float rtol, float atol) {
  float worst = 0.f; uint32_t wi = 0;
  for (uint32_t i = 0; i < n; ++i) {
    float d = fabsf(ref[i] - got[i]), t = atol + rtol * fabsf(ref[i]);
    if (d - t > worst) { worst = d - t; wi = i; }
  }
  if (worst > 0.f) {
    printf("SIM_TEST %s FAIL i=%u ref=%f got=%f\n", tag, wi, ref[wi], got[wi]);
    return 1;
  }
  return 0;
}
```

- 허용 오차(rtol/atol): matmul 2e-3/1e-3 · rmsnorm 5e-3/2e-3 · rope 5e-3/2e-3 · add·embed 1e-3/1e-4 · silu_mul 2e-2/5e-3 · attn 2e-2/5e-3 · logits(fp32) 5e-3/1e-2 · graph 최종 logits 3e-2/5e-2

### 파일 구조 (최종)

```
nntrainer/tensor/hexagon/
├── htp/
│   ├── nntr_htp_common.h        # [수정] 와이어 포맷 v2 + validate
│   ├── executor.c               # [수정] FastRPC 글루 → htp_graph 연결
│   ├── htp_graph.{h,c}          # [신규] op-list 해석 실행기
│   ├── worker_pool.{h,c}        # [신규] QuRT 워커 풀 + 배리어
│   ├── ops/
│   │   ├── htp_ops.h            # [신규] ctx·op 시그니처·디스패치 표
│   │   ├── op_matmul.c          # MATMUL_W8A8 + MATMUL_LOGITS
│   │   ├── op_rmsnorm.c / op_rope.c / op_attn.c
│   │   ├── op_eltwise.c         # ADD + SILU_MUL
│   │   └── op_embed.c
│   └── hvx/
│       ├── hvx_quant.h          # [신규] per-token 양자화(스칼라) + int8 dot(HVX)
│       ├── hvx_f16.h            # [신규] fp16 벡터 헬퍼(위드닝 누산 등)
│       └── (ggml 차용) dma-queue.{c,h}, hvx-exp.h, hvx-base.h,
│            hvx-floor.h, hvx-inverse.h, hex-utils.h
├── host/ (M2 변경 없음)
└── meson.build (M2 변경 없음 — skel 소스는 build_skel.sh가 관리)

test/hexagon/sim/
├── sim_test_main.c  sim_test_util.h  ref_ops.{h,c}
└── test_smoke.c test_pool.c test_exp.c test_quant.c test_matmul.c
    test_matmul_dma.c test_rmsnorm.c test_rope.c test_eltwise.c
    test_embed.c test_attn.c test_logits.c test_graph.c

tools/hexagon/
├── build_sim_test.sh  run_sim_test.sh   # [신규]
└── build_skel.sh                        # [수정, Task 15]
```

---

### Task 1: 시뮬레이터 테스트 하네스

**Files:**
- Create: `tools/hexagon/build_sim_test.sh`, `tools/hexagon/run_sim_test.sh`
- Create: `test/hexagon/sim/sim_test_main.c`, `test/hexagon/sim/sim_test_util.h`, `test/hexagon/sim/test_smoke.c`

**Interfaces:**
- Produces: `build_sim_test.sh`(전체 sim 테스트 lib 빌드), `run_sim_test.sh <test-name>`(hexagon-sim 실행, exit 0=PASS). 이후 모든 Task가 이 두 스크립트로 검증.
- Produces: `sim_test_util.h`의 `frand()`/`rng_state`/`cmp_f()` (위 공통 설계 코드 그대로).

- [ ] **Step 1: 스모크 테스트 작성** — QuRT 스레드, HVX 유닛 조회·lock, HVX vadd, VTCM 확보(HAP_compute_res)를 한 번에 검증해 이후 Task의 환경 리스크를 전부 여기서 소진한다.

```c
/* test/hexagon/sim/test_smoke.c */
#include <stdio.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <qurt.h>
#include <HAP_compute_res.h>

static void thread_fn(void *arg) { *(volatile int *)arg = 1; }

int test_smoke(void) {
  int units = qurt_hvx_get_units();
  printf("SIM_TEST smoke hvx_units=%d\n", units);
  if (units <= 0) return 1;

  /* QuRT thread round-trip */
  static char stack[8192] __attribute__((aligned(16)));
  volatile int flag = 0;
  qurt_thread_t tid; qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, sizeof(stack));
  qurt_thread_attr_set_priority(&attr, 100);
  if (qurt_thread_create(&tid, &attr, thread_fn, (void *)&flag) != QURT_EOK)
    return 1;
  int status;
  qurt_thread_join(tid, &status);
  if (flag != 1) return 1;

  /* HVX vadd: 128B int8 */
  if (qurt_hvx_lock(QURT_HVX_MODE_128B) != QURT_EOK) return 1;
  static int8_t a[128] __attribute__((aligned(128)));
  static int8_t b[128] __attribute__((aligned(128)));
  static int8_t y[128] __attribute__((aligned(128)));
  for (int i = 0; i < 128; ++i) { a[i] = (int8_t)i; b[i] = 3; }
  *(HVX_Vector *)y =
    Q6_Vb_vadd_VbVb(*(const HVX_Vector *)a, *(const HVX_Vector *)b);
  qurt_hvx_unlock();
  for (int i = 0; i < 128; ++i)
    if (y[i] != (int8_t)(i + 3)) return 1;

  /* VTCM 1MB acquire/release */
  compute_res_attr_t rattr;
  HAP_compute_res_attr_init(&rattr);
  HAP_compute_res_attr_set_vtcm_param(&rattr, 1024 * 1024, 1);
  unsigned ctx_id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
  if (ctx_id == 0) { printf("SIM_TEST smoke vtcm acquire fail\n"); return 1; }
  void *vtcm = HAP_compute_res_attr_get_vtcm_ptr(&rattr);
  if (!vtcm) { HAP_compute_res_release(ctx_id); return 1; }
  memset(vtcm, 0xA5, 1024 * 1024);
  HAP_compute_res_release(ctx_id);

  printf("SIM_TEST smoke PASS\n");
  return 0;
}
```

- [ ] **Step 2: 테스트 main + util 작성**

```c
/* test/hexagon/sim/sim_test_main.c */
#include <stdio.h>
#include <string.h>
int test_smoke(void);
/* Task마다 여기 extern 선언 + 아래 항목 1줄 추가 */
static const struct { const char *name; int (*fn)(void); } tests[] = {
  {"smoke", test_smoke},
};
int main(int argc, char **argv) {
  if (argc < 2) { printf("SIM_TEST usage: <name>\n"); return 2; }
  for (unsigned i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i)
    if (!strcmp(argv[1], tests[i].name)) return tests[i].fn();
  printf("SIM_TEST unknown test: %s\n", argv[1]);
  return 2;
}
```

`sim_test_util.h`는 공통 설계의 `frand`/`cmp_f` 코드 그대로 (+`#include <math.h> <stdio.h> <stdint.h>`, include guard).

- [ ] **Step 3: 빌드/실행 스크립트 작성**

```bash
#!/bin/bash
# tools/hexagon/build_sim_test.sh
# Cross-builds the hexagon-sim test lib (v75; SDK 6.0.0.2 has QuRT sim
# images up to v75). Prereq: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
set -eu
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"
HEX_ARCH="${HEX_ARCH:-v75}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTP_DIR="$REPO/nntrainer/tensor/hexagon/htp"
SIM_DIR="$REPO/test/hexagon/sim"
OUT="$REPO/build_hexagon/sim"
mkdir -p "$OUT"

SRCS=("$SIM_DIR"/*.c)
# htp 소스는 Task 진행에 따라 자동 포함 (ops/, hvx/, worker_pool, htp_graph)
for f in "$HTP_DIR"/worker_pool.c "$HTP_DIR"/htp_graph.c \
         "$HTP_DIR"/ops/*.c "$HTP_DIR"/hvx/*.c; do
  [ -e "$f" ] && SRCS+=("$f")
done

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -G0 -O2 -g -fPIC -shared \
    -Wall -Werror -Wno-unused-function \
    -I "$HTP_DIR" -I "$HTP_DIR/ops" -I "$HTP_DIR/hvx" -I "$SIM_DIR" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    "${SRCS[@]}" \
    -o "$OUT/libnntr_sim_test.so"
echo "built: $OUT/libnntr_sim_test.so ($HEX_ARCH)"
```

```bash
#!/bin/bash
# tools/hexagon/run_sim_test.sh <test-name> [args...]
set -eu
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"
HEX_ARCH="${HEX_ARCH:-v75}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_hexagon/sim"
RUNMAIN="$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/hexagon_toolv87_${HEX_ARCH}/run_main_on_hexagon_sim"
cd "$OUT"
"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-sim" -m"$HEX_ARCH" \
    --simulated_returnval --usefs "$OUT" --nullptr=2 \
    "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/sdksim_bin/runelf.pbn" -- \
    "$RUNMAIN" ./libnntr_sim_test.so "$@"
```

- [ ] **Step 4: 검증 (사용자 실행) — 명령어 제시 후 정지**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.0.0.2/setup_sdk_env.source
./tools/hexagon/build_sim_test.sh
./tools/hexagon/run_sim_test.sh smoke; echo "exit=$?"
```

Expected: `SIM_TEST smoke hvx_units=<N>` (N≥1), `SIM_TEST smoke PASS`, `exit=0`.
주의: 이 Task는 하네스 자체가 산출물 — 시뮬레이터 호출 형식(runelf.pbn/run_main 인자)이 SDK 문서와 다르면 출력 기반으로 스크립트를 수정해 재검증한다.

- [ ] **Step 5: 승인 후 커밋**

```bash
git add tools/hexagon/build_sim_test.sh tools/hexagon/run_sim_test.sh test/hexagon/sim/
git commit  # [htp] Add the hexagon-sim test harness for HVX kernel work
```

---

### Task 2: 와이어 포맷 v2 + op-list 검증기

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/nntr_htp_common.h`
- Modify: `test/hexagon/test_oplist_header.c`

**Interfaces:**
- Produces: 공통 설계의 v2 구조체·enum·플래그 전부, 그리고
  `int nntr_htp_oplist_validate(const void *buf, uint32_t len, const uint32_t buf_size[NNTR_HTP_BUF_COUNT])` — 0 ok / 1 bad ptr·size / 2 bad magic / 3 version / 4 bad header field / 5 bad op. 기존 `nntr_htp_oplist_check`는 유지(헤더 접두만 검사, M1 경로).
- 이후 모든 DSP·호스트 코드가 이 헤더 하나를 와이어 ABI로 공유한다.

- [ ] **Step 1: 헤더 확장** — 공통 설계의 v2 코드 그대로 추가하고 `ABI_VERSION 2u`로 범프. 크기 체크 2종:

```c
typedef char nntr_htp_op_desc_size_check
  [(sizeof(struct nntr_htp_op_desc) == 64) ? 1 : -1];
/* oplist_header size check도 64로 갱신 */
```

- [ ] **Step 2: validate 구현 (plain C, 헤더 내 static inline)** — 규칙:
  - `len == 64 + n_ops*64`, magic/version, `head_dim == 128`, `hidden%64==0`, `ffn%64==0`, `n_heads%n_kv_heads==0`, `max_chunk>=1`, 전부 아니면 4.
  - op별: `kind < KIND_COUNT`, 각 ref의 `buf < BUF_COUNT`, `offset%128==0`, matmul류 `k%128==0`, 아니면 5.
  - 경계: op별 시맨틱스 표의 크기 공식으로 `offset+bytes <= buf_size[buf]` 확인 (`m==0`은 `max_chunk`로 평가). `buf_size[TOKENS] = max_chunk*4`, `buf_size[LOGITS] = vocab*4`는 호출자가 채워서 전달. ROPE는 out 검사 생략(제자리).
  - KV는 ref가 아니라 layer로 접근: ATTN op에 대해 `2*n_layers*n_kv_heads*max_seq*head_dim*2 <= buf_size[KV]` 확인.

```c
static inline uint32_t nntr__ref_bytes(uint32_t kind, int which /*0..3=in0..out*/,
                                       uint32_t m, uint32_t k, uint32_t n,
                                       const struct nntr_htp_oplist_header *h);
/* 표 그대로: EMBED in0=m*4, in1=(uint64로 곱해 체크) vocab*k, in2=vocab*4, out=m*k*2 ...
   MATMUL: in0=m*k*2, in1=n*k, in2=n*4, out=m*n*2 (LOGITS out=n*4) 등 */
```

(구현 시 곱은 uint64_t로 계산해 오버플로 방지.)

- [ ] **Step 3: x86 테스트 확장** — `test_oplist_header.c`에 추가: 유효한 2-op 리스트(RMSNORM+MATMUL) PASS, 그리고 각 실패 코드별 1케이스(길이 불일치→1, head_dim≠128→4, 미지 kind→5, 비정렬 offset→5, ACT 경계 초과→5, 버전≠2→3). 기존 v1 검사 assert는 유지(구조체 필드 접두 호환이므로 그대로 컴파일됨).

- [ ] **Step 4: 검증 (사용자 실행) — 명령어 제시 후 정지**

```bash
gcc -Wall -Werror -o /tmp/test_oplist test/hexagon/test_oplist_header.c && /tmp/test_oplist
```

Expected: `oplist header check: PASS`, exit 0.

- [ ] **Step 5: 승인 후 커밋** — `[htp] Extend the op-list wire format to v2 with op descriptors`

---

### Task 3: QuRT 워커 풀

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/worker_pool.{h,c}`
- Create: `test/hexagon/sim/test_pool.c` / Modify: `sim_test_main.c` (등록)

**Interfaces:**
- Produces:

```c
struct wp_pool;
typedef void (*wp_job_fn)(void *arg, int worker_id, int n_workers);
struct wp_pool *wp_create(int n_workers); /* <=0 → qurt_hvx_get_units() */
void wp_run(struct wp_pool *p, wp_job_fn fn, void *arg); /* 전 워커 완료 후 리턴 */
int wp_size(const struct wp_pool *p);
void wp_destroy(struct wp_pool *p);
```

- [ ] **Step 1: 테스트 작성** — `wp_create(0)` 후 `wp_size()`≥1; job이 각 워커 슬롯에 `worker_id` 기록 + 원자 카운터 증가; `wp_run` 2회 연속(배리어 재사용) 후 카운터==2×n, 슬롯 전부 기록됨; `wp_run`을 100회 반복해 행업 없음(배리어 누수 검출); destroy 후 리턴.

```c
/* test/hexagon/sim/test_pool.c 핵심 */
struct pjob { int slot[16]; unsigned count; };
static void pfn(void *arg, int wid, int nw) {
  struct pjob *j = arg;
  j->slot[wid] = wid + 1;
  __atomic_add_fetch(&j->count, 1, __ATOMIC_SEQ_CST);
  (void)nw;
}
int test_pool(void) {
  struct wp_pool *p = wp_create(0);
  if (!p) return 1;
  int n = wp_size(p);
  printf("SIM_TEST pool n_workers=%d\n", n);
  if (n < 1 || n > 16) return 1;
  struct pjob j; memset(&j, 0, sizeof(j));
  for (int it = 0; it < 100; ++it) wp_run(p, pfn, &j);
  if (j.count != (unsigned)(100 * n)) return 1;
  for (int w = 0; w < n; ++w) if (j.slot[w] != w + 1) return 1;
  wp_destroy(p);
  printf("SIM_TEST pool PASS\n");
  return 0;
}
```

- [ ] **Step 2: 구현** — ggml-hexagon 워커 풀 패턴 차용(구조만, 코드는 신규):
  - `wp_create`: n = 인자>0 ? 인자 : `qurt_hvx_get_units()`; 워커 스레드 n개 생성(스택 64KB, `qurt_thread_create`), 각 워커는 시작 시 `qurt_hvx_lock(QURT_HVX_MODE_128B)` 1회.
  - 디스패치: `qurt_barrier_t` 2개(start/done) 또는 세마포어 페어 — job 세대 카운터 방식: `wp_run`이 `{fn,arg,generation++}` 게시 후 start 세마포어 n회 up, done 세마포어 n회 down. 워커 루프: start down → fn(arg, wid, n) → done up. 종료는 `fn==NULL` 게시.
  - 호출 스레드는 워커에 포함되지 않음(단순성 우선; 워커 수 = HVX 유닛 수 원칙 유지).

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh pool; echo "exit=$?"
```

Expected: `SIM_TEST pool n_workers=<N>`, `SIM_TEST pool PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the QuRT worker pool with per-op barrier semantics`

---

### Task 4: ggml-hexagon 차용분 임포트 (exp 프리미티브 검증)

**Files:**
- Create (copy+adapt): `nntrainer/tensor/hexagon/htp/hvx/` ← `/home/j2z0/Project/llama.cpp/ggml/src/ggml-hexagon/htp/`의 `dma-queue.{c,h}`, `hvx-exp.h`, `hvx-base.h`, `hvx-floor.h`, `hvx-inverse.h`, `hex-utils.h`
- Modify: `NOTICE` (ggml MIT 항목 추가)
- Create: `test/hexagon/sim/test_exp.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Produces: `hvx_exp_f32`/벡터 단위 exp·inverse 프리미티브 (원본 API 유지), `dma-queue.h`의 DMA 큐 API (Task 7에서 사용).

- [ ] **Step 1: 파일 복사 + 헤더 표기** — 각 파일 최상단에 추가:

```c
// SPDX-License-Identifier: MIT
// Imported from llama.cpp ggml-hexagon (https://github.com/ggml-org/llama.cpp)
// Copyright (c) 2023-2026 The ggml authors. See NOTICE.
```

  - `dma-queue.h`의 `#include "hex-profile.h"`는 미차용이므로 해당 include 제거 + 프로파일 매크로 no-op 처리(수정 사실을 헤더 주석에 명기).
  - `NOTICE`에 "nntrainer/tensor/hexagon/htp/hvx/ 일부 파일은 llama.cpp ggml-hexagon(MIT)에서 가져옴 + MIT 전문 참조" 항목 추가.
  - 컴파일만 되면 됨: `build_sim_test.sh`가 `hvx/*.c`를 자동 포함.

- [ ] **Step 2: exp 테스트 작성** — fp32 4096개 `x ∈ [-20, 4]`(frand 스케일)로 `hvx_exp_f32(x, y, n)` vs `expf`, `cmp_f("exp", ...)` rtol 1e-3/atol 1e-6. (원본 API 이름이 다르면 테스트를 원본 이름에 맞춘다 — 차용분은 최소 수정 원칙.)

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh exp; echo "exit=$?"
```

Expected: `SIM_TEST exp PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Import HVX exp/inverse and dma-queue primitives from ggml-hexagon` (바디에 MIT 출처·수정 내역 명기)

---

### Task 5: per-token 동적 양자화 + int8 dot 프리미티브

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/hvx/hvx_quant.h`
- Create: `test/hexagon/sim/ref_ops.{h,c}` (quant/dot부터 시작, 이후 Task마다 추가)
- Create: `test/hexagon/sim/test_quant.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Produces:

```c
/* 스칼라 양자화: 비용 O(M·K)≈0.1%라 M2는 스칼라. 반환 = scale (absmax/127) */
static inline float htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k);
/* HVX int8 dot: k%128==0, w/x 128B 정렬 */
static inline int32_t hvx_dot_i8(const int8_t *w, const int8_t *x, uint32_t k);
```

- Produces (ref): `ref_quant_row(const __fp16*, int8_t*, uint32_t)` (동일 산식 → bit-exact), `ref_dot_i8(...)` (스칼라 int32).

- [ ] **Step 1: 구현**

```c
static inline float htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k) {
  float amax = 0.f;
  for (uint32_t i = 0; i < k; ++i) {
    float v = fabsf((float)x[i]);
    if (v > amax) amax = v;
  }
  float inv = amax > 0.f ? 127.f / amax : 0.f;
  for (uint32_t i = 0; i < k; ++i)
    q[i] = (int8_t)lrintf((float)x[i] * inv);
  return amax / 127.f;
}

static inline int32_t hvx_reduce_vw(HVX_Vector v) {
  for (int s = 4; s < 128; s <<= 1)
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, s));
  int32_t out[32] __attribute__((aligned(128)));
  *(HVX_Vector *)out = v;
  return out[0];
}

static inline int32_t hvx_dot_i8(const int8_t *w, const int8_t *x, uint32_t k) {
  const HVX_Vector *wv = (const HVX_Vector *)w, *xv = (const HVX_Vector *)x;
  HVX_Vector acc = Q6_V_vzero();
  for (uint32_t i = 0; i < k / 128; ++i)
    acc = Q6_Vw_vrmpyacc_VwVbVb(acc, wv[i], xv[i]); /* signed·signed, 128 MAC/명령 */
  return hvx_reduce_vw(acc);
}
```

- [ ] **Step 2: 테스트 작성** — K=1024: (a) 랜덤 fp16 행 → `htp_quant_row_fp16` vs `ref_quant_row` int8 전부 일치 + scale float 일치, (b) all-zero 행 → q 전부 0, scale 0 (0-나눗셈 가드), (c) 랜덤 int8 w·x 128B 정렬 버퍼 → `hvx_dot_i8` vs `ref_dot_i8` int32 완전 일치 (K=128, 1024 두 케이스).

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh quant; echo "exit=$?"
```

Expected: `SIM_TEST quant PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add per-token dynamic quantization and the vrmpy int8 dot`

---

### Task 6: MATMUL_W8A8 커널 (DDR 직독 기준 구현)

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/ops/htp_ops.h` (공통 설계의 ctx·헬퍼·op 선언 전부)
- Create: `nntrainer/tensor/hexagon/htp/ops/op_matmul.c`
- Modify: `test/hexagon/sim/ref_ops.{h,c}` — `ref_matmul_w8a8` 추가
- Create: `test/hexagon/sim/test_matmul.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Consumes: Task 3 `wp_*`, Task 5 `htp_quant_row_fp16`/`hvx_dot_i8`, Task 2 desc 구조.
- Produces: `void htp_op_matmul_w8a8(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);` — in0 fp16[m][k], in1 int8[n][k](행 stride=k), in2 fp32[n], out fp16[m][n]. ctx의 `xq`/`xq_scale` 스크래치 사용.
- Produces (ref): `void ref_matmul_w8a8(const __fp16 *x, const int8_t *w, const float *sw, __fp16 *y, uint32_t m, uint32_t k, uint32_t n);` — 내부에서 `ref_quant_row` 후 int32 dot, `y = (fp16)(dot * sw[j] * sx)`.

- [ ] **Step 1: 구현**

```c
struct mm_job { struct htp_exec_ctx *c; const struct nntr_htp_op_desc *d; uint32_t m; };

static void mm_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(c, d->out);
  /* 워커별 N-슬랩 [n0,n1) — 바깥 N, 안쪽 M (VTCM 타일 재사용 구조와 동형) */
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);
  for (uint32_t jn = n0; jn < n1; ++jn) {
    const int8_t *wrow = w + (size_t)jn * k;
    for (uint32_t t = 0; t < j->m; ++t) {
      int32_t acc = hvx_dot_i8(wrow, c->xq + (size_t)t * k, k);
      y[(size_t)t * n + jn] = (__fp16)((float)acc * sw[jn] * c->xq_scale[t]);
    }
  }
}

void htp_op_matmul_w8a8(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  const uint32_t m = htp_m(c, d), k = d->k;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  for (uint32_t t = 0; t < m; ++t) /* op 진입 시 1회 양자화 */
    c->xq_scale[t] = htp_quant_row_fp16(x + (size_t)t * k, c->xq + (size_t)t * k, k);
  struct mm_job j = {c, d, m};
  wp_run(c->pool, mm_worker, &j);
}
```

`htp_ops.h`에는 ctx 구조체(공통 설계), `htp_ref_ptr`/`htp_m` 헬퍼, op 함수 8종 선언, `extern const htp_op_fn htp_op_table[NNTR_HTP_OP_KIND_COUNT];` (표 정의는 Task 14의 htp_graph.c).

- [ ] **Step 2: 테스트 작성** — 테스트가 ctx를 손수 구성(ACT 힙 버퍼, `wp_create(0)`, xq 스크래치 malloc, 128B 정렬은 `memalign(128, ...)`). 케이스: (M=1, K=1024, N=256)과 (M=8, K=1024, N=256), 랜덤 fp16 X·int8 W·fp32 scale(0.001~0.02). `cmp_f` rtol 2e-3/atol 1e-3 (fp16으로 float 캐스팅 후 비교).

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh matmul; echo "exit=$?"
```

Expected: `SIM_TEST matmul PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the MATMUL_W8A8 HVX kernel with fused activation quant`

---

### Task 7: VTCM + DMA 더블 버퍼 가중치 스트리밍

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/ops/op_matmul.c`, `ops/htp_ops.h` (vtcm 필드 사용)
- Create: `test/hexagon/sim/test_matmul_dma.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Consumes: Task 4 `dma-queue` API, Task 6 커널.
- Produces: 동일 시그니처 `htp_op_matmul_w8a8` — `c->vtcm != NULL`이면 DMA 스트리밍 경로, NULL이면 Task 6 DDR 직독 경로(폴백 유지, 테스트가 두 경로 비교).

- [ ] **Step 1: 구현** — 워커별 VTCM 분할: `slab = c->vtcm + wid*(c->vtcm_size/nw)`, 절반씩 더블 버퍼(각 `HALF = (vtcm_size/nw - m*k(Xq) - 128)/2` 중 k의 배수 행 수로 절사, 최소 1행 보장 — 안 되면 DDR 경로 폴백):
  - 진입 시 Xq(m×k)를 워커 슬랩 앞부분에 복사(vrmpy 피연산자 VTCM 상주).
  - 루프: N-슬랩을 `rows_per_buf = HALF/k` 행 단위 청크로 나눔; `dma_queue_push(청크 c+1)` → `dma_queue_wait(청크 c)` → 청크 c의 행들로 Task 6과 동일한 dot·scale·store → 반복 (계산과 DMA 중첩).
  - scale 벡터(fp32[n])는 작으므로 DDR 직독 유지.
- [ ] **Step 2: 테스트 작성** — M=8, K=1024, N=3072(여러 청크 강제). ctx 두 벌: vtcm=NULL(기준 경로)과 vtcm=HAP_compute_res 4MB 확보. 검증 2중: (a) DMA 경로 vs `ref_matmul_w8a8` rtol 2e-3, (b) DMA 경로 vs DDR 경로 **비트 동일**(같은 정수 산식이므로 fp16 결과가 완전히 같아야 함 — 스트리밍이 数値에 영향 없음을 증명).
- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh matmul_dma; echo "exit=$?"
```

Expected: `SIM_TEST matmul_dma PASS`, `exit=0`. (시뮬레이터 user-DMA 미지원 증상이 나오면 출력 전문을 보고 판단 — 이 Task만 블로킹하고 이후 Task는 DDR 경로로 진행 가능하도록 폴백을 남겨둔다.)

- [ ] **Step 4: 승인 후 커밋** — `[htp] Stream matmul weights through VTCM with dma-queue double buffering`

---

### Task 8: RMSNORM 커널 (QK-Norm 겸용)

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/ops/op_rmsnorm.c`
- Create: `nntrainer/tensor/hexagon/htp/hvx/hvx_f16.h` (fp16 위드닝 누산·스칼라 곱 헬퍼)
- Modify: `ref_ops.{h,c}` — `ref_rmsnorm` 추가
- Create: `test/hexagon/sim/test_rmsnorm.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Produces: `void htp_op_rmsnorm(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);` — PER_HEAD면 행을 head_dim(128) 청크로 독립 정규화(gamma는 [head_dim] 공유), 아니면 행 전체(n) 정규화(gamma[n]). `eps = fp32(param0 비트)`.
- Produces (hvx_f16.h):

```c
/* Σ x_i² (fp32 위드닝): Q6_Wqf32_vmpy_VhfVhf 누산 → sf 변환 → 스칼라 합 */
static inline float hvx_sumsq_fp16(const __fp16 *x, uint32_t n);
/* y_i = (fp16)(x_i * r * g_i): r을 hf로 vsplat, Q6_Vqf16_vmpy 2단 */
static inline void hvx_scale_mul_fp16(const __fp16 *x, const __fp16 *g,
                                      float r, __fp16 *y, uint32_t n);
```

- Produces (ref): `ref_rmsnorm(x, gamma, y, m, n, chunk /*=n or head_dim*/, eps)` — fp32 스칼라.

- [ ] **Step 1: 구현** — 워커 분할은 행(t) 단위(`m*n/head 청크` 아님 — 행이 충분히 많고 단순함 우선). 청크마다 `r = 1/sqrtf(sumsq/chunk + eps)` 스칼라 계산 후 `hvx_scale_mul_fp16`. `hvx_sumsq_fp16`은 64-half 벡터 단위 `Q6_Wqf32_vmpy_VhfVhf` 누산(qf32 pair 2개 유지) → 루프 끝에 `Q6_Vsf_equals_Vqf32` 변환·128B 스토어 → 스칼라 32+32개 합산.
- [ ] **Step 2: 테스트 작성** — (a) 일반: M=4, n=1024, eps=1e-6, gamma 랜덤(0.5~1.5), (b) PER_HEAD: M=4, n=512(4헤드×128), gamma[128]. `cmp_f` rtol 5e-3/atol 2e-3.
- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh rmsnorm; echo "exit=$?"
```

Expected: `SIM_TEST rmsnorm PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the RMSNORM HVX kernel covering QK-Norm per-head mode`

---

### Task 9: ROPE 커널

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/ops/op_rope.c`
- Modify: `ref_ops.{h,c}` — `ref_rope` + `ref_rope_table_fill` 추가
- Create: `test/hexagon/sim/test_rope.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Produces: `void htp_op_rope(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);` — q(in0)·k(in1) 제자리 회전, 테이블(in2) fp16 `[max_seq][cos64‖sin64]`, 토큰 t는 테이블 행 `c->pos + t` 사용. head_dim=128 = hf 벡터 2개(x0=앞 64, x1=뒤 64):

```
out0 = x0*cos − x1*sin ; out1 = x1*cos + x0*sin   (rotate-half, 쌍 (i, i+64))
```

- Produces (ref): `ref_rope_table_fill(__fp16 *table, uint32_t max_seq, float theta /*=1e6*/)` — `inv_freq_i = theta^(-2i/128)`, `cos[p][i]=cosf(p*inv_freq_i)`; `ref_rope(x, table, m, n_heads, pos)` fp32 스칼라. (이 테이블 생성기는 M3 lowering의 레퍼런스도 겸한다.)

- [ ] **Step 1: 구현** — 워커 분할: (토큰 t × 헤드) 평탄 인덱스. 벡터 연산: `Q6_Vqf16_vmpy_VhfVhf` 4회 + `Q6_Vqf16_vsub/vadd_Vqf16Vqf16` + `Q6_Vhf_equals_Vqf16` 2회, q와 k 각각(k는 n_kv_heads개 헤드만).
- [ ] **Step 2: 테스트 작성** — cfg: n_heads=4, n_kv_heads=2, max_seq=64. 테이블은 `ref_rope_table_fill`로 생성해 WEIGHTS 버퍼에 배치. M=8, pos=5 (경계: pos+M=13<64). q·k 모두 `cmp_f` rtol 5e-3/atol 2e-3.
- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh rope; echo "exit=$?"
```

Expected: `SIM_TEST rope PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the ROPE HVX kernel using the precomputed cos/sin table`

---

### Task 10: ADD + SILU_MUL 커널

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/ops/op_eltwise.c`
- Modify: `ref_ops.{h,c}` — `ref_add`, `ref_silu_mul`
- Create: `test/hexagon/sim/test_eltwise.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Produces: `htp_op_add`, `htp_op_silu_mul` (시그니처 동일 `(ctx, desc)`).
  - ADD: 64-half 벡터 스트립으로 `Q6_Vqf16_vadd_VhfVhf` → `Q6_Vhf_equals_Vqf16`.
  - SILU_MUL: 스트립(64 half)마다 gate를 fp32 2벡터로 위드닝(`Q6_Wqf32_vmpy_VhfVhf(g, splat(1.0hf))` 또는 vcvt) → 차용 exp 프리미티브로 `e = exp(−g)` → `silu = g / (1+e)` (차용 inverse 프리미티브) → hf로 내림 → up과 `Q6_Vqf16_vmpy_VhfVhf`. 대형 스크래치 불필요(스트립 단위).
- Produces (ref): `ref_add(a,b,y,count)`, `ref_silu_mul(g,u,y,count)` — `silu(x)=x/(1+expf(-x))` fp32.

- [ ] **Step 1: 구현** — 워커 분할: 행(t) 단위, 행 내부 벡터 루프.
- [ ] **Step 2: 테스트 작성** — M=4, n=3072(ffn), 입력 범위 [-8, 8] (frand×8, silu 포화 구간 포함). ADD rtol 1e-3/atol 1e-4, SILU_MUL rtol 2e-2/atol 5e-3.
- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh eltwise; echo "exit=$?"
```

Expected: `SIM_TEST eltwise PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the ADD and SILU_MUL HVX kernels`

---

### Task 11: EMBED 커널

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/ops/op_embed.c`
- Modify: `ref_ops.{h,c}` — `ref_embed`
- Create: `test/hexagon/sim/test_embed.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Produces: `htp_op_embed` — token_ids(in0=TOKENS)로 int8 행 gather 후 `out[t][i] = (fp16)(w[row][i] * scale[row])`. 행 dequant는 스칼라(메모리 바운드 op, M≤128행 × hidden — 무시 가능 비용). 워커 분할: 토큰 t.
- Produces (ref): `ref_embed(tokens, w, scale, y, m, k)`.

- [ ] **Step 1: 구현 + 테스트** — vocab=512, k=256, M=8, 토큰 id는 {0, 1, 255, 511 포함}(경계 행). `cmp_f` rtol 1e-3/atol 1e-4.
- [ ] **Step 2: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh embed; echo "exit=$?"
```

Expected: `SIM_TEST embed PASS`, `exit=0`.

- [ ] **Step 3: 승인 후 커밋** — `[htp] Add the EMBED gather-dequant kernel`

---

### Task 12: ATTN 커널 (KV append + causal SDPA + GQA)

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/ops/op_attn.c`
- Modify: `ref_ops.{h,c}` — `ref_attn` (KV 배열을 직접 받아 append+SDPA)
- Create: `test/hexagon/sim/test_attn.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Consumes: Task 4 exp 프리미티브, `hvx_f16.h` 위드닝 헬퍼.
- Produces: `htp_op_attn` — 워커 분할: kv_head 단위(스펙). 절차(워커의 각 kv_head h):
  1. **KV append:** t∈[0,m): in1/in2의 h 헤드 행(128 half)을 KV의 `[layer][h][pos+t]`에 복사 (K, V 각각).
  2. **SDPA:** 이 h를 쓰는 q 헤드들(GQA 그룹 `n_heads/n_kv_heads`개)에 대해, 토큰 t마다 `L = pos+t+1`:
     - scores fp32[L] (워커 스크래치 `c->attn_scratch + wid*max_seq`): `s_j = scale · dot_fp32(q[t,h_q], K[h][j])` — dot은 hf 벡터 2개 `Q6_Wqf32_vmpy_VhfVhf` 누산 → 스칼라 reduce (`hvx_sumsq_fp16`와 동일 패턴의 일반화 `hvx_dot_fp16`을 hvx_f16.h에 추가).
     - 소프트맥스: 스칼라 max → `hvx_exp_f32(scores-max)`(차용) → 스칼라 합 → 역수 1회.
     - 출력: `out[t,h_q] = Σ_j p_j · V[h][j]` — qf32 pair 누산 2벡터, `p_j`는 hf splat, 끝에 `Q6_Vhf_equals_Wqf32`… (V 행이 hf 벡터 2개이므로 누산도 2쌍) → `(1/sum)` 곱해 저장.
- Produces (ref): `ref_attn(q, k_new, v_new, kv_k, kv_v, out, m, pos, cfg...)` — fp32 스칼라, 커널과 동일한 KV 레이아웃 사용.

- [ ] **Step 1: 구현** (위 절차 그대로; causal은 L 상한으로 자연 구현, 마스크 텐서 없음)
- [ ] **Step 2: 테스트 작성** — cfg: n_layers=2, n_heads=4, n_kv_heads=2, head_dim=128, max_seq=64. 시나리오: layer=1(비제로 KV 오프셋 검증), ① prefill M=8/pos=0 → ref와 out 비교, ② 이어서 decode M=1/pos=8 (①이 채운 KV 재사용) → 비교. KV 버퍼 자체도 ref와 비교(append 정확성). `cmp_f` rtol 2e-2/atol 5e-3.
- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh attn; echo "exit=$?"
```

Expected: `SIM_TEST attn PASS`, `exit=0`.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the fused ATTN kernel with KV append, causal SDPA and GQA`

---

### Task 13: MATMUL_LOGITS 커널

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/ops/op_matmul.c` — `htp_op_matmul_logits` 추가
- Modify: `ref_ops.{h,c}` — `ref_matmul_logits`
- Create: `test/hexagon/sim/test_logits.c` / Modify: `sim_test_main.c`

**Interfaces:**
- Consumes: Task 6의 양자화·dot 경로.
- Produces: `htp_op_matmul_logits` — in0의 **마지막 토큰 행**(`(n_tokens-1)*k*2` 오프셋)만 양자화(m=1 고정), out은 fp32[n] (LOGITS 버퍼, fp16 변환 없음): `logits[j] = dot·sw[j]·sx`. DMA 경로는 재사용하되 m=1 특성상 Task 6 DDR 경로로도 충분(코드 공유: 내부 공통 함수 `mm_run(c, d, x_row0, m, fp32_out)`로 두 op 통합).
- Produces (ref): `ref_matmul_logits(x_last, w, sw, out_f32, k, n)`.

- [ ] **Step 1: 구현 + 테스트** — K=1024, N=2048(vocab 축소판), M=4(마지막 행만 쓰는지 검증 위해 앞 행들 쓰레기값). `cmp_f` rtol 5e-3/atol 1e-2 (fp32 직접 비교).
- [ ] **Step 2: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh logits; echo "exit=$?"
```

Expected: `SIM_TEST logits PASS`, `exit=0`.

- [ ] **Step 3: 승인 후 커밋** — `[htp] Add the MATMUL_LOGITS kernel emitting fp32 last-token logits`

---

### Task 14: op-list 그래프 실행기 + 합성 2레이어 golden 테스트

**Files:**
- Create: `nntrainer/tensor/hexagon/htp/htp_graph.{h,c}`
- Create: `test/hexagon/sim/test_graph.c` / Modify: `sim_test_main.c`
- Modify: `ref_ops.{h,c}` — `ref_graph_forward` (op-list를 그대로 해석하는 스칼라 실행기; 단 활성화를 op 경계마다 `__fp16` 왕복시켜 fp16 상주를 모사)

**Interfaces:**
- Consumes: Task 2 validate, Task 3 pool, Task 6–13 op 함수 전부.
- Produces:

```c
struct htp_graph; /* opaque: 헤더 사본, ops 포인터, ctx, 스크래치 소유 */
int htp_graph_init(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                   uint8_t *weights, uint32_t wsize,
                   uint8_t *kv, uint32_t kvsize,
                   uint8_t *act, uint32_t actsize); /* validate + pool/스크래치/VTCM 확보, 0=ok */
int htp_graph_forward(struct htp_graph *g, const int32_t *tokens,
                      uint32_t n_tokens, uint32_t pos,
                      float *logits, uint32_t n_logits); /* 0=ok */
void htp_graph_destroy(struct htp_graph *g);
extern const htp_op_fn htp_op_table[NNTR_HTP_OP_KIND_COUNT]; /* 여기서 정의 */
```

- `htp_graph_forward`: ctx의 TOKENS/LOGITS 버퍼 포인터 갱신 → `n_tokens`/`pos` 설정 → `for (i<n_ops) htp_op_table[ops[i].kind](&ctx, &ops[i])` — 검사 없음(전부 init에서 완료), 디스패치는 switch/표 1회.
- `htp_graph_init` 스크래치 산정: `k_max = max(matmul류 k)` 스캔 → `xq[max_chunk][k_max]`, `xq_scale[max_chunk]`, `attn_scratch[n_workers][max_seq]` malloc; VTCM은 `HAP_compute_res_acquire` 시도, 실패 시 NULL(DDR 경로) — M2에서는 실패가 에러 아님.

- [ ] **Step 1: 테스트 작성** — tiny cfg: n_layers=2, hidden=256, n_heads=4, n_kv_heads=2, head_dim=128, ffn=512, vocab=512, max_seq=64, max_chunk=8.
  - 테스트 내 미니 lowering `build_tiny_oplist()`: 레이어당 16 op + EMBED/최종 RMSNORM/LOGITS = 35 op을 시맨틱스 표대로 손 배치(WEIGHTS: 랜덤 int8 proj 6종×2레이어 + gamma들 + rope 테이블 + tied lm_head/embed, ACT: 오프셋 계획 주석 포함, 전부 128B 정렬).
  - 실행: `htp_graph_init` → ① prefill tokens 8개/pos=0 → ② decode 1개/pos=8. 각 forward 후 logits를 `ref_graph_forward`(동일 oplist·버퍼 사본)와 `cmp_f` rtol 3e-2/atol 5e-2.
  - 추가 케이스: oplist의 head_dim을 64로 조작 → `htp_graph_init != 0` (validate 배선 확인).
- [ ] **Step 2: 구현** (위 인터페이스 그대로)
- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh graph; echo "exit=$?"
```

Expected: `SIM_TEST graph PASS`, `exit=0`. **이것이 M2의 최종 golden 게이트.**

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the op-list graph executor with the synthetic golden test`

---

### Task 15: FastRPC 글루 통합 + 디바이스 skel 빌드 확인

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/executor.c`
- Modify: `tools/hexagon/build_skel.sh` (소스 목록에 htp_graph.c, worker_pool.c, ops/*.c, hvx/*.c 추가)
- Modify: `docs/backend_guide/HEXAGON_HVX.md` (M2 결과 반영)

**Interfaces:**
- Consumes: Task 14 `htp_graph_*`.
- Produces: `nntr_htp_init` — `nntr_htp_oplist_check` 통과 후 `n_ops>0`이면 oplist를 DSP 힙에 복사하고 `htp_graph_init`(실패 코드→AEE_EBADPARM/AEE_EUNSUPPORTED 매핑); `n_ops==0`이면 기존 M1 더미 상태 유지. `nntr_htp_forward` — 그래프 초기화된 세션이면 `htp_graph_forward(tokens, token_idsLen, pos, logits, logitsLen)`, 아니면 기존 더미 패턴(**M1 디바이스 테스트 호환**). close에서 `htp_graph_destroy`.

- [ ] **Step 1: 구현** — 세션 구조체에 `struct htp_graph graph; int graph_ready;` 추가. init의 검증 실패는 실행 전 거부(스펙: 검증은 전부 init에).
- [ ] **Step 2: 문서 업데이트** — `docs/backend_guide/HEXAGON_HVX.md`에 M2 반영:
  - §3 Source layout: htp_graph/worker_pool/ops/hvx 디렉토리와 역할, ggml-hexagon 차용분(MIT) 명시
  - §4 Building: 시뮬레이터 테스트 절(`build_sim_test.sh`/`run_sim_test.sh <name>`, v75 타깃 이유, SDK 6.0.0.2 전제) 추가
  - §2 또는 신규 절: 와이어 포맷 v2 요약(64B 헤더 + 64B op_desc, ABI v2, op 9종 표 링크), forward의 그래프 경로 vs `n_ops==0` 더미 경로
  - §6 Current status: M2 완료 상태와 M3(호스트 lowering) 예고로 갱신
- [ ] **Step 3: 검증 (사용자 실행) — 정지** — 3종:

```bash
# (a) x86 헤더 테스트 여전히 PASS
gcc -Wall -Werror -o /tmp/test_oplist test/hexagon/test_oplist_header.c && /tmp/test_oplist
# (b) sim 전체 재실행 (회귀)
./tools/hexagon/build_sim_test.sh
for t in smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph; do
  ./tools/hexagon/run_sim_test.sh $t || { echo "FAIL $t"; break; }
done
# (c) 디바이스 skel 빌드 + M1 왕복 테스트 그대로 PASS (S25 Ultra, R3CY205ZMND)
./tools/hexagon/build_skel.sh
ANDROID_NDK=<ndk> ./tools/hexagon/build_host_test.sh
./tools/hexagon/run_device_test.sh R3CY205ZMND
```

Expected: (a) PASS, (b) 13개 전부 exit 0, (c) `RPC_TEST PASS` (더미 경로 유지 증명).

- [ ] **Step 4: 승인 후 커밋** — `[htp] Wire the graph executor into the FastRPC session` (바디에 "n_ops==0 dummy path kept for the M1 round-trip test" 명기; 문서 갱신 포함)

---

## Self-Review 결과

- **스펙 커버리지:** op 9종(예약 슬롯은 enum 여유로 반영) ✓ / 실행기(순차 루프+워커 풀+배리어) Task 3·14 ✓ / KV cache 레이아웃·포맷 필드(헤더 reserved로 확보) Task 2·12 ✓ / vrmpy matmul + 융합 양자화 Task 5·6 ✓ / DMA 더블 버퍼 Task 7 ✓ / M 안쪽 루프(디코드=프리필 단일 커널) Task 6 구조 ✓ / ggml 차용+라이선스 Task 4 ✓ / 시뮬레이터 커널 단위 + 합성 op-list golden Task 1~14 ✓
- **의도적 범위 결정 (스펙과의 차이, 구현자 주의):** ① 활성화 양자화는 스칼라 C (비용 ~0.1%, 스펙 QnA 근거) — "입력 양자화 HVX화"는 후속 최적화. ② EMBED dequant 스칼라(메모리 바운드). ③ sim 타깃 v75(SDK 6.0.0.2에 v79 sim 이미지 없음; HVX 128B 동일). ④ HAP_power 투표는 M1 스펙 영역이라 M2 범위 외(성능은 M5).
- **타입 일관성:** `wp_*`/`htp_exec_ctx`/`htp_ref_ptr`/`htp_m`/`hvx_dot_i8`/`htp_quant_row_fp16` 명칭이 Task 3·5·6·14에서 동일함을 확인. op 시그니처 `(struct htp_exec_ctx *, const struct nntr_htp_op_desc *)` 전 Task 공통.
- **알려진 리스크:** hexagon-clang qf16/qf32 인트린식 정확한 이름(Q6_*)은 SDK 헤더(`hexagon_protos.h`)와 대조해 조정 필요할 수 있음 — 산식·구조가 규범이고 인트린식 철자는 컴파일 에러로 즉시 드러남. Task 7의 sim user-DMA 지원이 유일한 외부 불확실성이며 DDR 폴백으로 격리됨.
