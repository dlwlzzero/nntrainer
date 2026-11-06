# Rules harvest: group "tensor" (cpu_backend, tensor_core)

Inputs: unit_cpu_backend.json (225 comments, 79 PRs), unit_tensor_core.json
(211 comments, 97 PRs), `git log refs/remotes/upstream/main --since=2026-03-01`
on both path sets (82 + 72 commits). Every code claim below was checked against
`refs/remotes/upstream/main` (640ac6e0e).

Layout recommendation (answers to the two questions):

- **Arch subdirs (arm/, x86/, fallback/, ggml_interface/, cblas_interface/,
  kai_interface/) do not need their own files.** One
  `nntrainer/tensor/cpu_backend/AGENTS.md` covers them. The most important
  rules are about the three backends together (declare in all three, the
  fallback path has to stay in step, armv7/x86/MSVC builds). A file per arch would
  split those rules up, so an agent working in `x86/` would never see the
  arm/fallback rules it needs.
- **Do not merge tensor_core and cpu_backend into one file.** The rules barely
  overlap. cpu_backend is about multi-arch dispatch, ISA flags and MSVC. tensor_core
  is about the ComputeOps call path, dtype switch sites, quantization axis and
  mmap. Use two nested files: `nntrainer/tensor/AGENTS.md` (tensor_core) and
  `nntrainer/tensor/cpu_backend/AGENTS.md`. Put one line in the cpu_backend file
  pointing to `../AGENTS.md`, because some agent tools read only the nearest
  file. TC1 (ComputeOps) touches both and belongs in the parent.
  Note that `nntrainer/tensor/AGENTS.md` would also be inherited by
  `cl_operations/` and `htp_backend/`. TC1 applies there too. TC2–TC4 do no harm.

---

## cpu_backend

### CB1. Dispatchers dispatch; kernels live in the per-ISA, per-precision file
- **Rule**: Keep `arm_compute_backend*.cpp` / `x86_compute_backend*.cpp` as thin
  dispatchers. Put NEON bodies in `arm/neon_impl.cpp` (fp16:
  `neon_impl_fp16.cpp`), AVX2 bodies in `x86/avx2_impl.cpp` (`avx2_impl_fp16.cpp`),
  and scalar bodies in `fallback/fallback_internal.cpp` (`fallback_internal_fp16.cpp`).
  Any function taking or returning `_FP16` goes in the `*_fp16.cpp` sibling,
  which is only compiled with `-Denable-fp16=true`. ggml-derived kernels go in
  `nntr_ggml_impl_{avx,neon,fallback}.cpp`. Reuse an existing `__fallback_*` or
  `neon_mathfun.h` helper (`exp_ps`, ...) instead of re-implementing it inline.
- **Why**: kernels inlined in the dispatcher or in a non-fp16 file break
  `enable-fp16=false` builds and get duplicated across backends.
- **Evidence**: jaemini-shin https://github.com/nntrainer/nntrainer/pull/4032#discussion_r3498130648 ,
  https://github.com/nntrainer/nntrainer/pull/4032#discussion_r3498538986 (5 move-this-file comments in one PR);
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/4268#discussion_r3800214482 ;
  skykongkong8 https://github.com/nntrainer/nntrainer/pull/3307#discussion_r2193684229 (exp_ps exists);
  commit 8bb5e1726 "[fix] Replace repeated code lines with fallback functions".
- **Strength**: 4 PRs + 1 commit; maintainers (jaemini-shin, skykongkong8).
  The layout is also stated in `cpu_backend/README.md`.
- **Verify**: `meson setup b0 -Denable-fp16=false && ninja -C b0` and
  `meson setup b1 -Denable-fp16=true && ninja -C b1`.

### CB2. A new backend function is declared and defined in all three backends
- **Rule**: `cpu_backend.h` only selects a header. The real declarations are
  repeated in `arm/arm_compute_backend.h`, `x86/x86_compute_backend.h` and
  `fallback/fallback.h`. Add a new function to all three with byte-identical
  signatures, give it a scalar `__fallback_*` in `fallback_internal.{h,cpp}`,
  and define it in all three `.cpp` files. `fallback.h/.cpp` are compiled only
  when the arch is neither arm nor x86 (`fallback/meson.build`), and **no CI
  job builds them**, so read them carefully. If Tensor code needs the op, also
  add it to `ComputeOps` (see TC1). A NEON-only kernel needs an x86 path
  before anything outside `arm/` calls it.
- **Why**: missing arms show up as link errors on another platform, as double
  definitions or missing returns in the never-built fallback, or as x86 build
  breaks.
- **Evidence**: myungjoo https://github.com/nntrainer/nntrainer/pull/3519#discussion_r2488962373 ("double definition? (not built in our github action scenarios?)"),
  myungjoo https://github.com/nntrainer/nntrainer/pull/3519#discussion_r2488946959 (missing return);
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3350#discussion_r2230723951 ;
  commits 377a35ea7 / c23210b58 (Q4_0-FP16 kernels existed only in the NEON file, x86 build broke);
  commit 4a56ca8da (one size formula had to be fixed in all three backend headers).
- **Strength**: 2 review PRs + 3 commits; myungjoo stated it outright.
- **Verify**: `meson test -C build unittest_nntrainer_cpu_backend unittest_nntrainer_fallback`
  (the second one tests `fallback_internal` on the host).

### CB3. Every SIMD fast path in a dispatcher has a fallback `else`
- **Rule**: When a dispatcher guards a SIMD kernel (`incX == 1 && incY == 1`,
  `head_dim == 64 || 128`, `N % 8 == 0`, a supported dtype ...), the `else`
  branch calls the matching `__fallback_*`, or throws explicitly. Never return
  silently. arm, x86 and fallback must give the same result for the same
  arguments.
- **Why**: a silent no-op leaves the output uninitialised while the caller
  carries on as if the kernel ran.
- **Evidence**: DonghakPark https://github.com/nntrainer/nntrainer/pull/3883#discussion_r3144960059 ;
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/3933#discussion_r3300840129 ("Every backend implementation should be worked in the same manner");
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/3926#discussion_r3278320069 ;
  djeong20 https://github.com/nntrainer/nntrainer/pull/3204#discussion_r2110664096 ;
  (Copilot bot raised the same thing for strided `scopy` in #3910. It is not counted.)
- **Strength**: 4 human PRs; maintainers.
- **Verify**: `unittest_nntrainer_cpu_backend` with a non-unit stride or tail size.

### CB4. arm/ code must also build for 32-bit armv7 (Tizen armv7l)
- **Rule**: meson defines `ARMV7=1` on 32-bit arm and builds `neon_impl.cpp`
  there. Reach aarch64-only intrinsics (`vfmaq_f32`, `vaddvq_f32`, `vmaxvq_f32`,
  `vdivq_f32`, ...) through the shims in `arm/armv7_neon.h` or the
  `VFMAQ_F32` macro, not through new `#ifdef ARMV7` blocks in `.cpp` files.
  The only existing `#ifndef ARMV7` guards in `arm_compute_backend.cpp` are
  around KleidiAI, because `kai_interface/` is not built on armv7. armv7 has no
  FP16 NEON kernels (it builds with `enable-fp16=false`), so do not declare fp16
  helpers for it.
- **Why**: aarch64 intrinsics break the Tizen armv7l GBS build.
- **Evidence**: skykongkong8 https://github.com/nntrainer/nntrainer/pull/3307#discussion_r2212552715 ("nntrainer does not enjoy adding #ifdefs like ARMV7 in the *.cpp");
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3965#discussion_r3359397900 (vfmaq vs vmlaq);
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/4000#discussion_r3526733864 ;
  commit 77478061a "[fix] Add armv7_neon header for Tizen GBS build".
- **Strength**: 3 PRs + 1 commit; maintainers.
- **Verify**: Tizen/GBS armv7l CI job (`gbs_build.yml`). Cannot be checked on an x86 host.

### CB5. ISA extensions beyond the baseline are opt-in and macro-guarded
- **Rule**: The aarch64 baseline is `armv8.2-a+dotprod+i8mm` (`+fp16` with
  enable-fp16). `fp16fml`, `sve`, `sve2`, `sme` and `sme2` are meson options that
  are off by default (`enable-fp16fml`, ...). Guard such code with the compiler's
  ACLE macro (`__ARM_FEATURE_FP16_FML`, `__ARM_FEATURE_SVE2`, ...) and keep a
  baseline path. Do not force the extension with `__attribute__((target(...)))`
  and do not invent a meson-side macro. On x86 the baseline is
  AVX2+FMA+F16C (`/arch:AVX2` on MSVC). A wider ISA (AVX-VNNI, AVX-512) goes in
  its own TU compiled with the extra flag, selected by cpuid at run time,
  following the `nntr_ggml_impl_avxvnni.cpp` pattern from ddda82583. Do not
  turn it on globally.
- **Why**: a target attribute or global flag produces a binary that SIGILLs on
  baseline devices, or fails to build for other `-march` values (armv9.2-a).
- **Evidence**: jaemini-shin https://github.com/nntrainer/nntrainer/pull/4045#discussion_r3527658399 (fp16fml);
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/4000#discussion_r3526576558 (attribute breaks armv9.2-a);
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3889#discussion_r3134552105 (compiler-predefined macro, not meson option);
  myungjoo https://github.com/nntrainer/nntrainer/pull/3543#discussion_r2492572226 (prefer run-time arch selection);
  commits 7c90cfc69 (SVE kernels broke the armv9.2-a build), ddda82583.
- **Strength**: 4 PRs + 2 commits; 3 maintainers.
- **Verify**: `meson setup b -Darm-arch=armv9.2-a` cross build. On x86:
  `-Denable-avxvnni=disabled` must still build.

### CB6. FP16 code must compile when `_FP16` is a wrapper class
- **Rule**: In shared or x86 code, write half precision as `_FP16` only. It is
  `__fp16`, `_Float16`, or the uint16-backed `nntrainer::Half` (MSVC,
  `-Dfp16-impl=wrapper`), see `api/ccapi/include/tensor_dim.h`. Use `__fp16` and
  `float16x8_t` only inside `arm/` files, and never `float16_t`. Do arithmetic
  and comparisons in `float` (convert, compute, convert back). Do not type-pun
  or bit-test `_FP16` values.
- **Why**: MSVC has no native half type, and x86 has no FP16 arithmetic.
  Native-half assumptions break the Windows build or silently miscompute.
- **Evidence**: skykongkong8 https://github.com/nntrainer/nntrainer/pull/2975#discussion_r1978427863 ;
  skykongkong8 https://github.com/nntrainer/nntrainer/pull/3307#discussion_r2193695293 (`float16_t` vs `__fp16`);
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/4032#discussion_r3498572619 (FP16 arithmetic on x86);
  piotrrak https://github.com/nntrainer/nntrainer/pull/3001#discussion_r1992260975 + myungjoo https://github.com/nntrainer/nntrainer/pull/3046#discussion_r2007050222 (fp16 bit punning);
  commit 304bab086 (Half wrapper / fp16-impl option).
- **Strength**: 4 PRs + 1 commit; maintainers.
- **Verify**: `meson setup bw -Denable-fp16=true -Dfp16-impl=wrapper && ninja -C bw && meson test -C bw unittest_nntrainer_tensor_fp16 unittest_half_fp16`.
  (The NEON FP16 tests `unittest_nntrainer_cpu_backend_fp16`, `unittest_neon_gelu`
  and `unittest_nntrainer_tensor_neon_fp16` exist only in `test/jni/Android.mk`,
  not in meson.)

### CB7. MSVC/Windows-specific pitfalls in kernel code
- **Rule**: Guard GCC/Clang-only spellings (`__attribute__((target/always_inline))`,
  `#pragma GCC unroll`) with `__GNUC__`/`__clang__`, and give an empty `#else`
  for attribute macros. Keep `__restrict` off any declaration or specialization
  whose signature must match a primary template, because MSVC mangles it.
  Headers that can pull in `<windows.h>` keep the `NOMINMAX`/`#undef max`
  guard already used in `arm_compute_backend.h`, `x86_compute_backend.h` and
  `fallback.h`. Aligned allocation goes through `MemAllocator`
  (`_aligned_malloc` on Windows), not raw `aligned_alloc`/`posix_memalign`.
- **Why**: each of these has broken a Windows build (syntax error, LNK2019, min/max
  macro clash, missing POSIX API).
- **Evidence**: myungjoo https://github.com/nntrainer/nntrainer/pull/3558#discussion_r2517011253 + songgot https://github.com/nntrainer/nntrainer/pull/3558#discussion_r2535869562 (NOMINMAX);
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/3967#discussion_r3377513874 (non-POSIX aligned alloc);
  Seunghui98 https://github.com/nntrainer/nntrainer/pull/4086#discussion_r3612610171 (attribute macro without `#else`);
  commits a398be553 (target attr broke MSVC), e7cd886db (`__restrict` LNK2019 on MSVC).
- **Strength**: 3 PRs + 2 commits; myungjoo, jaemini-shin, jijoongmoon (commit).
- **Verify**: Windows Meson / Windows-on-ARM CI jobs. No local check on Linux.

### CB8. BLAS calls stay inside cblas_interface and behind USE_BLAS
- **Rule**: Call OpenBLAS only from `cblas_interface/cblas_interface.cpp` and
  only through `__cblas_*` wrappers. In the dispatchers, wrap each use in
  `#ifdef USE_BLAS ... #else <fallback/neon/avx2> #endif`, and include
  `cblas_interface.h` only under `USE_BLAS`. Do not call
  `openblas_set_num_threads` from a per-op wrapper, because it is process-global
  state. Thread count is set once via `__openblas_set_num_threads`.
- **Why**: `-Denable-blas=false` drops `cblas_interface/` from the include path
  and the build fails. A per-op thread-count call leaks into every later BLAS call.
- **Evidence**: djeong20 https://github.com/nntrainer/nntrainer/pull/2930#discussion_r1974866848 (enable-blas=false fatal error);
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3342#discussion_r2633174265 ;
  skykongkong8 https://github.com/nntrainer/nntrainer/pull/3342#discussion_r2218466558 ;
  myungjoo https://github.com/nntrainer/nntrainer/pull/3626#discussion_r2652761067 (side effect of set-num-threads).
- **Strength**: 3 PRs; 4 maintainers.
- **Verify**: `meson setup nb -Denable-blas=false && ninja -C nb && meson test -C nb unittest_nntrainer_cpu_backend`.

---

## tensor_core

### TC1. Tensor types reach kernels only through ComputeOps
- **Rule**: In `float_tensor.cpp`, `half_tensor.cpp`, `char_tensor.cpp` and
  similar files, call kernels as `getOps()->op(...)`. Never call SIMD code or
  arch-specific helpers directly, and never add `#ifdef __AVX2__` /
  `__ARM_NEON` / `__aarch64__` / `ARMV7`. float_tensor.cpp still has two legacy
  blocks; do not add more. A new kernel gets a cpu_backend function (CB2), a
  virtual in `cpu_backend/compute_ops.h` whose default body in `compute_ops.cpp`
  throws, and a forwarding override in `cpu_ops_table.h`/`.cpp`. GPU/NPU
  backends then override it or fall back cleanly.
- **Why**: direct calls bypass the per-tensor backend (OpenCL/CUDA/HTP), and arch
  ifdefs outside cpu_backend headers are what maintainers have asked to remove
  for two years.
- **Evidence**: myungjoo https://github.com/nntrainer/nntrainer/pull/2832#discussion_r1893654807 ;
  skykongkong8 https://github.com/nntrainer/nntrainer/pull/3575#discussion_r2554578645 ("nntrainer doesn't really want to add more hw-arch-specific flags like #ifdef __AVX2__ outside of cpu_backend.h");
  skykongkong8 https://github.com/nntrainer/nntrainer/pull/2850#discussion_r1903384960 ;
  commits c48db97e8 ("route Tensor ops via per-vendor ContextData"), 7d8a06a48 ("Take the scalar Q4_0 GEMM from the CPU ops table"), 599d09f45.
- **Strength**: 3 review PRs + 3 jijoongmoon commits; maintainers stated it.
- **Verify**: `meson test -C build unittest_compute_ops_dispatch unittest_nntrainer_tensor`.

### TC2. A new dtype (or a new weight dtype for an op) covers every switch site, both precisions
- **Rule**: Grep for an existing sibling dtype (e.g. `Q6_K`, `QS4CX`) and add an
  arm to each switch it appears in. At minimum: the `Tensor` ctor from
  `TensorBase`, `operator=` and `operator==`/`itensorCompare` in `tensor.cpp`;
  `FloatTensor::dot` **and** `HalfTensor::dot`; the `Tdatatype` enum in
  `api/ccapi/include/tensor_dim.h`; `dtypeToStr` in `tensor_api_graph.cpp`;
  `model_tensor_type` parsing. Set the right `Tdatatype` in the new class's ctor.
  An unsupported combination throws (`NNTR_THROW_IF`). It never falls through an
  empty `break`, and it never ignores `trans`/`beta`.
- **Why**: missed arms give a null `itensor_`, false inequality, garbage output on
  the FP16 path only, or "unsupported datatype" in fp16 builds.
- **Evidence**: jaemini-shin https://github.com/nntrainer/nntrainer/pull/3730#discussion_r2796749683 ;
  skykongkong8/EunjuYang https://github.com/nntrainer/nntrainer/pull/2866#discussion_r1912577197 (wrong Tdatatype copied);
  EunjuYang https://github.com/nntrainer/nntrainer/pull/3132#discussion_r2059401233 ;
  commits be28c29e4 (QS4CX missing from clone/assign/compare), caa2aa12c (HalfTensor::dot Q4_0 was an empty `break`), fae2757f1, d0f01acb0 (beta dropped, trans ignored).
- **Strength**: 3 review PRs + 4 commits; jijoongmoon (commits), jaemini-shin, EunjuYang, skykongkong8.
- **Verify**: `meson test -C build unittest_nntrainer_tensor unittest_nntrainer_tensor_types`, plus
  `unittest_nntrainer_tensor_fp16` in a `-Denable-fp16=true` build.

### TC3. Quantized weights: scales on width(); payload layout only via existing helpers
- **Rule**: QS4CX and Q4_0 carry one scale per `width()` (N = output channel). A
  weight the layer indexes by row (embedding LUT) has its scales on `height()`
  and must not be quantized on the width axis unless it is transposed first.
  Read and write nibble payloads and packing parameters only through the existing
  helpers (`Int4Utils`, `Q4_0Utils`, `kleidiai_interface.h` and `kai/pack`). Do not
  hard-code or re-derive the layout, and do not hard-code KleidiAI packing
  parameters, which differ between the fp32 and fp16 kernels.
- **Why**: the record does not store the quantization axis, so a wrong axis or
  a second spelling of the layout comes back as wrong tokens, not an error.
- **Evidence**: jaemini-shin https://github.com/nntrainer/nntrainer/pull/4305#discussion_r3891880036 (embedding LUT has height() scales),
  https://github.com/nntrainer/nntrainer/pull/4305#discussion_r3891897119 (packing params via interface),
  https://github.com/nntrainer/nntrainer/pull/4305#discussion_r3891912377 (helpers already in kai/pack);
  commits facdfe5ef ("Refuse to quantize a weight the layer indexes by row"), 4ba86c201 ("nntrainer uses width axis as quantized axis"), d0f01acb0 (layout spelled a third time, wrong).
- **Strength**: 1 review PR (3 comments) + 3 commits by 2 maintainers (jijoongmoon, jaemini-shin).
- **Verify**: `meson test -C build unittest_nntrainer_quantizer unittest_safetensors_quantize`.

### TC4. mmap / FSU paths: runtime page size, Windows branch, checked failure
- **Rule**: In `swap_device.cpp` and in `Tensor::activate`/`deactivate`, align
  file offsets to the runtime page size (`sysconf(_SC_PAGESIZE)`, or
  `GetSystemInfo` under `_WIN32`), never to 4 KiB. Keep the existing `_WIN32`
  branches when editing. Check for `MAP_FAILED` and `fd == -1` and throw before
  storing the pointer.
- **Why**: 16 KiB-page Android returns EINVAL, and an unchecked `MAP_FAILED`
  segfaults later.
- **Evidence**: DonghakPark https://github.com/nntrainer/nntrainer/pull/3103#discussion_r2034290216 ;
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/2904#discussion_r1945622116 (Windows code removed);
  dkjung https://github.com/nntrainer/nntrainer/pull/3004#discussion_r1992825095 + piotrrak https://github.com/nntrainer/nntrainer/pull/3004#discussion_r1992601642 ;
  commits eac535109 (runtime page size), 5268d7efe (fd=-1 → MAP_FAILED segfault).
- **Strength**: 3 review PRs + 2 commits; maintainers.
- **Verify**: `meson test -C build unittest_nntrainer_tensor` (it has the virtual-tensor
  lifecycle test with an unaligned offset).

### TC5 (weaker, keep only if room). New TensorBase ops get a throwing default
- **Rule**: Add a new op as a non-pure virtual in `tensor_base.{h,cpp}` that throws
  "... is not supported in tensor data type <X>", as the 38 existing defaults
  do. Override it only where the op means something (e.g. not argmax on BCQ).
  Build it from existing ops where you can (`neg` = `multiply(-1)`). Do not add
  parameters to the public `Tensor` API without a caller.
- **Why**: otherwise every tensor class has to implement meaningless ops, and
  the Tensor API drifts.
- **Evidence**: djeong20 https://github.com/nntrainer/nntrainer/pull/3036#discussion_r2004666969 ;
  djeong20 https://github.com/nntrainer/nntrainer/pull/2862#discussion_r1942641728 ;
  EunjuYang https://github.com/nntrainer/nntrainer/pull/3304#discussion_r2194602249 .
- **Strength**: 3 PRs; 2 maintainers. All comments are older than the review window's bulk; the code pattern still holds.
- **Verify**: none specific.

---

## Global (belongs in root AGENTS.md, not a unit file)

### G1. Satisfy the static.check doxygen and newline checkers on every changed C/C++ file
- **Rule**: `static.check.yml` runs `doxygen-tag.sh <changed> 1` (advanced mode)
  on changed files. The first comment block must contain `@file @brief @author
  @bug`. Every multi-line `/* */` comment must start with `/**`, so write
  non-doxygen prose as `//` lines. Doxygen tags are lowercase. Reviewers
  additionally require `@brief` and one `@param` per parameter on every header
  declaration, and the `@param` must be updated when the signature changes.
  Files end with a newline (`newline.sh`). `misspelling.sh` runs too.
- **Why**: most frequent review nit in both units, and CI fails on it.
- **Evidence**: jaemini-shin https://github.com/nntrainer/nntrainer/pull/4086#discussion_r3612899647 ;
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/3711#discussion_r2715359257 ;
  songgot https://github.com/nntrainer/nntrainer/pull/3626#discussion_r2644489019 ;
  EunjuYang https://github.com/nntrainer/nntrainer/pull/3189#discussion_r2106412593 ;
  baek2sm https://github.com/nntrainer/nntrainer/pull/3116#discussion_r2044263677 (newline);
  commits 1986b1d1b, c252b6608, fb83d30d1.
- **Strength**: >= 15 PRs across both units (also #2930, #3236, #3307, #3628, #3740, #4105, #2838, #2846, #2906, #2927, #3132, #3889).
- **Verify**: `git diff --name-only <base> > /tmp/f && bash .github/workflows/static.check.scripts/doxygen-tag.sh /tmp/f 1 && bash .github/workflows/static.check.scripts/newline.sh /tmp/f`.

### G2. File header metadata: @date is the creation date; do not touch headers you did not change
- **Rule**: Do not bump `@date` or add yourself as `@author` for small edits.
  Add an author only for a new file or a significant rewrite. Leave the header
  of a file with no code change alone. Code moved into a new file keeps its
  original copyright/author, and the new file gets an accurate `@date`.
  Use real names and emails, never placeholders.
- **Why**: header churn makes noisy diffs and misattributes authorship.
- **Evidence**: jijoongmoon https://github.com/nntrainer/nntrainer/pull/3739#discussion_r2840783657 ("We usually use this date to create data, not update date"),
  https://github.com/nntrainer/nntrainer/pull/3739#discussion_r2840798595 ;
  Seunghui98 https://github.com/nntrainer/nntrainer/pull/4086#discussion_r3612612225 ;
  dkjung https://github.com/nntrainer/nntrainer/pull/3133#discussion_r2057137213 ;
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3591#discussion_r2570737407 .
- **Strength**: 4 PRs; jijoongmoon stated it as convention.
- **Verify**: none.

### G3. An installed header must match both package manifests
- **Rule**: Every header added to an installed list (`tensor_headers`,
  `cpu_backend_headers`, `nntrainer_headers`, ...) must also be listed in
  `packaging/nntrainer.spec` and `debian/nntrainer-dev.install`, in the same
  commit. Removing or renaming a header removes it from both. Keep
  internal-only headers off the install list.
- **Why**: a mismatch either way fails the Debian (`dh_missing --fail-missing`) and Tizen RPM
  (unpackaged file) jobs.
- **Evidence**: commits 2c199b8b0, c7d30ddbd (residency_policy.h), 83d77cf6e (removed int4_tensor.h), 9963563cd, 23a0b0c17, e99dc33f0. All are jijoongmoon commits from 2026-03 onward.
- **Strength**: 6 fix commits by a maintainer (no review comments, because it is caught by CI after merge).
- **Verify**: `grep -n '<header>' packaging/nntrainer.spec debian/nntrainer-dev.install`; Pdebuild and GBS CI jobs.

### G4. Cast before multiplying sizes (CodeQL `cpp/integer-multiplication-cast-to-long`)
- **Rule**: When a product of `unsigned int`/`uint32_t` dimensions feeds a `size_t`
  allocation, offset or loop bound, cast one operand to `size_t` before the multiply.
- **Why**: a 32-bit wrap on large models. CodeQL is a PR gate.
- **Evidence**: commits a1544f85b (16 sites incl. float_tensor.cpp, int4_utils.cpp; jijoongmoon), 7cdfffa5b (ggml_interface_omp.cpp; jaemini-shin).
- **Strength**: 2 commits by 2 maintainers.
- **Verify**: CodeQL Advanced CI job.

### G5. Include guards, not `#pragma once`
- **Rule**: Use `#ifndef __NAME_H__ / #define __NAME_H__` guards. 259 headers under
  `nntrainer/` use them, 6 use `#pragma once`.
- **Why**: consistency. The Tizen armv7 build has hit duplicate-definition trouble with `pragma once` (77478061a).
- **Evidence**: jijoongmoon https://github.com/nntrainer/nntrainer/pull/3739#discussion_r2840834361 ; commit 77478061a.
- **Strength**: 1 PR + 1 commit; maintainer stated it as convention.
- **Verify**: none.

### G6. meson: choose flags by compiler id, arch by cpu_family, OS by system
- **Rule**: Condition compiler-specific flags (`/MD`, `/arch`, `-m...`) on
  `cxx.get_id()`, CPU-specific sources on `host_machine.cpu_family()`, and OS
  differences on `host_machine.system()`. Do not mix them up. The one deliberate
  exception is `platform == 'android'` meaning arm, because the NDK hides the arch
  from meson. Copy that existing condition; do not invent new mixes.
- **Why**: `/MD` breaks Linux, and android-or-aarch64 conditions misfire on armv7-linux or x64-android.
- **Evidence**: myungjoo https://github.com/nntrainer/nntrainer/pull/3567#discussion_r2540807499 ,
  https://github.com/nntrainer/nntrainer/pull/3568#discussion_r2545122048 ,
  https://github.com/nntrainer/nntrainer/pull/3615#discussion_r2635115114 .
- **Strength**: 3 PRs; myungjoo.
- **Verify**: none locally beyond configuring for two targets.

---

## Rejected

- OpenMP loop index must be signed / partition remainder (skykongkong8 #3186, #3189, lhs8928 #3231): obsolete. OpenMP was replaced by `ThreadManager::parallel_for` (5a3123805), and no `#pragma omp` is left under nntrainer/tensor.
- No `thread_local` scratch in kernels (jaemini-shin #3967, #4087): upstream code contradicts it (5 `thread_local` buffers in ggml_interface and x86 backends). One reviewer.
- s/d/h BLAS name prefixes (myungjoo #3626 snrm2→dnrm2, djeong20 #2930 hdot): the code overloads the `s`-prefixed names on `_FP16` instead.
- fp16 bit punning via memcpy vs union (piotrrak #3001, myungjoo #3046): the reviewers disagree. Folded into CB6 as "do not bit-test `_FP16`".
- Float `==` vs `std::fpclassify(...) == FP_ZERO` (djeong20 #2862, lhs8928 #3052, myungjoo #3284): generic advice, old, and the idioms suggested are inconsistent.
- Cite the source or paper for ported kernels and magic constants (#3153, #3154, #3163, #3798, #3967): recurring, but generic "document your code".
- Unused variables, dead code, typos, `switch` vs `if`, redundant lines: generic nits. misspelling.sh is covered by G1.
- Unrelated changes in a PR (jijoongmoon #3889, dlwlzzero #4162): already in root ("one commit = one topic").
- Common `windows_compat.h` header (songgot #3558): suggested, never adopted. Folded into CB7 as "keep the existing guard".
- Include order std-before-project (djeong20 #2930): single PR.
- "FP16 only where it pays" (jaemini-shin #3926, #3933): a design judgement per kernel, not an agent rule.
- Flash-attention / ggml interface divergence notes (#4044, #4045, #3350): roadmap comments, not rules.

## Unit verdict

- **cpu_backend**: file worthwhile, **8 rules** (CB1–CB8). One file at
  `nntrainer/tensor/cpu_backend/AGENTS.md` covers all arch subdirs.
- **tensor_core**: file worthwhile, **4 rules** (TC1–TC4), plus TC5 if a fifth is
  wanted. Use a separate file at `nntrainer/tensor/AGENTS.md`, not merged with cpu_backend.
- **Global**: 6 candidates (G1–G4 strong; G5, G6 maintainer-stated but thinner).
