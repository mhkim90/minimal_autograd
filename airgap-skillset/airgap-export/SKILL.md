---
name: airgap-export
description: >
  Use INSIDE a closed/secure network when a problem, or an
  architecture/approach question, must be sent to an external LLM for
  diagnosis or a design opinion. Rewrites real source into a de-identified
  case package — markdown report, or a self-contained buildable repro in
  the source's own language — and applies the returned fix or
  recommendation back onto real code. Triggers: "폐쇄망", "반출",
  "sanitize", "case package", "/airgap".
---

# airgap-export (INSIDE only)

- OUT: real code + symptom → `case_<id>.out.md`, human-reviewed, carried out by hand
- IN: `case_<id>.in.md` from outside → mapped back → applied

`.airgap/case_<id>.map` links the two. It NEVER leaves.

## Hard rules

1. Nothing crosses out until a human reads the final package end to end.
2. Export the failure mechanism, never the domain purpose.
3. Algorithm bodies may cross once names are masked (see Naming) — most
   algorithms are not the secret; the data they process, and a short list
   of domain-specific algorithms, are (see Domain sensitivity). For an
   algorithm on that list, or whenever it's unclear, replace real math with
   a dummy payload of identical shape, dtype, layout, and control flow
   instead. If a required dummy payload does not reproduce the bug, the
   payload is not what is broken — narrow further before exporting.
4. Never write a semantic description of the code. The largest leak channel
   is restating a stripped comment as helpful prose ("computes the
   correction field for..."). Describe only what the toolchain printed,
   what the numbers are, what the control flow does.
5. Comments and docstrings are deleted, not translated. Package is ASCII
   English only.
6. Try first, export second. Attempt diagnosis locally; export only after a
   stated hypothesis was tested and failed. Record the failures — they are
   the most valuable part of the package.
7. Log every export (case id, files, date).

## Domain sensitivity

Default is masking, not replacement: alias function/variable/type names
(see Naming) and let control flow, math, and structure cross as-is. Fall
back to Hard rule 3's dummy-payload replacement only for an algorithm on
the sensitive list below, or a close variant of one.

Sensitive list (extend per site; keep the extended list itself inside):

- lithography
- OPC (optical proximity correction)
- EUV process algorithms

Add these same terms to `terms.txt` so the leak scan also catches a missed
case. If it's unclear whether an algorithm belongs on this list, stop and
ask before drafting the export — do not decide silently.

## Naming

Sequential and meaningless, per case: `fn1 fn2` (functions), `v1 v2`
(variables), `T1` (types), `P1` (paths), `K1` (kernels). Never semantic —
`applyMaskCorrection` → `fn1`, not `applyCorrection`. If a name still reads
like a hint, it is wrong.

Record every substitution in `.airgap/case_<id>.map` as `alias<TAB>real`,
one per line, before writing the package. Without it the returned fix
cannot be applied.

## Language

Follow the source. C/C++/CUDA → strip `//` and `/* */`. Python → strip `#`
lines and `"""` / `'''` docstrings. Anything else → strip that language's
comment syntax by hand. A repro is written in the same language as the
original, unless the bug is language-independent.

## Leak scan before carrying out

The model does not approve its own output. Run this against the finished
package:

```bash
C=case_<id>.out.md
sed 's/#.*//' terms.txt | tr -s ' \t' '\n\n' | sed '/^$/d' > /tmp/dl   # one term per line
grep -niFf /tmp/dl "$C"                            # denylist terms (case-insensitive substring)
grep -nP '[^\x00-\x7f]' "$C"                      # native-language text
grep -nE '(^|[ ("])(/|[A-Za-z]:\\)[^ "]+' "$C"     # absolute paths
grep -nE '[0-9]+\.[0-9]+' "$C"                     # physical constants
```

The `tr` step matters: `terms.txt` holds several terms per line, and
`grep -f` would otherwise treat a whole line as one pattern and match
nothing.

Every hit is reviewed by a human, not auto-fixed. Keep numbers that are
pure machine facts (sizes, strides, timings); drop or round numbers that
carry physical meaning.

Extend `terms.txt` with site, product, team, tool, and customer names. It
stays inside.

## OUT format

```
# CASE <id> / OUT
mode: report | repro

CLASS: build | runtime-error | numerical-mismatch | perf | memory | race | hang | design-review

ENVIRONMENT
  compiler / CUDA / driver, GPU arch (sm_XX) and count, libs + versions, build flags

OBSERVED
  Exact error text or measurement, verbatim, paths aliased. No paraphrase.

EXPECTED
  What should have happened, and how it is measured.

SHAPE FACTS
  Shapes, dtypes, strides, alignment, smem bytes, launch config, occupancy, stream/graph
  structure, timings. Numbers only, no meaning attached.

CODE
  Aliased snippet, or self-contained repro. State which.
  Repro rules: single file; only C++17 / CUDA / ArrayFire / Flashlight / Eigen / OpenMP / stdlib
  (or plain Python + numpy); no project headers; device code behind #ifdef __CUDACC__; a CPU
  reference path and a main() printing PASS/FAIL, because outside has no GPU and likely no nvcc;
  intended compile line as the first comment.

TRIED
  Each attempt → what changed → result, including my own rejected hypotheses.

CONSTRAINTS
  No real names, paths, or purpose will ever be supplied — do not ask.
  Tools I have inside: <ncu | nsys | compute-sanitizer | cuda-gdb | printf | host-debugger | none>
  Rebuild: <free | restricted | none>   Rerun: <on-demand | batch | one-shot>
  Round trips are manual and expensive. The answer must be self-contained.

ASK
  One precise question.
```

For `CLASS: design-review`, fields shift meaning: ENVIRONMENT may be N/A.
OBSERVED is the approach as currently designed (structure, not a symptom).
EXPECTED is the property, guarantee, or risk the approach must satisfy, and
how it would be checked. TRIED is the alternative approaches already
considered internally and why each was rejected. ASK is the one specific
judgment call you want an opinion on, not "review this."

## Worked example

```
# CASE a7 / OUT
mode: repro

CLASS: numerical-mismatch

ENVIRONMENT
  nvcc 12.4, driver 550.54, sm_90 x1, g++ 11.4, -O3 -lineinfo

OBSERVED
  fn1 output diverges from the CPU reference at 4096 of 16777216 elements.
  max abs diff 3.05e-05, all divergent indices satisfy (i % 128) >= 120.
  Deterministic across runs. Divergence disappears at blockDim 64.

EXPECTED
  Match within 1e-06 for all elements, as it does for blockDim 64.

SHAPE FACTS
  in  v1: 4096x4096 float32, row-major, 512B-aligned
  out v2: same
  launch <<<32768, 128>>>, smem 8320B, 40 regs, occupancy 0.5
  tail of each 128-wide row segment is the divergent region

CODE (reconstructed repro, dummy payload — real arithmetic replaced by a
      shape-identical reduction that reproduces the divergence)
  // nvcc -std=c++17 -arch=sm_90 -O3 repro.cu -o repro   (CPU path: g++ -std=c++17)
  [ ~60 lines: K1 kernel + CPU reference + main() printing PASS/FAIL ]

TRIED
  - Suspected FTZ/contraction: rebuilt with -fmad=false → unchanged. Rejected.
  - Suspected race on smem tail: compute-sanitizer --tool racecheck clean. Rejected.
  - blockDim 64 clean, 128 and 256 dirty → scales with block width, not grid size.

CONSTRAINTS
  Tools inside: ncu, compute-sanitizer, printf. Rebuild: free. Rerun: on-demand.

ASK
  Why does the last 8 lanes of each 128-wide segment diverge, given racecheck is clean?
```

## Worked example (design-review)

```
# CASE b3 / OUT
mode: report

CLASS: design-review

ENVIRONMENT
  N/A

OBSERVED
  fn1 processes v1 (a queue of T1) with a single consumer thread pinned to
  one core; producers enqueue under a spinlock. Target: 200k items/sec
  sustained, item size 64B.

EXPECTED
  Producers must never block past 5us p99 even under consumer stall.
  Unverified whether the spinlock or the single-consumer design is the
  binding constraint at target throughput.

SHAPE FACTS
  v1 capacity: 1M items, ring buffer, 64B-aligned slots
  producer count: up to 16 threads
  consumer: 1 thread, no batching currently

CODE (masked, names aliased, structure and control flow real — this
      algorithm is not on the sensitive list)
  [ ~40 lines: fn1 enqueue/dequeue, K1 spinlock ]

TRIED
  - Considered lock-free MPSC ring: rejected internally, no measured
    evidence yet either way.
  - Considered batched consumer (drain N per wake): not implemented,
    unsure if it violates the p99 producer-block constraint.

CONSTRAINTS
  No real names, paths, or purpose will ever be supplied — do not ask.
  Tools I have inside: <perf | none>   Rebuild: free   Rerun: on-demand
  Round trips are manual and expensive. The answer must be self-contained.

ASK
  Does the spinlock or the single-consumer design bound p99 producer
  latency first at 200k items/sec, and is lock-free MPSC or batching the
  right fix?
```

## IN

Map aliases back via `.airgap/case_<id>.map`, apply, run the VERIFY steps
that came back, record the outcome — including partial failure. Do not
re-derive the fix from memory of the original code. Names outside invented
are unmapped; rename them to project convention by hand.

## Limits

- Aliases are per case. The same symbol in a later case gets a different
  alias, so outside cannot carry knowledge across cases. Reuse a map file
  if that continuity is worth more than the isolation.
- The grep scan finds known strings and shapes, nothing else. It cannot
  see intent.
- Site policy may forbid exporting code derivatives at all. Confirming
  that is on you, not on this skill.
- A `design-review` reply is an opinion, not a proof. Validate any
  recommended change against your own gates before trusting it.
