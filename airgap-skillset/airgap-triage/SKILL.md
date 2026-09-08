---
name: airgap-triage
description: >
  Use OUTSIDE the closed network when the user pastes a de-identified case
  package from a secure site (header "# CASE <id> / OUT", placeholder names
  like fn1 / v2 / P1). Diagnoses from aliased code alone, or gives a design
  opinion for a `CLASS: design-review` package, never asks for
  de-identified context, and answers in a one-shot format that can be
  carried back and applied by hand. Triggers: "CASE ... / OUT",
  "/airgap-triage".
---

# airgap-triage (OUTSIDE)

The person on the other end cannot copy-paste, cannot re-run on demand, and
cannot tell you what the code is for. One answer = one physical trip.
Optimize for that.

## Rules

1. Never ask for real names, paths, or purpose. Aliases are ground truth.
   `fn1` is a fine name to reason about and to keep using.
2. Never guess the domain. Do not write "this looks like an
   image-processing / semiconductor / simulation pipeline". Do not
   reintroduce domain vocabulary even if you infer it. Reason about memory,
   control flow, numerics, and toolchain only.
3. Reuse their alias spelling exactly. Renaming `fn1` to anything readable
   breaks their mapping back.
4. A capable model already worked this. For a bug, everything under TRIED
   is exhausted — do not lead with the obvious cause unless you can show
   why their rejection was wrong. For `CLASS: design-review`, TRIED lists
   alternatives already considered internally — do not simply re-propose
   one. Either way, go for what needs knowledge they lack:
   architecture-specific semantics, library version behavior,
   undefined-behavior corners, hardware model details.
5. The snippet is incomplete and possibly reconstructed. Say which unseen
   piece would change your answer instead of assuming a body you cannot see
   is correct.
6. Build only from their tool inventory. Every check you propose must run
   with the tools listed under CONSTRAINTS. Where it would not, say what
   that check would have shown and stop there.
7. No round-trip-per-hypothesis. Rank causes, then give a decision tree
   they can walk alone.
8. Verify what you can. If a repro is included, compile and run its CPU
   path before answering; report exactly what you ran. Say plainly when a
   claim is unverified for lack of a GPU or nvcc.
9. If the package leaks something identifying — a real path,
   native-language text, a product name — say so in one line at the top so
   they can tighten the filter, then continue.

## Reply format

```
# CASE <id> / IN

VERDICT      Most likely cause, one paragraph, mechanism level.
EVIDENCE     Which line, field, or number supports it — and what would falsify it.
FIX          Unified diff or complete replacement blocks, in their alias names.
             "// unchanged" only for lines the fix does not touch.
IF NOT       Ranked alternatives. Each: check to run → observation → conclusion → fix.
VERIFY       Exact commands and expected output, runnable with their listed tools.
NEW SYMBOLS  Names I introduced; they are unmapped and will be renamed inside.
NEXT CAPTURE Concrete artifacts to bring out if it still fails — counters, printf points,
             register/smem numbers, bisection cases. Never a request for context.
```

For `CLASS: design-review`, VERDICT is the assessment of the approach (not
a root cause), EVIDENCE is what in the package supports it, and FIX is the
recommended change or an explicit "approach is sound as described."

## Style

Match their compression level. No preamble, no restating their problem
back at them, no praise. Do not soften uncertainty — rank it.
