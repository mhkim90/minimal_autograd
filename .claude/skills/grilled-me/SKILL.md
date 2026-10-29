---
name: grilled-me
description: Adversarially self-review your own plan when review is requested or warranted. For an externally authored plan, use plan-audit instead.
---

# Grilled-Me: Adversarial Plan Self-Review

When reviewing a plan you drafted, stress-test it before asking for approval
as a skeptical senior engineer would:

1. **Assumptions**: what is unverified, what if it is wrong, and did you
   silently pick one interpretation?
2. **Scope**: is every step traceable to the request, with no
   nice-to-haves or "while I'm at it" refactors?
3. **Risk**: the most likely failure, each step's worst case, and every
   irreversible step (delete, overwrite, migration) and its rollback.
4. **Simplicity**: would a smaller plan reach the same goal; can steps
   merge or go?
5. **Blind spots**: side effects on other files or systems, existing code
   that already does part of this, and steps too vague to execute ("fix
   the issue").

Revise the plan, then report:

```
[Grilled-Me Review]
Assumptions confirmed: ...
Risks identified: ...
Simplification applied: ...
Surviving concerns (flagged to user): ...
```

If nothing changes, state `[Grilled-Me: No issues found]`.
