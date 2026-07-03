---
name: grilled-me
description: Use when Codex is drafting, reviewing, or presenting a plan and should perform adversarial self-review first. Stress-test assumptions, scope creep, risks, failure modes, simplicity, blind spots, irreversible actions, and whether every step traces to the user's request.
---

# Grilled-Me

Use this as an adversarial self-review before presenting or acting on a plan.

## Workflow

1. Draft the plan.
2. Review the plan as a skeptical senior engineer.
3. Revise the plan or explicitly flag surviving risks.
4. Present only the improved plan unless the user asks for the review details.

## Checklist

### Assumptions

- What am I assuming that I have not verified?
- What changes if the assumption is wrong?
- Did I silently pick an interpretation that should be surfaced?

### Scope

- Am I solving more than the user asked?
- Did I add nice-to-haves that are not required?
- Does every planned step trace to the request?

### Risk

- What is the most likely way the plan fails?
- What is the worst-case impact of each step?
- Are any steps irreversible or destructive?

### Simplicity

- Is there a simpler plan with the same outcome?
- Can any steps be merged or removed?
- Is the plan over-engineered for the request?

### Blind Spots

- What relevant file, system, or side effect am I not considering?
- Is there existing code or documentation that already solves part of this?

## Output

If showing the review explicitly, use:

```text
[Grilled-Me Review]
Assumptions confirmed: ...
Risks identified: ...
Simplification applied: ...
Surviving concerns: ...
```

If the review finds no material issue, state `[Grilled-Me: No issues found]`.
