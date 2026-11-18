---
name: autonomous-goal-loop
description: Handle unattended, machine-measured work under one owner-approved envelope. Only frozen-subject, model-free Evidence runs with the verified lab P4 runtime and approved host are eligible. Refuse autonomous editing or search.
---

# Autonomous Goal Loop

**Evidence only.** Repeat a measurement on a frozen subject; the collector and evaluator, not a model, decide the outcome. `model: null` means no invocation or fallback. Ratchet (editing toward a metric) and Search are not implemented. Refuse those requests rather than relabeling them Evidence.

## Admit a run

This skill contains **no executable runtime**. Use the existing P4 implementation only when all three bindings hold:

1. **Source.** From an authorized checkout of the trusted `lab` repository, verify commit `938d930d8d1d9e618f31e87dd5e59e88bb78b908` and its `experiments/goal-loop-p4` tree `d6eb17467c05b5c143918ee0ad3cda413df055b2`. Verify the files actually used by the driver match the pin, including its declared dependencies; exclude working-tree overlays or shadowing files. A branch name, neighboring path, matching commit marker alone or copy of this skill is insufficient. Do not fetch, copy, install or reset anything to satisfy the gate without separate authority.
2. **Authority.** Require an owner-approved, run-specific P4 unattended charter and its bound approval artifacts: L1/L2, frozen subject outside writable state, machine-checkable metric and stopping rule, baseline, budgets/deadline, explicit effects, allowed paths and closed model-free route. Respect the pinned README's “Not defended” limits; do not run if they exceed the approved envelope. Charter approval itself can execute commands; never bootstrap it merely to check eligibility.
3. **Host.** Use only the authorized P4 unattended driver in the approved contained host unit. Its native source/subject, receipt, live unit binding, containment and budget checks must pass **before a trial**. Host evidence must bind this host, unit, charter and source. Historic receipts, a passing supervised fixture, and controller visibility are not live validation. Never substitute supervised mode, a shell loop, a new service, or wider permissions.

If source, charter or approved host route cannot be established read-only, report the missing binding and stop **before dispatch or side effects**. Once dispatched within that envelope, leave validation, locked ledger, durable stop and recovery to P4; propagate refusals, never simulate a successful check or restart around one. Keep external records; no cleanup or disposal authority. Report actual stop reason and observed measurements as operational, not audit-grade, evidence.

A loop grants no subject edits, delivery, readiness, merge, target inspection, sync or deletion. An iteration is not a phase. Separate code changes and publication follow [phase-gated-implementation](../phase-gated-implementation/SKILL.md).

## Evidence and design references

For P4 behavior consult `experiments/goal-loop-p4/README.md`, `loop/` and `evidence/` **at the pin above**. The historical design notes below are not alternate runtime instructions or host evidence. Read only for the relevant design question:

- [Future design](references/future-design.md): classes, authority, integrity and stop conditions.
- [Entry contract](references/entry-contract.md): Read before separately authorized unattended entry approval, entry or iteration-start changes, or explaining an entry refusal. The pinned implementation remains authoritative; reading this note grants no approval.
- [Charter and ledger](references/charter-and-ledger.md): protected state and authority boundaries.
- [Driver and recovery](references/driver-and-recovery.md): driver scope and recovery boundaries.
