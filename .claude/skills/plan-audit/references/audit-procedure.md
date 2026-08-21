# Audit Procedure

Apply the checks that match the artifact. Mark irrelevant checks as not
applicable rather than forcing every plan through a code-refactor template.

## 1. State What the Plan Is Trying to Do

Summarize:

- intended outcome;
- affected system and users;
- proposed phases or major steps;
- explicit non-goals;
- success and rollback conditions.

If any of these are absent, record that before inferring intent.

## 2. Check Scope and Non-Goals

Verify that the plan:

- separates required work from follow-up work;
- names externally visible behavior that must not change;
- identifies files, components, interfaces, or systems that may change;
- excludes tempting adjacent work;
- avoids phase boundaries so broad that drift cannot be detected.

Flag a phase that requires a boundary such as `src/**` when a narrower
vertical slice should be possible.

## 3. Check Contracts

For APIs, data structures, kernels, storage, views, and low-level code, check:

- memory and element order;
- mutability and observable state;
- aliasing and view-versus-copy behavior;
- ownership and lifetime;
- synchronization and concurrency assumptions;
- error and partial-failure behavior;
- compatibility with current callers and serialized data.

For configuration, documentation, migrations, operations, or experiments,
check:

- inputs, outputs, and source of truth;
- idempotency and repeatability;
- compatibility and rollout ordering;
- rollback and recovery;
- failure visibility and observability;
- environment and permission assumptions;
- measurable comparison or baseline requirements.

If the plan freezes existing behavior with characterization tests, distinguish
an intentional guarantee from a known bug preserved only for migration safety.

## 4. Check Slicing and Sequencing

Prefer vertical phases that leave the system coherent and testable. For each
phase, ask:

- Does it produce an observable, validated result?
- Are its dependencies already available?
- Can it be rolled back or stopped without corrupting later work?
- Does it require two implementations to coexist?
- If so, is the compatibility window bounded and is one side a thin adapter?
- Does a later phase depend on evidence that an earlier phase deletes or makes
  impossible to collect?

Identify the first phase at which each sequencing problem becomes blocking.

## 5. Map Claims to Enforcement

For every important claim, determine whether it is enforced by:

- an executable test or validation command;
- build, type, schema, permission, or configuration enforcement;
- a review-only convention;
- manual observation;
- nothing.

Documentation wording is not executable enforcement. A guardrail is not a
sandbox. Call out claims whose stated confidence exceeds their enforcement.

## 6. Check Gates and Exit Criteria

Each phase should declare:

- a red gate or pre-change observation when meaningful;
- the expected failure or current behavior and why it is the right signal;
- a green validation command or inspection;
- falsifiable exit criteria;
- stop conditions;
- rollback or handoff behavior after failure;
- an attempt or iteration bound for retry loops.

Do not accept “tests pass,” “works,” or “performance improves” without naming
the relevant command, workload, baseline, or threshold.

## 7. Check Downstream Coupling

Trace important changes to:

- callers and consumers;
- adapters and compatibility layers;
- tests and fixtures;
- documentation and examples;
- build, packaging, deployment, and CI;
- monitoring, benchmarks, or operational procedures.

Report hidden consumers or coupled follow-up work that would otherwise appear
only during implementation.

## 8. Check Decision Provenance

Separate:

- decisions already approved by the owner;
- facts supported by repository evidence;
- assumptions made by the plan author;
- recommendations introduced by the audit;
- decisions that still require a human.

Do not convert a recommendation into an approved requirement through confident
wording.
