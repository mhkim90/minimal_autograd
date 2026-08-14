# Templates

Read this immediately before sending a delegate prompt, Sol-expert capsule, or
phase report. Fill every applicable field; return the resulting evidence to the
core.

## Implementation prompt

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <command/check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <configured default | agent="luna" | sol-expert then Luna>
Elapsed-time checkpoint / final-synthesis grace (full Sol only) / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; do not commit or edit outside scope
- max attempts: <approved cap>; stop after two same-blocker failures
- use a materially revised approach before any third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```

## Sol-expert capsule

```text
Bounded read-only consultation; do not implement, edit, publish, or delegate.
Scope: <approved scope>
Question: <difficult preflight question or repeated blocker>
Evidence: <compact diff/test evidence>
Answer from this capsule when possible. If inspection is needed, each batch
resolves one named decision using a relevant file range or narrow symbol; never
use repository-wide enumeration/search or whole-file reads when a range will do.
Use at most four inspection batches; if evidence remains insufficient, return
Stop and name the missing evidence. A follow-up starts a new session with a
refreshed compact capsule.
Return: findings; proposed approach; acceptance gate; stop/go.
```

## Phase report

```text
Phase <N> complete: <commit or uncommitted state>
Publication: <published or stopped before prohibited action>
Plan preflight: <artifact/revision, audit, freshness, manual gates | L1 fast path: qualification and waiver | initiative bundle: phase envelope and manual boundaries>
Plan PR: <draft URL | normal L1 PR | none: publication prohibited>; plan: <path@SHA | none: L1 fast path>; owner approval: <plan evidence | L1 waiver | pending/invalidated>; readiness: <draft | ready | merged | none>
Scope: <approved globs>; changed files: <list>
Controller: Claude Code; active model: <runtime evidence>
Safety risk level: <L1-L4>; implementation difficulty: <mechanical/economy | standard | difficult>
Implementation route: <configured default | agent="luna" | sol-expert then Luna | agent="sol">
Implementation sessions / retries: <count> / <count>; route evidence: <requested agent, job/session IDs, bound/reported model, warnings>
Elapsed time: <per role>; checkpoints: <count>; wait-policy reviews / final-synthesis grace: <none or list>
Active local sessions at gate: <none or blocked: IDs/status>
Reviewer trigger reason: <reason or none>; independent reviewer/state/verdict: <details or pending>
Red gate: <right failure then pass>; attempts: <used>/<cap>
Validation: <commands/results>; metrics: <values>; usage: <accounting and completeness warnings, if available>
Plan deviations: <none or rationale>
Gate: <published; auto-proceeding | published; proceeding-within-approved-initiative-bundle | published; waiting-for-next-phase-approval | blocked>
```

Usage is observational evidence only; do not introduce fixed token, price, or
provider-private-path budgets or semantics.
