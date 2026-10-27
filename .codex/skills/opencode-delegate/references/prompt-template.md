# Prompt Template

Canonical implementation prompt; phase templates link here rather than copy it.
Load the [actual-delegate capability gate](../SKILL.md#boundary),
[failure taxonomy and recovery](../SKILL.md#failure-classification-and-recovery),
[native evidence rules](../SKILL.md#native-evidence-and-diagnostics), and
[wait procedure](wait-policy.md) at their transitions. Read-only experts need
capabilities for authorized reads, not an editor gate.

```text
Context/task: <project, one bounded phase, exact edit/run loop>
Binding: <repository, exact worktree/branch, expected HEAD, approved dirty state,
absolute Git-dir/common-dir, immutable plan path/SHA256, original P0/provenance>
Files/scope and required commands: <approved paths, narrow relevant reads/checks>
Red gate: <check and expected right-reason failure; fixture error is not red>
Acceptance: <GREEN checks/metrics, or explicitly required preparation RED
milestone; never treat RED as implementation GREEN; approved scope remains fixed>
Safety risk: <L1-L4>; difficulty: <mechanical/economy | standard | difficult>
Route: <luna | justified sol-implementer | justified astra-implementer |
fresh read-only astra-expert | controller-bound astra-orchestrator |
explicit user configured default>; named roles omit model/variant
Route evidence: <requested role, job/session lineage, bound/reported model,
warnings; selector contradiction stops, missing observation alone warns>
Expert evidence, if applicable: <question/reason, cheaper-route insufficiency,
expected evidence, stop condition, prior diff/test evidence>

Capability preflight (implementation only):
Use actual exposed execution/read tools and explicitly select the approved
worktree. Fail-fast compare status (optional Git writes disabled), branch, HEAD,
absolute Git-dir/common-dir and exact plan SHA256; preserve original checkout,
metadata, plan/P0 and other dirty work. Actual invocations/exits are evidence,
not printed JSON, wrapper completion, supplied visibility, or prose.
If no native editor is exposed, discover an existing permitted shell editor;
missing apply_patch is not proof editing is unavailable. Do not assume patch,
install, or bypass permissions. Listing is not proof: first authorized minimal
in-scope edit plus immediate readback, or expressly authorized absent-marker
create/readback/removal, proves editing. Record exact invocation/exit/readback.
Recheck after relevant checkout/runtime context changes, not on every command.

Time: <actual launch; progress checkpoints; task-specific synthesis allowance;
maximum wait; explicit overall deadline if any; no unexplained micro-budgets>
Recovery: <one controller-launched fresh same-named-role session per phase after
confirmed terminal/result/diff review, or none; retain caps, lineage and budgets>
Candidate allocation: <counting unit and scope; otherwise-unallocated phase
default five cycles, prospectively; explicit lower bounds still control, and reload
never reopens an allocation; each declared stage allocation cap and used/remaining;
every applicable shared-limit cap and used/remaining; no implicit fresh allowance
per stage>
Constraints:
- one bounded phase; no commit/stage/push/PR/delegation or out-of-scope edits
- follow the [canonical candidate-accounting contract](../SKILL.md#candidate-accounting)
  for repair, counting, preparation RED, and applicable shared limits; ordinary
  in-scope repairs need no new owner prompt
- preserve consumed history; carry counts only through already-bound shared limits,
  not into a separately approved successor's own allocation
- suspend affected edits for essential absence/interruption, even midphase;
  authoritative checkout/HEAD/plan mismatch, access denial or drift is hard STOP
- after two same-blocker failures return evidence for controller expert help; every
  further candidate on that blocker requires a materially revised approach and
  remaining allowance; expert help grants no new consultation allotment
- no self-launched recovery, live-job duplication, role/model/destination change,
  denial evasion, invented tool execution, or cap/deadline reset
- only necessary in-scope source/diff to the bound configured destination
Return briefly: changed files, candidate attempts, commands/results, unresolved
issues, route/job/session evidence, and capability proof or narrow STOP.
Use the linked native-evidence contract: distinguish model report from native
absence/denial and uncertain cause; normalized diagnostics are bounded/redacted,
never raw private provider content. Retain exact authorized manual invocation,
exit and readback evidence with credentials redacted. Never treat uncertainty
as proof or resume absent capability without approved fresh proof.
```

## Astra-expert capsule guard

Only after the entry skill selects the bounded read-only route, give one focused
question, approved scope, and compact relevant diff/test evidence. Require
findings, proposed approach, acceptance gate, and stop/go. Answer from the
capsule when possible; inspect only to resolve named decisions with relevant
ranges/narrow symbols, never repository-wide/secret enumeration or whole-file
reads when a range suffices. Use declared wait/stop limits, not a universal
batch ceiling. Stop when evidence is unavailable or further inspection no
longer addresses the question. One consultation plus at most one follow-up;
follow-up uses a new session and refreshed capsule, not accumulated tool history.
Never implement, edit, publish, delegate, or invoke repository skills.
