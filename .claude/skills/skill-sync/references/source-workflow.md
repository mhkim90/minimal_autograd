# Source Workflow

Read this reference immediately before source-side publication work. The core
skill's required inputs, preflight, scope, approval, staging, GitHub route, and
stop rules remain authoritative.

Draft is a GitHub PR state, not a separate delivery artifact. A verified
plan-only source PR contains no source-policy implementation,
executable/configuration/runtime change, generated output, or downstream sync.
Plan approval authorizes only the plan's content, scope, and route; it never
authorizes readiness or merge. Delivery requires separate explicit authority
for the exact repository/worktree, branch, paths or diff, and permitted
delivery action. After validation and required review/eligibility gates pass,
marking the draft ready is mechanical. Then issue the only final owner request:
`Approve merging <repository> PR #<N>?`.

At request issuance, bind the eligible repository, PR, head, base, and required
checks/reviews, then obtain approval. Immediately revalidate the binding and
merge only with GitHub's approved-head/expected-head precondition and normal
merge authority. Any later head or base drift, or eligibility-regressing
check/review drift, invalidates the approval; request fresh approval.
Revalidation cannot revive invalid authority. The user need not provide a SHA.
Protections, required checks, and reviewer rules remain in force.

Maintain the PR-lifecycle ledger for every source PR event, including created,
adopted, updated, readied, reopened, closed, and merged PRs. Record the full
repository/role/branch/head/base/file-set/file-mode/classification snapshot,
lifecycle, owner disposition, authority evidence, and last validation. No
source implementation, phase completion, sync, or success report proceeds
while an entry is unresolved; a Stop report may list unresolved entries.

Only an exact plan-only PR or verified non-operational documentation-only PR may
use a combined content-and-merge prompt. Revalidate that full snapshot before
and after approval. Mixed, operational, uncertain, generated, renamed,
symlinked, executable, configuration-like, profile, skill, agent, policy, workflow,
configuration, code, or plan-plus-implementation content uses ordinary merge
control and any drift requires fresh approval.

For downstream propagation, ordinary per-target rules remain in force unless
`skill-sync` has first qualified and frozen a mechanical-sync bundle. That
exception applies only to its named downstream delivery and merge records; it
never changes source implementation or source-PR approval requirements.

For a new or material shared-skill change, follow
`phase-gated-implementation`:

1. Create a verified plan-only PR in GitHub's draft state in the authoritative
   source repository. Include source paths, target matrix, validation,
   non-goals, and stop rules.
2. Wait for approval of the plan content and scope; this is not merge approval.
3. After required checks and reviews pass, mark the unchanged plan-only PR ready
   mechanically. Bind its repository, PR, head, base, and required
   checks/reviews, issue the exact final merge request above, and obtain the
   owner's approval. Immediately revalidate the binding, then merge only with
   GitHub's approved-head/expected-head precondition and normal merge authority.
4. Before source implementation, verify that the plan PR is merged. If it is
   unmerged, record a complete exception with its reason, implementation
   branch/PR when available, and concrete resolution event. A complete bounded
   exception may permit implementation on that same plan PR and branch only
   while its named resolution event remains pending and only within the named
   scope; it never permits downstream sync. When the event occurs or the
   exception condition expires, pause implementation and downstream sync, and
   resume only after the plan PR is merged or the exception is explicitly
   renewed. Without an active exception, create one separate implementation PR
   from the approved plan state in the authoritative source repository.
   Implement only the approved source manifest. Initialize a new Codex skill
   with the skill-authoring tool and include `agents/openai.yaml`; create the
   corresponding Claude skill only when it is in scope.
5. Run `quick_validate.py` for every source skill folder and `git diff --check`.
   Confirm that the implementation PR diff contains only the approved source
   manifest. Mark it ready mechanically only after all source gates pass, then
   issue its own exact final merge request and require independent owner
   approval and merge authority.

Where no independent plan gate is required, plan and implementation may share
one PR: early plan approval is a checkpoint, and final approval occurs only
after the complete implementation diff is available. L3 and any independent-
plan-gate work normally retain a merged plan-only PR followed by a separate
implementation PR. A complete bounded unmerged-plan exception may permit
same-PR implementation only while its resolution remains pending; pause and
renew the exception before continuing when its condition expires.

For an L1 mechanical fast-path source update, use the phase-gate qualification
instead: put the exact source/target manifest, validation, non-goals, and
publication policy in one normal source PR; skip the plan artifact and
plan-approval wait, but retain target preflight, validation, explicit staging,
and separate owner merge approval. For a pure target sync of an already merged
source revision, record that source revision and skip source implementation;
do not skip target preflight or gates. Use the qualified mechanical-bundle route
only when its full frozen-record requirements are met; otherwise retain normal
target-specific approval.

For an approved initiative bundle, keep every declared source phase on the
same draft source PR. A green internal phase records its commit and evidence;
it does not create another source PR or require a new plan approval unless the
approved envelope changes or a manual boundary applies.
