# Usage correlation

Load this reference only when `usage_mcp` is configured and a prospective
workflow is being started, handled at terminal state, or closed.

1. Codex attribution uses only the exact current
   `context_window.window_id`; never substitute a thread ID, session ID,
   timestamp, filename, old window, or provider identity. An exact Codex
   workflow starts
   with `usage_workflow_start(controller="codex",
   controller_session_id=<exact current context_window.window_id>)`.
2. If exact Codex context is unavailable, start
   `usage_workflow_start(controller="provider_only", repo_id=<opaque value
   only if already authoritative>)` with `controller_session_id` omitted/empty.
   This is not Codex attribution; never backfill or rebind it later. OpenCode
   and Claude controllers do not perform Codex discovery: their normal fallback
   is this same sessionless provider-only workflow, never a guessed non-Codex
   controller identity. Report `controller-attribution-unavailable`.
3. `repo_id` is optional opaque metadata only. Do not derive or store raw cwd,
   repository URL/name, remote, or path. It enriches only newly ingested
   records that explicitly carry the workflow ID or match an exact trusted
   `(surface, session_id)`; never heuristically correlate, backfill, or rewrite
   conflicting metadata.
4. Propagate the returned opaque `workflow_id` to every in-scope call that
   accepts it: OpenCode runs, continuations, and forks (including async), and
   Claude runs (including async).
5. At every terminal success, error, cancellation, or timeout, get the
   terminal result, then query `opencode_usage_get` or `claude_usage_get` by
   `job_id` or `workflow_id`. Bind only an exact provider `session_id` returned
   by that result or normalized record. First inspect
   `usage_status(workflow_id)` and skip an existing binding: duplicate
   `usage_workflow_bind` calls conflict. Never invent a session ID. Missing or
   unavailable usage remains a visible partial-warning outcome.
6. At terminal state, attempt
   `usage_workflow_finish(workflow_id)`, then scoped
   `usage_ingest(workflow_id=workflow_id, sources=["codex", "opencode",
   "claude"])`, then scoped
   `usage_report(workflow_id=workflow_id, group_by=["surface", "model"],
   format="json")` regardless of partial or unavailable measurement. Do not
   block implementation or prompt on these attempts. Finish and scoped
   ingestion can be replayed safely; a failed finish is left open rather than
   fabricated.

Never backdate a missed start: report its controller delta unavailable and
begin a new workflow only for subsequent work. Provider-only reports retain
top-level `complete` for provider/report completeness and add
`controller_attribution.status="unattributed"` with the
`controller-attribution-unavailable` warning. Report both dimensions. Partial,
unavailable, unresolved-model, or unattributed usage never changes a quality
gate or becomes a token/price enforcement policy. Exact Codex workflow reports
and legacy behavior remain unchanged.
