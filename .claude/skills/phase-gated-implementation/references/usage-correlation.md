# Usage correlation

Load this reference only when `usage_mcp` is configured and a prospective
workflow is being started, handled at terminal state, or closed.

1. Before the first provider execution, a Codex controller calls
   `usage_workflow_start(controller="codex", controller_session_id=<exact
   context_window.window_id>)`. Never substitute a root thread ID or a
   time-based fallback. An OpenCode- or Claude-controlled workflow may use its
   own controller value for provider-only correlation; it has no
   controller-token boundary. Supply a non-Codex identity only when it is
   already authoritative. Otherwise report controller attribution unavailable;
   never impersonate Codex.
2. Propagate the returned opaque `workflow_id` to every in-scope call that
   accepts it: OpenCode runs, continuations, and forks (including async), and
   Claude runs (including async).
3. At every terminal success, error, cancellation, or timeout, get the
   terminal result, then query `opencode_usage_get` or `claude_usage_get` by
   `job_id` or `workflow_id`. Bind only an exact provider `session_id` returned
   by that result or normalized record. First inspect
   `usage_status(workflow_id)` and skip an existing binding: duplicate
   `usage_workflow_bind` calls conflict. Never invent a session ID. Missing or
   unavailable usage remains a visible partial-warning outcome.
4. Once all known provider work is terminal, call
   `usage_workflow_finish(workflow_id)`, then scoped
   `usage_ingest(workflow_id=workflow_id, sources=["codex", "opencode",
   "claude"])`, then scoped
   `usage_report(workflow_id=workflow_id, group_by=["surface", "model"],
   format="json")`. Finish and scoped ingestion can be replayed safely; a
   failed finish is left open rather than fabricated.

Never backdate a missed start: report its controller delta unavailable and
begin a new workflow only for subsequent work. Partial, unavailable, or
unresolved-model usage never changes a quality gate or becomes a token/price
enforcement policy.
