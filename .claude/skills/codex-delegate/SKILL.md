---
name: codex-delegate
description: Codex (GPT-5.5) is the engine for PHYSICS / deep thinking / adversarial design review / hard implementation. DISCUSS (read-only) for reasoning & second opinions; EXECUTE (workspace-write) for difficult multi-file changes. Trigger on "ask codex", "second opinion", "discuss with codex", "delegate to codex", or any hard/physics reasoning or implementation.
---

# Codex Delegate

Two MCP tools drive Codex:
- `mcp__codex__codex` — start a session (`prompt` required; set `sandbox`, `approval-policy`, `cwd`, `model`).
- `mcp__codex__codex-reply` — continue a thread (`threadId` + `prompt`).

The Codex MCP environment does **not** provide GPU access. Use Codex for CUDA/GPU
reasoning, kernel review, and code edits, but do not ask it to run GPU-dependent
builds, tests, benchmarks, profilers, or device probes. Run GPU validation from
Claude/OpenCode/local shell where a GPU is actually available.

Codex runs on **GPT-5.5** by default (`model_reasoning_effort = "high"` in `~/.codex/config.toml`).
Override with the `model` param if needed (e.g. `gpt-5.2`, `gpt-5.2-codex`).

In the delegation model Codex is the **hard-work engine**: numerics/algorithm design
(autograd correctness, gradient formulas, optimizer behavior), deep thinking,
adversarial design review, and implementing *difficult* things — a genuinely
independent (non-Claude) reasoner. Claude manages; OpenCode handles the tedious
mechanical jobs (see [[opencode-delegate]]).

## Two modes

| Mode | `sandbox` | `approval-policy` | Use it for |
|---|---|---|---|
| **Discuss** | `read-only` | `never` | Second opinion, adversarial review of a plan/root-cause, design forks (e.g. autograd tape design, tensor shape semantics, optimizer updates). Codex reads files and argues — does NOT touch the tree. |
| **Execute** | `workspace-write` | `never` | Long CLI runs, sweeps, multi-file edits. The Codex twin of opencode-delegate. The sandbox is the safety boundary, so `never` is fine for unattended runs. |

Always set `cwd` to the repository root (`<repo-root>`).

## When to use Codex vs OpenCode

- **Hard reasoning / adversarial review / hard implementation** (e.g. numerical
  correctness of a new op, gradient derivation, autograd tape behavior, optimizer
  design tradeoffs) → **Codex**. It genuinely cross-checks Claude's own reasoning
  and is the engine for difficult work.
- **Tedious / mechanical / long-running jobs** (precise edits, build+test sweeps,
  applying an agreed plan) → **OpenCode** (see [[opencode-delegate]]).
- **Routing, verification, decisions** → Claude keeps these inline.

Do NOT use either engine for: single-step commands (just run them), or ambiguous
design calls that should be resolved with the user first.

## Discuss mode — how to prompt

Give Codex enough to form an *independent* judgment, and ask for a verdict, not a summary:

```
sandbox: read-only
prompt:
  Context: <1-2 sentences on the project + the specific decision>.
  Read first: <key files, e.g. include/...:line, src/...:line, test/...:line>.
  Question: <the fork / claim to stress-test>.
  I currently lean <X>. Argue the strongest case AGAINST that and tell me which
  option you'd pick and why. Be concrete; cite file:line where relevant.
```

Relay Codex's verdict and where it *disagrees* with your own take — the disagreement
is the value. Then bring the decision back to the user.

## Execute mode — how to prompt

Same cold-context discipline as opencode-delegate (it works without conversation history):

```
sandbox: workspace-write
prompt:
  Context: <what this project is and what we're doing>.
  Task: <exact steps>.
  Files to read first: <list>.
  Success criteria: <what done looks like, e.g. which of the three test binaries
    must print ALL TESTS PASSED>.
  Constraints:
    - Working dir: <repo-root>
    - Never commit unless explicitly told
    - Match existing code style
```

## Thread management

- **New task / new discussion** → call `mcp__codex__codex` (fresh thread).
- **Continue** → `mcp__codex__codex-reply(threadId=<id>, prompt=...)`. Capture the
  thread id from the initial `codex` response and store it if you plan to follow up.
- Long thread degrading → start a fresh `codex` session with a one-paragraph recap
  instead of replying into the stale one.

## After delegation

1. Report the thread id to the user for reference.
2. State which mode (discuss / execute) and what Codex was asked.
3. Relay findings concisely — for discuss, lead with Codex's verdict and any
   disagreement with your own reasoning; for execute, summarize what changed and
   the verification result. Don't dump the raw response.
4. If Codex hits a blocker or asks a question, surface it to the user before continuing.
