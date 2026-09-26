# Prompt Template

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Relevant files and required commands: <narrow list>
Red gate: <check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Routing: <luna | justified sol-implementer | justified astra-implementer | fresh read-only astra-expert | controller-bound astra-orchestrator | explicit user configured default>; omit model/variant for named agents
Expert escalation evidence: <role or none>; reason/question: <...>; why cheaper route is insufficient: <...>; requested/resolved model: <...>; effort: <...>; job/session: <...>; expected evidence: <...>; stop condition: <...>
Model binding: <named profile GPT-6 #high or explicit unnamed user route; reported match/warnings>
Elapsed-time checkpoint / final-synthesis grace (full Astra expert only) / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; no commit or edits outside scope
- include relevant files, commands, and stop rules in the cold capsule
- stop after two same-blocker failures; revise materially before a third attempt
- disclose only necessary in-scope source/diff to the configured delegate
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```

## Astra-expert capsule guard

Use this only after the entry skill has selected the bounded read-only
Astra-expert route. Give one focused question or blocker, approved scope, and
selected compact diff/test evidence. Require findings, proposed approach,
acceptance gate, and stop/go. Answer from the capsule when possible; an
inspection batch resolves one named decision using a relevant range or narrow
symbol, never repository-wide enumeration/search or a whole-file read when a
range suffices. Limit inspection to four batches and return Stop with missing
evidence if it remains insufficient. A follow-up uses a new session and a
refreshed capsule, not accumulated tool history.
