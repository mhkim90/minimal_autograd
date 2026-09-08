# Porting record — autonomous-goal-loop

Source manifest for this skill's port into airgap-skillset, per the
Sol-expert preflight finding that "copied verbatim" needs a reproducible
definition. This file is the mechanical target for the allowlisted-diff
check; it does not add a safety control to the loop itself.

## Pinned source revision

- `../lab` (sibling repo), commit `9c950bf433a8745c7257f235f24bb4070becd238`
  (clean working tree at time of copy).
- Better provenance, found during this port: `claude_mcp` itself already
  carries a byte-identical, unmerged copy of the doc layer on
  `agent/autonomous-goal-loop-sync` (commit `06c524e`, part of an unrelated,
  separate in-progress bundle — not touched by this work). Diffed
  `agent/autonomous-goal-loop-sync:.claude/skills/autonomous-goal-loop/SKILL.md`
  against `../lab`'s current `SKILL.md`: zero differences. Either commit is
  an equally valid pin for the doc layer; `../lab` is used below since it is
  also the source for the reference implementation, which
  `autonomous-goal-loop-sync` does not carry.

## Copied files (path, sha256, mode, all from `../lab` at the pinned revision)

### Doc layer → `airgap-skillset/autonomous-goal-loop/`

| Source path | sha256 | mode | Target |
| --- | --- | --- | --- |
| `.claude/skills/autonomous-goal-loop/SKILL.md` | `edd8e5146b6f22c17ea474dd6ecca2dffb4cb2b77c8c8271b1f7f0842c9fe785` | 664 | `SKILL.md` (edited — see below) |
| `.claude/skills/autonomous-goal-loop/references/charter-and-ledger.md` | `efa89309b403eb0a038f7f0eb4497e6e160134d192b38bb97403871ffae5372a` | 664 | `references/charter-and-ledger.md` (verbatim) |
| `.claude/skills/autonomous-goal-loop/references/driver-and-recovery.md` | `afc8f6f5abefd2c2c2990afb1ad2ae46f65aed88f39de5370e1d6ed1a624e9a4` | 664 | `references/driver-and-recovery.md` (edited — see below) |

### Reference implementation → `airgap-skillset/autonomous-goal-loop/reference-impl/`

| Source path | sha256 | mode | Target |
| --- | --- | --- | --- |
| `experiments/goal-loop-p4/README.md` | `accc30cddc63f642609ef016f2b5148ed96bb57a8eeba5782a9ff9166bdf6fe5` | 664 | `README.md` (edited — see below) |
| `experiments/goal-loop-p4/loop/bounds_test.py` | `bcbb31a6c8fd066c52ba741f304c3553a229b289f46777dc6e4cefbc4d5490f9` | 664 | `loop/bounds_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/bundle_b4_b6_test.py` | `68c5f140cd7997fda9cfb394d10b1f3b58e7ea7952090b95563b6ea0014339b9` | 664 | `loop/bundle_b4_b6_test.py` (edited — see below) |
| `experiments/goal-loop-p4/loop/clock.py` | `53a323cabc227a5c8717a0f7d0fb0793523dae47fb6e2f2eedc37ceb7fef03ca` | 664 | `loop/clock.py` (verbatim) |
| `experiments/goal-loop-p4/loop/collector.py` | `a52950e7045b5d5601f0144a20822a7793f9d89bcba02d074f983ee8e453b2e9` | 664 | `loop/collector.py` (verbatim) |
| `experiments/goal-loop-p4/loop/conftest.py` | `5b69306284213af2778a98445999e1b3faa2c6c651e0bc4d0dbd1b32c2f67fdf` | 664 | `loop/conftest.py` (verbatim) |
| `experiments/goal-loop-p4/loop/driver_reserve_and_digest_test.py` | `68e1ea8684e83615d19a789242781ffd91812546c61a1285354b65bd108014f4` | 664 | `loop/driver_reserve_and_digest_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/evaluator.py` | `ae28025943a44391409c6a3369d4f1422aa466a577067954d5729b3e512664d5` | 664 | `loop/evaluator.py` (verbatim) |
| `experiments/goal-loop-p4/loop/goal-loop-p4.service` | `a0045020f08b6f97765a827e5ad304f8d5e208b969256b1516ee9d087661358e` | 664 | `loop/goal-loop-p4.service.example` (templated — see below; **not** verbatim, renamed) |
| `experiments/goal-loop-p4/loop/iterate.py` | `bf0ab4cf4dd788f6630ad149ad45a7b0a129a077b1bf1fcc0ab513fec8b4333b` | 664 | `loop/iterate.py` (verbatim) |
| `experiments/goal-loop-p4/loop/ledger_fd_lock_test.py` | `f94a4cc56833e1b8f926d4c42c0b0cae23780a9ba8bc3e972cb9a89bcdaf6912` | 664 | `loop/ledger_fd_lock_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/ledger.py` | `41dd2963741a891f1b89ed0cf0cf3704afaf8ee46f38c40f31fa5cc480f3259a` | 664 | `loop/ledger.py` (verbatim) |
| `experiments/goal-loop-p4/loop/make_containment_receipt.py` | `7824aea842e8867e95ec081ecf6f33045bf93fcd6ac6b146efe9148e2fc77188` | 664 | `loop/make_containment_receipt.py` (verbatim) |
| `experiments/goal-loop-p4/loop/negative_test.py` | `12ab284eb0c36675d03784f937e9f08432c2937f81781b4867d859f557f7655c` | 664 | `loop/negative_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/notifier.py` | `f6f6414618c372cefdaf8220180a6523405be0ce656c852a7e71fb7d2ab5a1d3` | 664 | `loop/notifier.py` (verbatim) |
| `experiments/goal-loop-p4/loop/preflight.py` | `9c330c14de11c27002ae6404bfd0764a62f11694fa4c1510f707b542bedb72cd` | 664 | `loop/preflight.py` (verbatim) |
| `experiments/goal-loop-p4/loop/receipt_declaration_test.py` | `66eda7febb33cb0730245d484dbdef131d2b0ebf434d6d042cd34292c01f200b` | 664 | `loop/receipt_declaration_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/red_gate_test.py` | `670e47c8c18f9ce81c26c2166c74cf683c7256a8619939364b32b2fe857704a8` | 664 | `loop/red_gate_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/regression_test.py` | `f0e838f06023ed4241fd811d6f4666bbd7ddaccab2b4c17da62a28c3967452d5` | 664 | `loop/regression_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/run.sh` | `8231e31ac2fe6cd10546d224922c6ce790faef555dd5a88eda3eb79f134ea106` | 775 | `loop/run.sh` (verbatim) |
| `experiments/goal-loop-p4/loop/run-supervised.sh` | `93c74f02895605e9d29d9513f6aee7448a269dffe693a44d4340d74bc2b21b58` | 755 | `loop/run-supervised.sh` (verbatim) |
| `experiments/goal-loop-p4/loop/runtime_binding_probe.py` | `74f5bb1356fd52435a1e74df4b5c950c81f12c8de539ae39382793b3e58f2976` | 664 | `loop/runtime_binding_probe.py` (verbatim) |
| `experiments/goal-loop-p4/loop/safety_test.py` | `f386fef5883b147ac32c12041e7bfef37555e57e3dc02aa2b1c97993417b8d5c` | 664 | `loop/safety_test.py` (verbatim) |
| `experiments/goal-loop-p4/loop/security_bundle_test.py` | `a9f054e0e0dd4752fffe25442c0033173a5f84c9591ebeae4de6bc03049c4545` | 664 | `loop/security_bundle_test.py` (edited — see below) |
| `experiments/goal-loop-p4/loop/source_manifest.py` | `5609648fd2c19b4839167b7d616d155e7841b5384a2bb1a3cb2ada88368c5354` | 664 | `loop/source_manifest.py` (verbatim) |
| `experiments/goal-loop-p4/loop/stop.py` | `90859556d65b62b5daa4b6518750b27853d8be934d1d331caf7f696fd2e6613b` | 664 | `loop/stop.py` (verbatim) |
| `experiments/goal-loop-p4/loop/stop.sh` | `d6ad84273427cd81edf83d2b06a9c0b850f8d10447d8996fb7ad96687655b515` | 775 | `loop/stop.sh` (verbatim) |
| `experiments/goal-loop-p4/loop/testkit.py` | `78d828de6156fc3626217c02488479f4cd9b3d59a0119444cdeeecb8d8fbec36` | 664 | `loop/testkit.py` (verbatim) |

### `evidence/bundle1/` → `airgap-skillset/autonomous-goal-loop/reference-impl/evidence/bundle1/`

Revised during the port: `evidence/bundle1/` (the four-probe negative-test
receipt) turned out to be a **required test fixture**, not just deployment
evidence — `receipt_declaration_test.py` (7 tests) and one test in
`bundle_b4_b6_test.py` open it directly to exercise the receipt-parsing and
digest-binding contract in `preflight.py`. Excluding it (the original plan)
broke those 7 tests. `evidence/bundle2/` and `evidence/bundle3/` have no
such dependency — grepped, nothing in the kept suite reads them — so they
stay excluded per the original reasoning (operator-supplied, one-machine
evidence).

| Source path | sha256 | mode | Target |
| --- | --- | --- | --- |
| `evidence/bundle1/negative_test_receipts.json` | (unchanged from source) | 664 | `evidence/bundle1/negative_test_receipts.json` (verbatim) |
| `evidence/bundle1/targets.json` | (unchanged from source) | 664 | `evidence/bundle1/targets.json` (verbatim) |
| `evidence/bundle1/control.json` | (unchanged from source) | 664 | `evidence/bundle1/control.json` (verbatim) |
| `evidence/bundle1/contained.json` | (unchanged from source) | 664 | `evidence/bundle1/contained.json` (verbatim) |
| `evidence/bundle1/receipt-declaration.json` | (unchanged from source) | 664 | `evidence/bundle1/receipt-declaration.json` (edited — see below) |

Explicitly **not copied**: `evidence/bundle2/`, `evidence/bundle3/` (this
machine's own operator-supplied containment receipts, not needed by any
kept test — not portable to a different machine; see the skill's own
threat-model language), `__pycache__/`, `.pytest_cache/` (build artifacts,
not source).

## Documented deviations from verbatim (the only ones an allowlisted diff
should find)

1. `SKILL.md` — router table: `opencode-delegate` row removed,
   `memory-continuity` row redirected to `handoff`; headless-invocation
   `claude -p` example reworded harness-neutral; Handback line added about
   airgap `phase-gated-implementation`; in-doc reference path
   `experiments/goal-loop-p4/loop/` rewritten to `reference-impl/loop/`.
2. `references/driver-and-recovery.md` — same headless-invocation reword,
   same path rewrite; the generic containment/timer unit examples in this
   file are unchanged (they were already placeholder-style, not this
   machine's real values).
3. `loop/goal-loop-p4.service` → `loop/goal-loop-p4.service.example` —
   renamed; `WorkingDirectory=`/`ExecStart=` replaced with template
   placeholder paths (not this dev machine's real path, not silently
   presented as a real deployment). Every other field (`TimeoutStartSec`,
   `PrivateNetwork=yes`, `ProtectSystem=strict`, resource ceilings, etc.)
   unchanged.
4. `loop/bundle_b4_b6_test.py` — two edits: (a) removed
   `test_skill_mirrors_are_byte_identical` (asserts `lab`'s own
   `.claude`/`.codex` dual-tree mirror layout, which airgap-skillset does
   not have), with a one-line comment explaining why; (b) the test reading
   `goal-loop-p4.service` now reads `goal-loop-p4.service.example` and
   asserts the security-relevant fields listed above rather than an exact
   deployment path.
5. `loop/security_bundle_test.py` — same filename-and-assertion update as
   4(b), same file, different test in that suite.
6. `evidence/bundle1/receipt-declaration.json` — `receipt_path` rewritten
   from the original lab checkout's absolute path to this checkout's;
   `provenance_note` and `what_this_is` extended with one sentence each
   noting the fixture's ported, checkout-relative nature. `targets`,
   `receipt_sha256`, and `run_id` unchanged — `targets` is only ever
   compared for JSON equality against the receipt's own `targets` field
   (never touched on disk, so the original lab values stay valid as inert
   labels).
7. `README.md` — one blockquote note prepended after the title, pointing
   at this file for what's excluded/renamed/relocated. No other line
   changed; the rest of the file's verification history and threat-model
   prose is left as accurate historical record of the original checkout.

Every other copied file is expected to diff empty against its pinned
source. Phase 2/3 run that diff mechanically; this table is the checklist,
not the proof.
