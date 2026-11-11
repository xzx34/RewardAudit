# Contributing

## Setup and checks

Use Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests examples
```

The development extra includes the optional data and math packages. To check the default
install, use a separate environment, install `-e .` and `pytest`, and run the same tests.

## Vocabulary and changes

A **probe** sends replies of known correctness to a checker. A **finding** records how
the checker behaved; read its example against the task's rules before calling it a defect.
A **non-solving response** does not solve the task. An **equivalent rewrite** gives the
same right answer in another form. Use these neutral terms in code, reports and documents.

Ids are only appended. Never reuse a skipped or retired probe, defect or finding id,
or change what an existing id means. Retired probe ids are M11, M12, M13, M17, Q31 and N11;
they are never reused. See the [stability policy](docs/guide.md#what-stays-the-same-across-versions).
Every change a user can notice needs a test and a line under `Unreleased` in `CHANGELOG.md`.
Keep special cases taken from real checkers: they make judgements more accurate.

## How the code is organised

Paths here are under `src/rewardaudit/`:

- `__init__.py`: version and documented Python exports; `py.typed` marks the package as typed.
- `cli.py`: commands, arguments, progress, output and exit codes.
- `api.py`: the `audit()` entry point, checker routing, dataset input and audit configuration.
- `pipeline.py`: stage ordering, probe calls, saved calls and the `AuditResult` object.
- `runner.py`: worker processes, parallel calls, deadlines, memory limits and restarts.
- `worker.py`: JSON Lines protocol, adapter loading, scoring, network blocking and import observation.
- `adapters/__init__.py`, `adapters/base.py`: adapter selection, the common interface and task sampling.
- `adapters/dataset.py`: dataset loading, columns, row ids and empty-reference filtering.
- `adapters/function_adapter.py`: function signatures, reference values and returned scores.
- `adapters/verifiers_adapter.py`: environment and taskset loading and their own scoring paths.
- `envs.py`: Hub package installation, separate environments and verifiers version selection.
- `types.py`, `errors.py`: tasks, verdicts, observations and audit errors.
- `gate.py`: unsupported settings and reference checks that can stop an audit.
- `goldtype.py`, `prompt.py`: answer types and the prompt's answer instructions, options and ordering.
- `contract.py`: accepted reply formats, reference scores and discriminating score components.
- `equiv.py`: equivalence, numeric and structured parsing, and answers stated in replies.
- `probes/base.py`, `probes/registry.py`: probe definitions, context, family lists and selection.
- `probes/null.py`, `candidates.py`, `leak.py`: N, E and L non-solving responses.
- `probes/hedge.py`, `mutate.py`, `rewrite.py`: O hedges, M wrong answers and Q equivalent rewrites.
- `probes/reference.py`, `robustness.py`: G reference variations and P robustness inputs.
- `probes/structured.py`: shared list and grid probes in the M and Q families.
- `metrics.py`: call tallies, rates, intervals, findings, severity and defect collection.
- `defects.py`: the catalogue and each defect's probe membership.
- `confirm.py`: repeated evidence calls and shortening accepted replies.
- `dataset_checks.py`: checks on all rows, without checker calls.
- `report.py`: terminal output, Markdown and JSON reports, and generated `repro.py`.
- `comparison.py`: alignment and comparison of two saved calls files.
- `explain.py`, `demo.py`: id lookup, generated reference documentation and built-in demonstrations.

An audit starts in `cli.py`, which passes arguments to `api.audit()`. The API builds a checker specification and
`AuditConfig`, then calls `pipeline.run_audit()`. A Hub target first goes through `envs.py` to choose its interpreter.
The pipeline opens a `RunnerPool`; each `worker.py` process builds an adapter, loads tasks and returns scores.
The runner enforces per-call time and memory limits and turns worker outcomes into verdicts.

The pipeline samples tasks, checks that the setting is supported and calls `contract.infer_contract()`. Reference
calls establish accepted formats and full scores; wrong-answer controls select correctness components of a composite
reward. After the contract gate and determinism check, only rows whose reference passes receive probes.

The registry supplies probes in tier order. Their generators use `ProbeContext` and the contract to write replies;
the workers score them, and `metrics.Tally` collects observations into findings and summary rates. A probe may yield
acceptance or rejection findings and robustness findings. Dataset checks cover every row independently.
`metrics.collect_defects()` groups the evidence according to `defects.py`. `confirm.confirm()` sends each high defect's available
example five more times and, where eligible, removes sentences and words while acceptance persists. Finally,
`report.py` renders the result and optional report files; `pipeline.exit_code()` supplies the command's status.

## Add a probe

Source paths below are relative to `src/rewardaudit/`; other paths are from the repository root.

1. Append a generator and `Probe(...)` to its family's list: `probes/null.py` (N),
   `candidates.py` (E), `leak.py` (L), `hedge.py` (O), `mutate.py` (M), `rewrite.py` (Q),
   `reference.py` (G), or `robustness.py` (P), all under `probes/`. Structured M/Q probes
   use `probes/structured.py`'s `PROBES_M` / `PROBES_Q`. `probes/registry.py` joins the lists.
2. Use `probes/base.py`'s `Probe` and `ProbeContext`; keep unsuitable tasks out with
   `applies`. Add the id to exactly one defect in `defects.py`.
3. Touch `metrics.py` only for a new severity rule or note; categories already follow
   the probe's family and label. Prompt rules live in `prompt.py`, equivalence in `equiv.py`.
4. Update `tests/test_probe_labels.py`'s id list and applicable fixture tasks. Add a
   behaviour test in the family's test file: check replies, inapplicable tasks, and
   that non-solving variants do not contain the reference as a correct answer.
   `tests/test_defects.py` checks that every probe belongs to exactly one defect.
5. Update counts in `README.md`, `README.zh-CN.md` and `docs/guide.md`, and the changelog. Regenerate
   `docs/reference.md` using `explain.render_reference()`; the freshness test in
   `tests/test_report_cli.py` gives the command if the page differs.

## Add an adapter

1. Add `adapters/<name>_adapter.py`, subclassing `adapters/base.py`'s `Adapter`.
   Implement `all_tasks()` and `score_raw()`; set `name`, `meta` and `max_score()` as needed.
2. Register the spec's `kind` in `adapters/__init__.py`'s `build_adapter`.
3. Add target routing in `api.py` and target help in `cli.py`.
4. Check `runner.py`'s task loading for non-function kinds: every worker loads tasks.
5. Check `adapters/function_adapter.py`'s `reference_aliases` and `reference_is_answer`,
   used by `pipeline.py` for G2/G3; decide what the new kind's reference means.
6. Supply metadata read by `gate.py`'s `classify_meta` so unsupported tasks are declined.
7. Check `report.py`'s scoring-path, framework and style descriptions and its `repro.py`
   generation, including the Hub-specific environment setup.
8. If a separate interpreter is needed, check `envs.py` and `pipeline.py`'s `AuditConfig.python`.
9. Add adapter tests alongside `tests/test_function_adapter.py` and
   `tests/test_verifiers_adapter.py`, usage in `docs/guide.md`, and a changelog line.
