# Contributing

## Setup and checks

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

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
or change what an existing id means. See the [stability policy](docs/guide.md#what-stays-the-same-across-versions).
Every change a user can notice needs a test and a line under `Unreleased` in `CHANGELOG.md`.
Keep special cases taken from real checkers: they make judgements more accurate.

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
5. Update counts in `README.md` and `docs/guide.md`, and the changelog. Regenerate
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
7. Check `report.py`'s scoring-path, framework and style descriptions and `repro.py`
   generation, including the Hub-specific environment setup.
8. If a separate interpreter is needed, check `envs.py` and `pipeline.py`'s `AuditConfig.python`.
9. Add adapter tests alongside `tests/test_function_adapter.py` and
   `tests/test_verifiers_adapter.py`, usage in `docs/guide.md`, and a changelog line.
