# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What stays the same across versions is listed in
the [usage guide](docs/guide.md#what-stays-the-same-across-versions).

## 0.3.0 - 2026-11-11

First public release.

### Added

- `rewardaudit audit` and Python `audit()` test checkers without a model, using 79 probes in eight families,
  checks S1 and S2, and a catalogue of defects D1–D22.
- Reports describe accepted non-solving, wrong and hedged answers, rejected right answers, partial credit for wrong
  answers, and robustness, with rates, 95% intervals and examples.
- Accepted reply formats and correctness components of composite rewards are inferred from reference calls and
  wrong-answer controls. Unsupported settings stop with a reason.
- High defects include five repeat scores and, where possible, a shortened reply still accepted.
- Scoring functions work by parameter name, including verl and TRL signatures; verifiers environments run their own
  scoring path, with Hub packages installed into separate environments.
- Dataset input supports JSON Lines, JSON, CSV, parquet, Hugging Face datasets and Python rows or DataFrames;
  nested columns, separate references and alternative accepted answers are supported.
- Empty references are omitted before sampling and reported; dataset checks cover all rows.
- `--format`, `--answer-field`, `--probes`, `--timeout` and `--budget` control the audit; reports name probes left out.
- `--calls` saves reference and probe calls; `rewardaudit compare` and Python `compare()` compare two calls files.
- `--out` saves Markdown, JSON and a reproduction script; reports record checker imports and dependency versions.
- Exit codes and `--fail-on` support CI; `demo`, id lookup and a generated reference explain the results.
- The guide, quickstart, Python API example and runnable verl, TRL and QA examples document use.
- Contributor instructions, issue template, citation metadata, typing marker and a CI workflow ship with the project.
