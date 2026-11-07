# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What stays the same across versions is listed in
the [usage guide](docs/guide.md#what-stays-the-same-across-versions).

## 0.1.0 - Unreleased

First public version.

### Added

- `rewardaudit audit`: tests a checker with 55 kinds of reply in 7 families, without a model, and answers five questions:
  whether it accepts non-answers, wrong answers and hedged answers, whether it rejects right answers, and whether it runs
  reliably. Each has a rate with a 95% interval, and each problem is a finding with an example.
- Sampling: `--n` rows drawn at random from the whole dataset, or `--n all`.
- Reply formats: finds which formats the checker accepts, and takes one of its own with `--format`.
- Scores with several parts: judges acceptance on the parts that separate right answers from wrong ones.
- Stops with a one-line reason when the numbers would mislead, such as reference answers that never earn full score.
- `--out DIR` writes `report.md`, `report.json` and `repro.py`, which re-scores the reply behind each finding.
- Exit codes and `--fail-on` for CI.
- `from rewardaudit import audit`, which takes functions, files, rows in memory and DataFrames.
- Scoring functions matched by parameter name, including the verl and TRL signatures, and `verifiers` environments.
- Five checks on the dataset itself, such as empty answers and duplicate prompts.
- `rewardaudit demo` and `rewardaudit probes`.
