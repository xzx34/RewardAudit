# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What stays the same across versions is listed in
the [usage guide](docs/guide.md#what-stays-the-same-across-versions).

## Unreleased

### Added

- Defects: the report now says what the checker does across probes, from a catalog of 21 kinds of defect, each with the probes
  behind it, its rates and an example. They are in `result.defects` and under `defects` in `report.json`.
- A sixth question, `wrong_partial`: how often a definite error earns more than nothing. `--fail-on` takes it.
- 22 new probes, 77 in all:
  - how answers are compared: M21 characters attached to the answer, M22 its opposite with a Unicode minus, M23 another unit,
    M24 a number off by 0.1% to 10%, M25 a hundred times the number or a hundredth, L5 the most common answer, L6 the prompt's
    last number;
  - other spellings: Q23 numbers, Q24 units, Q25 percentages, Q26 dates, Q27 words, Q30 option texts, Q31 LaTeX markup;
  - which answer is read: O12 two answers in two accepted formats, O13 only in the thinking block, O14 an option's letter with
    another's text, O15 a range around the answer, Q28 and Q29 long text around the answer;
  - robustness: P16 a reply's text run as code, P17 a million characters. P2 also sends invisible characters and a NUL byte
    inside a wrong answer.
- Findings for a checker that returns nan or inf, and for scoring much slower on long replies.

### Changed

- A finding stays low below ten scored calls or on one row, with the note "too few calls to rate". Failed calls and rejected
  reference answers need two rows.
- The terminal lists defects. The findings, one per probe, are under `--full` and in `report.md`.
- Exit code 1 now means a high or medium defect, which includes wrong answers paid in part.

### Fixed

- A fixed answer that happens to be right on a row now stays out of the rate on checkers that read the bare reply.
- A reference answer with a percent sign is now recognized inside a reply.

## 0.1.0 - 2026-11-06

First public version.

### Added

- `rewardaudit audit`: tests a checker with 55 kinds of reply in 7 families, without a model, and answers five questions:
  whether it accepts non-answers, wrong answers and hedged answers, whether it rejects right answers, and whether it runs
  reliably. Each has a rate with a 95% interval, and each problem is a finding with an example.
- Sampling: `--n` rows drawn at random from the whole dataset, or `--n all`.
- Reply formats: finds which formats the checker accepts, and takes one of its own with `--format`.
- Several accepted answers: `--answer-col answers.text.0 --reference-col answers.text`.
- Scores with several parts: judges acceptance on the parts that separate right answers from wrong ones.
- Stops with a one-line reason when the numbers would mislead, such as reference answers that never earn full score.
- `--out DIR` writes `report.md`, `report.json` and `repro.py`, which re-scores the reply behind each finding.
- Exit codes and `--fail-on` for CI.
- `from rewardaudit import audit`, which takes functions, files, rows in memory and DataFrames.
- Scoring functions matched by parameter name, including the verl and TRL signatures, and `verifiers` environments.
- Five checks on the dataset itself, such as empty answers and duplicate prompts.
- `rewardaudit demo` and `rewardaudit probes`.
