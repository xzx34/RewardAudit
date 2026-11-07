# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What stays the same across versions is listed in
the [usage guide](docs/guide.md#what-stays-the-same-across-versions).

## Unreleased

### Added

- Each high defect is confirmed in a line under it, in the terminal and in `report.md`: its example is sent 5 more times, and
  a wrong answer, hedged answer or non-answer accepted each time is cut down, sentences and then words taken out, to the
  shortest reply still accepted, short of the reference answer on its own. In `report.json` this is
  `defects[].confirmation`: `resends`, the scores of those calls, and `shortest_accepted`, with `completion` and `score`, left
  out when nothing could be taken out. These calls count in no rate or finding; the calls in the top line include them.
  The progress bar shows them as a stage of its own, `confirming defects`.
- `--answer-field`, and `audit(answer_field=True)`, state that the checker takes the reply to be its answer field, as a script
  scoring a file of predictions does. A right answer with text around it then stays out of Rejects right answers and its
  findings stay low, as when this is inferred from Q5 and Q6; without the option it is still inferred. The note at the top of
  the report then says it was stated.
- A line after the dataset checks, in the terminal and in `report.md`, names the third-party packages imported as the checker
  loaded and scored, with their versions, the first 8 by name, or `none`, such as
  `Checker imports: antlr4-python3-runtime 4.11.0, mpmath 1.3.0, sympy 1.14.0`. Packages imported to read the dataset, such
  as pyarrow or datasets, do not count, nor does cloudpickle, which carries a function over from Python. Another version of a
  package can change what a checker decides: under antlr4-python3-runtime 4.13, sympy's `parse_latex` raises an ImportError,
  and a checker that catches it rejects every answer it would have parsed. `report.json` has a new top-level key,
  `environment`, with `python`, `platform` and `packages` (every package, name to version), as the workers reported them;
  from Python it is `result.environment`. `schema_version` stays 1, since no existing key changed.

## 0.2.0 - 2026-11-07

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
- Exit code 1 now means a high defect. Before, any high or medium finding did, which on real checkers was nearly always: of 56,
  51 had one and 20 have a high defect. `--fail-on` still sets a limit on any rate.
- A checker that compares the whole reply with the reference answer is reported as such: replies with text around the answer
  stay out of Rejects right answers and their findings stay low.
- The check that a repeated call gets the same score also sends a reply with no answer, since a checker may guess when it finds none.

### Fixed

- A fixed answer that happens to be right on a row now stays out of the rate on checkers that read the bare reply.
- A reference answer with a percent sign is now recognized inside a reply.
- An option letter F or T is a letter where the prompt lists it, not false or true.
- A long answer with one digit changed no longer compares equal by similarity, so wrong answers on long LaTeX are sent.

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
