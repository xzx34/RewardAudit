# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What stays the same across versions is listed in
the [usage guide](docs/guide.md#what-stays-the-same-across-versions).

## 0.3.0 - 2026-11-11

### Added

- High defects show five repeat scores and, when possible, the shortest reply still accepted; confirmation is saved in `report.json`.
- `--answer-field` and `audit(answer_field=True)` state that the checker receives an extracted answer.
- Reports record the checker's imported packages and versions in `environment`, also available as `result.environment`.
- Empty reference answers in dataset files are left out before sampling, with their count reported.
- G2 tests other accepted answers supplied through `--reference-col`.
- G3 and D22 report when rewriting the reference changes the score of the same right reply.
- O16 tests paired option letters; M26 reverses relation signs; M27 tests small absolute tolerances; M28 changes parts of structured answers.
- E10 tests all candidates named in the prompt as one answer; there are now 79 probes in eight families and 22 defects.
- `--budget` and `audit(budget_s=...)` limit audit time and report probes left out.
- `--calls` and `audit(calls=...)` save reference and probe calls as JSON Lines.
- `rewardaudit compare A B` and Python `compare(a, b)` compare acceptance, replies and rates from two calls files.
- The guide explains equivalence dependencies, checker interfaces, stopped audits, Python callbacks, stability and coverage limits.
- A generated reference describes every probe, defect and other reported id.
- Runnable ten-row examples cover verl-style numeric rewards, TRL batch rewards and several accepted QA answers.
- Contributor instructions, an issue template, citation metadata and a typing marker; CI checks macOS, default installs and wheel examples.

### Changed

- Removed M11, M12, M13, M17 and Q31 because no real checker showed their defects; their ids are retired and are not reused.
- Q27 no longer removes accents and adds an English article only when the prompt contains the word `the`; its other variants stay unchanged.
- Q28 calls on prompts requiring the answer at the end count toward failed calls and scoring time, but no longer toward right-answer rejection findings.
- `rewardaudit probes <id>` explains defects, findings and evidence ids as well as probes; reports point to lookup.
- Command help describes positional arguments in the command description.
- Reports name the default `train` split when a Hugging Face dataset name omits the split.

### Fixed

- Shortening a numeric hedge keeps both answers without the optional math packages, whichever answer comes first.
- Checker imports exclude packages loaded by the tool's dataset equivalence checks.
- Advice for answer-field checkers no longer asks them to accept surrounding prose.
- A non-score return names the unsupported value and supported score types instead of saying the checker raised an error.
- The result says when too few calls prevented findings from being rated and suggests more rows.
- README and guide output, call counts and exit codes match current behaviour; README links work on PyPI.
- Finding examples come from the variant whose rate is reported.
- Markdown reports and `--full` use singular `call` and `row` for a count of one.
- Numeric rewrites leave pass keys and passcodes unchanged.
- TRL completion annotations recognise `Dict` as well as `dict` for chat messages.
- Q8 adds a trailing decimal zero before a percent sign.
- Empty answer lists are omitted even when a nested column selects inside an item.
- Prose probes respect instructions placing the answer first, last or inside thinking and answer blocks.
- O14 and G2 respect requests for an option letter rather than option text.
- M21 attaches letters only to Latin-text answers, excluding sets of option letters and whole prompt lines.
- Sign probes leave numbered option labels unchanged; G3 leaves single-letter references without a period.
- Q1 avoids inner spaces where a format repeats or quotes the answer; case rewrites leave cipher prompts alone.
- Period and word rewrites leave function calls with keyword arguments unchanged.
- Wrong numbers paired with right ones lie beyond common relative tolerances and avoid percentage equivalents.
- P2 omits nested formats that would leave the plain wrong answer unchanged.
- Timing findings require evidence from more than one long call before receiving medium severity.
- Stopped environment audits give advice that applies to their dataset, rubric and verifiers API.
- M6 preserves line numbers written as `*1*:`.
- Hub package versions use separate environments, so `owner/name@version` audits the requested version.
- Reused environments report their pinned verifiers version, and generated reproductions keep that pin.

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
- Exit code 1 now means a high defect. Before, any high or medium finding did. `--fail-on` still sets a limit on any rate.
- A checker that compares the whole reply with the reference answer is reported as such: replies with text around the answer
  stay out of Rejects right answers and their findings stay low.
- The check that a repeated call gets the same score also sends a reply with no answer, since a checker may guess when it finds none.

### Fixed

- A fixed answer that happens to be right on a row now stays out of the rate on checkers that read the bare reply.
- A reference answer with a percent sign is now recognized inside a reply.
- An option letter F or T is a letter where the prompt lists it, not false or true.
- A long answer with one digit changed no longer compares equal by similarity, so wrong answers on long LaTeX are sent.

## 0.1.0 - 2026-11-06

Initial version.

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
