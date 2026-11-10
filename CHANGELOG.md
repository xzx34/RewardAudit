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
  scoring a file of predictions does. Replies then write the answer bare wherever the checker accepts that as well as any
  other format, where a checker that also takes `$\boxed{...}$` would otherwise get that. A right answer with text around it
  stays out of Rejects right answers and its findings stay low, as when this is inferred from Q5 and Q6; without the option
  it is still inferred. The note at the top of the report then says it was stated.
- A line after the dataset checks, in the terminal and in `report.md`, names the third-party packages imported as the checker
  loaded and scored, with their versions, the first 8 by name, or `none`, such as
  `Checker imports: antlr4-python3-runtime 4.11.0, mpmath 1.3.0, sympy 1.14.0`. Packages imported to read the dataset, such
  as pyarrow or datasets, do not count, nor does cloudpickle, which carries a function over from Python. Another version of a
  package can change what a checker decides: under antlr4-python3-runtime 4.13, sympy's `parse_latex` raises an ImportError,
  and a checker that catches it rejects every answer it would have parsed. `report.json` has a new top-level key,
  `environment`, with `python`, `platform` and `packages` (every package, name to version), as the workers reported them;
  from Python it is `result.environment`. `schema_version` stays 1, since no existing key changed.
- Rows of a dataset file whose reference answer is empty (a blank string, `null`, or an empty list, as SQuAD 2.0 keeps its
  unanswerable questions) are left out before the draw, and the dataset checks say so: `100 rows with an empty reference were
  left out`, `empty_rows_left_out` in `dataset_issues`, low. Before, they were drawn with the rest, and a sample where half the
  reference answers were empty stopped the audit. The row counts in the report, `n_rows` among them, are of the rows left;
  `meta.empty_rows_left_out` in `report.json` has the number. A dataset whose reference answers are all empty stops with a
  message naming the column. An environment's own rows are not left out, and an empty answer there is still a dataset issue.
- G2, in a new family G: where the checker receives a list of accepted answers (`--reference-col`), each one other than the
  reference answer is sent in the accepted format, one per reply and up to 8 per row. One that loses the score is a finding
  under Rejects right answers, outside its rate, medium from one in ten; its note names some of them. D16 now reads "The
  dataset's own reference answer, or one of its other accepted answers, is treated like a wrong one." `rewardaudit probes G`
  lists G1 there too. 78 probes in all.
- `--budget`, such as `90s`, `10m`, `1h` or a number of seconds, and `audit(budget_s=...)` in seconds, limit the time of the
  whole audit, counted from its start. Once it has passed no further probe is sent; the probe whose calls are under way
  finishes, and the calls made count as usual. A line at the top of the report, in the terminal and in `report.md`, then
  names the probes that did not run: `Budget of 1m reached before ... (12 probes not run); the rows below are measured on
  fewer probes.` `report.json` lists them under `meta.probes_not_run`, a key that is there only then, and `report.md` gives
  the budget under How this was run. Under a budget the progress bar of the probes has no total. There is no limit by
  default.
- 4 new probes, 82 in all:
  - O16, under D6: on multiple-choice rows, the right letter with another option's, as `BC`, `B, C` and `B/C`. A checker
    that reads the first or the last letter accepts the replies where the right one stands there.
  - M26, under D1: where the reference answer holds a relation sign (`<`, `>`, `\le`, `\ge`, `\neq`, `≤`, `≥`), the
    sign turned around (`x < 3` for `x > 3`, `=` for `\neq`) and made strict or not strict (`x \le 3` for `x < 3`). A
    checker that keeps only the numbers, or drops what is not a letter or digit, accepts them. D1 now reads "A number with
    its sign flipped or its decimal point moved, or an answer with its relation sign turned around, compares equal to the
    reference answer."
  - M27, under D3: where the numeric reference answer is under 0.01 in size, zero among them, the answer plus 0.00005 and
    plus 0.00009, written in decimals, each only where that is more than a tenth of the answer, wider than any error M24
    sends. A checker with an absolute tolerance such as `abs(a - b) < 1e-4` accepts them and a relative tolerance does
    not; on zero, M24 sends nothing.
  - M28, under D20: where the reference answer is a list or a grid of 3 items or more, k of them replaced by wrong ones,
    from one up to all, at most 8 steps and closer together at both ends. Like M24 it stays out of the rate; its note gives
    the acceptance rate of each k, such as `accepted: k1 100%, k2 0%, all 0%`, and where some scores fall between nothing
    and full, each k's mean share of the score. Its severity is that of the reply with every item wrong, as a wrong answer.
- `--calls FILE`, and `audit(calls=...)`, write each call of the audit to a file, one JSON object a line and nothing else:
  `task_id`, `probe_id`, `variant`, `completion`, `kind`, `score` and `accepted`, whether the audit counted the reply as
  accepted, the last two `null` for a call with no score. The calls with the reference answer have `probe_id` `reference`
  and the format, or `wrong_answer`, as `variant`, and are written even when the audit stops after them. Calls that only
  check other calls are left out: the repeats that see whether a score stays the same, those that confirm a high defect, and
  the reference answer put into a reply that scored above it. Each probe's calls are written as they come in, so a large
  sample is not held in memory.
- `rewardaudit compare A B`, and `compare(a, b)` from Python, line up two such files by row, probe and variant and show where
  the two checkers decide differently: how many calls line up and the share they agree on, the calls in one file only, each
  probe where they differ with how often each side accepted what the other did not and up to three examples of each, and
  the rates of the summary for both sides, worked out from the calls alone and so not the audit's own. Where the two audits
  wrote their replies in different formats, it says on how many calls and the examples show both replies. Two files with no
  `task_id` in common exit with 2.
- G3, under a new defect D22, Reference formatting changes the verdict: the reply is the reference answer as it passed, in
  the accepted format, while the checker gets the reference answer written another way, one way per call and each only where
  it applies: spaces at its operators and commas taken out or put in (`x + 1` and `x+1`), `\frac` and `\dfrac` swapped, a
  period at the end, words in `\text{}` under a `\boxed{}` format, a percentage as a decimal and back, the unit taken off or
  the one the question names added, thousands separators added to a quantity or taken out. A checker that loses the score
  there scores the rows of a dataset that does not write its answers uniformly by how they are written. Like G2 it is a
  finding under Rejects right answers, outside its rate, medium at most, from one in ten. Its rate is over all its calls,
  since some writings apply on nearly every row and others on few; its note gives the rate of each writing, such as
  `rejected: period 100%, commas 0%`. It is sent where the checker gets the answer itself or a
  `reference_template` with `{answer}` in it: not with `--reference-col` or `ground_truth_json`, not in a verifiers
  environment, and not on a reference answer that is JSON. In `report.json` the example of a G3 finding has a new key,
  `reference`, the writing the checker got; `report.md` shows it next to the row's own, and `repro.py` scores the reply
  against it. The calls file has no such key. 83 probes in all.

### Fixed

- Where a probe sends distinct kinds of reply and its finding gives the rate of the kind that went that way most often, the
  example is now one of that kind. Before, it was taken from the first rows that went that way, and could be of another.
- `report.md` writes `1 call` and `1 row`, not `1 calls` and `1 rows`.
- A question that asks for a pass key or a passcode asks for a string of digits, not a quantity, as one that names a version
  or an index does: Q8 no longer sends `47,869` or `+47869` for the key `47869`, nor does G3 write it as the reference.
- A TRL-style reward whose `completions` is annotated `list[Dict[str, str]]` gets chat messages, as one annotated with
  `dict` does. Before, it was taken to want plain strings and failed on every call; the annotation is now read without
  regard to case.
- Q8 writes the trailing zero of a percentage on its number: `21.460%` for `21.46%`. Before, it was put after the percent
  sign, `21.46%0`, which is not the same number, and a checker that rejected it was counted as rejecting a right answer.
- A row whose list of answers is empty is left out under a column name that reaches below an item of the list, such as
  `answers.0.0` for answers kept as `[answer, conditions]` pairs, as it already was under `answers.text.0`. Before, the
  audit stopped, saying the row had no such column.
- Probes that put text before or after a right answer are no longer sent where the prompt says where the answer goes, since
  a checker that refuses such a reply follows the prompt. Where it asks for the answer first ("State the amount first"),
  Q5, Q21, Q22, Q29 and O4 are left out; where it asks for the answer at the end ("The last line of your response should
  be of the form ...", "... at the end of your response", "Your response should end with ..."), Q6 (Q28 is still sent,
  since its longest reply is the one call that times the checker on text after the answer); and where it writes out the
  reply as a thinking block directly followed by an answer block (`<think> ... </think><answer> ... </answer>`) and the
  checker takes that format, Q5, Q6, Q21, Q28 and Q29, whose text would fall outside both blocks.
  "Answer with the option's letter from the given choices directly" now counts as asking for only the answer.
- Where the prompt asks for an option's letter, O14 sends only a wrong letter with the right option's text, and G2 only the
  accepted answers that are letters. Before, a checker that read the right letter followed by another option's text was
  counted as accepting a hedge, and one that refused an option's text where the prompt asked for the letter as rejecting a
  right answer.
- M21 attaches letters and words only to an answer in Latin letters, and not to a set of option letters (`ACD`) or to an
  answer that is a whole line of the prompt, such as an option line. Before, `ACDx`, `$174x`, a group of numbers or a word
  in another script with a word or letter attached, which still name the same answer, were counted as wrong answers.
- M2 and M22 are not sent where the reference answer is the number of one of the choices the prompt lists on numbered lines
  (`2 - producer's desk`): a minus sign before it names no other choice.
- G3 no longer writes a reference answer that is a single letter with a period (`A.`): multiple-choice answer keys store
  an option letter bare, so a checker that compares the letter with the stored string never meets `A.` outside the audit.
- Q1 puts no spaces inside a format that writes the answer more than once or between quotes, where they change the answer
  itself; Q2 and Q27 leave prompts about a cipher (to encrypt or decrypt) alone, where the case of each letter is part of
  the answer; Q3's period and Q27 leave a function call with keyword arguments alone, where they make other code.
- The wrong number the O probes pair with a numeric reference answer, and that P2 hides characters in, is moved away from
  zero by at least 1 and by at least its own size, 2x for 30 and x + 1 for 0.5, and is never a hundred times or a
  hundredth of the answer. Before, it was x + 1, which a relative tolerance takes for the reference answer when x is
  large (101 is within 5% of 100), so a hedge between two answers the checker counts as right was reported as a hedge
  accepted. A loose tolerance is still measured where it belongs, by M1, M24 and M27, and an allowance for percentages by
  M25.
- P2 no longer sends twenty layers of a format that adds nothing to the answer, which are the plain wrong answer.
- A finding that scoring time grows with the length of the reply, measured on one long reply, is low with "too few calls to
  rate", as other findings resting on one row are. Before, it was medium.

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
