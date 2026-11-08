# Usage guide

1. [What you need](#what-you-need)
2. [Prepare the dataset](#prepare-the-dataset)
3. [Connect your checker](#connect-your-checker)
4. [Choose how many rows](#choose-how-many-rows)
5. [Read the result](#read-the-result)
6. [When the audit stops](#when-the-audit-stops)
7. [Audit a verifiers environment](#audit-a-verifiers-environment)
8. [Compare two checkers](#compare-two-checkers)
9. [Use it in CI](#use-it-in-ci)
10. [Use it from Python](#use-it-from-python)
11. [What stays the same across versions](#what-stays-the-same-across-versions)
12. [Troubleshooting](#troubleshooting)

## What you need

- **A checker:** a Python function that takes a model's reply and returns a score. It has to give the same score to the same
  reply and must not call a model.
- **A dataset:** rows with a prompt and a reference answer, the final answer a right reply should give. A worked solution is
  not one; see [below](#the-reference-answer-is-not-what-the-checker-expects).

Try `rewardaudit demo` first if you want to see the output without any files of yours.

## Prepare the dataset

| Source | How to pass it |
|---|---|
| JSON Lines | `--dataset data.jsonl` |
| JSON (a list of rows, or an object with a `data`, `rows`, `examples` or `tasks` list) | `--dataset data.json` |
| CSV | `--dataset data.csv` |
| Parquet (needs `rewardaudit[data]`) | `--dataset data.parquet` |
| Hugging Face (needs `rewardaudit[data]`) | `--dataset hf:openai/gsm8k:main:test`, that is `hf:name:config:split` |

**Columns.** The prompt is taken from the first column present among `prompt`, `question`, `problem`, `input`, `query`,
`instruction`, `text`. The reference answer is taken from `answer`, `solution`, `ground_truth`, `target`, `label`,
`final_answer`, `output`, `reference`, `reward_model.ground_truth`. If yours are named differently:

```bash
rewardaudit audit reward.py --dataset data.jsonl --prompt-col q --answer-col a
```

Dotted names reach into nested values (`reward_model.ground_truth`, `answers.text.0`). A prompt can be a string or a list of chat messages. An
answer that is a dict or list is passed on as it is. An `id` column names the rows in reports, if its values are unique.

## Connect your checker

Point rewardaudit at a file: `reward.py` looks for a function named `score`, `compute_score`, `reward`, `reward_fn`,
`reward_func`, `accuracy_reward` or `verify`; `reward.py:my_function` names one.

Parameters are matched by name, so these work as they are:

```python
def score(completion, answer): ...                       # also response, prediction, gold, target, label, ...
def score(prompt, completion, answer): ...
def compute_score(solution_str, ground_truth): ...       # verl; data_source and extra_info are filled in too
def reward(prompts, completions, **columns): ...         # TRL and open-r1; returns a list, other columns are passed
```

Parameters with other names get the reply and then the reference by position. Parameters named `prompt`, `messages`, `info`,
`metadata`, `data_source` or `extra_info` get those values from the row.

A function can return a number, a bool, a one-element list or tensor, or a dict. From a dict the main value is taken from
`score`, `overall`, `reward`, `rewards`, `total` or `accuracy`, and other numbers count as sub-scores. A dict with none of those
keys counts all its numbers, bools included, as parts of a total, so return one number if that is not what they are.

**Anything else: wrap it.** rewardaudit needs only a function that takes a reply and a reference answer and returns a score,
so a few lines in your own file connect any benchmark, including a checker that is a class or a method. Call your checker from
it without changing what the checker decides:

```python
# audit_me.py, next to the code it imports
from rewards import Grader

grader = Grader(strict=True)

def score(completion, answer):
    return grader.grade(response=completion, expected=answer)["points"]
```

### The reply format

rewardaudit first finds out how the checker wants a reply written: it sends each reference answer as it is, in `\boxed{}`, in
`<answer>` tags, after `Final Answer:`, after `####`, and in other common forms, along with any tag or answer line the prompts
ask for. Probes then build their replies on a form the checker accepts; some surround it with text or put it inside a sentence,
since that is what they test. If yours is none of these, name it:

```bash
rewardaudit audit reward.py --dataset data.jsonl --format '<solution>{answer}</solution>'
```

A format that is a mark and then the answer, such as `RESULT>> {answer}`, is mostly written on a line of its own.

**A checker that takes only the answer.** Some checkers are given an answer already extracted from the reply, not the reply
itself: the official scripts of DROP, SQuAD, TAT-QA and WikiTableQuestions score a file of predictions, one answer each. Add
`--answer-field` for such a checker (`answer_field=True` from Python): replies then write the answer bare wherever the
checker accepts that as well as any other format, and a right answer with text around it stays out of Rejects right
answers, its findings low. Without it, rewardaudit infers this from Q5 and Q6, when the bare answer passes and text before
or after it mostly loses the score; where it infers right, the result is the same.

### Several accepted answers

Many QA benchmarks accept any of several answers and keep them in a list. rewardaudit writes each reply from one answer and
compares scores, so name one for the replies and hand the whole list to the checker:

```bash
rewardaudit audit reward.py --dataset data.jsonl --answer-col answers.text.0 --reference-col answers.text
```

A number in a column name picks an item of a list, and a row whose list is empty has no answer and is left out. Your
function's reference parameter gets the whole list. Replies are written from the answer you name, and G2 sends each of the
others as well, one per reply and up to 8 per row, since a checker given the list should accept any of them.

### The reference answer is not what the checker expects

Some checkers want the reference in a particular form. Wrap them as above, for example to add `\boxed{}`, or to parse JSON.
If your dataset holds a worked solution, extract the final answer into a column first: rewardaudit rewrites the reference
answer to build its replies, so it has to be the answer itself.

G3 turns this around: it sends the reference answer as it passed, while the checker gets the reference answer written
another way, one way per call, such as `1,200` for `1200`, `\dfrac` for `\frac` or a period at the end. A checker that then
loses the score gives a dataset's rows different scores for how their answers are written (D22). G3 is sent only where the
checker gets the answer itself, or a `reference_template` with `{answer}` in it, which then holds the other writing: not
with `--reference-col` or `ground_truth_json`, nor in a verifiers environment, nor on a reference answer that is JSON. Its
finding's rate is over all its calls, and its note gives the rate of each writing. The example of its finding has
`reference`, the writing the checker got; `report.md` shows it and `repro.py` scores against it.

## Choose how many rows

`--n` is yours to choose. rewardaudit always reads the whole dataset and draws that many rows at random, without replacement,
using `--seed` (0 by default). The default is 100; `--n all` uses every row. The report states how many rows were drawn from
how many. A row whose reference answer is empty is left out first, since a reply to it could not be scored against anything.

**What more rows give you.** Each row costs about 90 checker calls. A behaviour that affects 5% of rows shows up in a sample
of 100 with a probability of 99.4%, and in a sample of 30 with 79%. Larger samples narrow the intervals and reach rarer rows.

**Time.** Measured on a laptop: every row of GSM8K (7,473 rows, 680,201 calls) with a regular-expression checker took 40 seconds
and about 400 MB with 4 workers. A slower checker sets the pace: at 10 ms per call, a thousand rows take about 15 minutes on one
worker. `--workers` runs that many processes (4 by default, or your CPU count if lower). A progress bar shows the time left.
`--probes N,O,M` runs only those families.

**A slow checker.** `--budget 10m` (or `90s`, `1h`; `budget_s=600` from Python) limits the whole audit, counted from its
start. Once that time has passed no further probe is sent: the probe whose calls are under way finishes, so the audit can
run over by about as long as one probe takes, and every call made counts as usual. A line at the top of the report then
names the probes that did not run, and `report.json` lists them under `meta.probes_not_run`. The summary rows are measured
on the probes that ran, so a probe left out can hide a finding; to keep every probe, draw fewer rows instead. Under a budget
the progress bar of the probes has no total, since counting their calls ahead means building every reply first.

## Read the result

```
Result: problems found (2 high, 3 medium)

  Accepts non-answers         no     0%  [0%, 14%]
  Accepts wrong answers       yes   14%  [11%, 17%]
  Accepts hedged answers      yes   26%  [23%, 30%]
  Rejects right answers       yes   19%  [16%, 22%]
  Pays wrong answers in part  no     0%  [0%, 1%]
  Runs reliably               yes  no errors, timeouts or crashes in 4,622 calls

Defects (6)
  HIGH    D1  Signs and punctuation ignored  M22 100%, M3 4%
              \boxed{−34}
              Sent 5 more times: 1.0 each.
  HIGH    D6  Hedged answers accepted  O11 100%
              The answer is either 35 or 34.
              Sent 5 more times: 1.0 each. Shortest reply still accepted (1.0):
              35 34.
  MEDIUM  D3  Numeric tolerance too wide  M24 75%, M25 38%
              \boxed{34.34}
  ...
```

Each row answers a question about the checker. On the first five rows, `yes` means a high or medium problem. **Runs reliably**
reads the other way round: `yes` is good.

The percentage is the share of calls that went that way, with a 95% interval. These are shares of the replies rewardaudit sends,
not of what a model would write. They are for comparing runs and for `--fail-on`. A low rate can sit next to a high finding,
since one kind of reply may be accepted every time and the others never, so read the defects to see what happens. The calls
in the top line also count the wrong answers and repeats used before the probes, and the calls that confirm a high defect
after them, so that number is a little higher.

- **Accepts non-answers:** the best reply built without reading the reference answer. It is chosen on part of the rows and
  measured on the others, so the choice does not inflate the rate. A reply that always says the most common answer is right
  on some rows by chance, so this counts only above that level.
- **Accepts wrong answers:** the reference answer with a definite error.
- **Accepts hedged answers:** a reply stating two answers, of which one is right, or stating the right one and taking it back.
- **Rejects right answers:** the reference answer written another way, still correct.
- **Pays wrong answers in part:** the reference answer with a definite error, earning more than nothing and less than the
  reference answer. The row says `yes` from one call in ten, when they earn a fifth of its score or more on average: less is
  usually a bonus for the format. A score such as F1 does this by design; whether that is what you want is your call.
- **Runs reliably:** errors, timeouts, crashes and network attempts, and whether repeating a call gives the same score, for the
  reference answer and for a reply with no answer, since a checker may guess when it finds none.

**Defects.** A defect says what the checker does, across the probes that show it: each line has its severity, its name, and the
probes behind it with their rates. A defect takes the highest severity of its evidence. The names describe behaviour, since the
checker's code is never read: a reply `-75` accepted for `75` shows that the sign was ignored, whether the checker strips it or
looks for a substring. Under a high defect, a line confirms its example: the scores of 5 more calls with it, and for a wrong
answer, hedged answer or non-answer accepted each time, the shortest reply still accepted when sentences and then words are
taken out of it, stopping short of the reference answer on its own, which it is right to accept. In `report.json` this is the
defect's `confirmation`, with `resends` and, when something could be taken out, `shortest_accepted` with its
`completion` and `score`; these calls count in no rate.

| Id | Defect | Evidence |
|---|---|---|
| D1 | Signs and punctuation ignored | M2, M3, M22, M26 |
| D2 | Substring match | M21, M18, E1, E5, O6 |
| D3 | Numeric tolerance too wide | M24, M25, M1, M27 |
| D4 | One of several answers is read | O5, O12, O4, E6 |
| D5 | Reads the thinking block | O10, O13 |
| D6 | Hedged answers accepted | O7, O11, O14, O15, O16, E2, E3, E8 |
| D7 | Negation ignored | O3 |
| D8 | Only a fixed-length window is read | Q28, Q29, when 100 characters of text keep the score and more lose it |
| D9 | Equivalent rewrites rejected | the other Q probes, and Q28 and Q29 when even 100 characters lose the score |
| D10 | Copying the prompt pays | L1, L2, L3, L4, L6, E9 |
| D11 | A fixed answer pays | L5 and the N family, above what a fixed answer gets by chance |
| D12 | Reply text run as code | P16, when the reply reaches a Python import |
| D13 | Malformed or extreme replies mishandled | P1, P2, P5, P15, P17, any probe's errors, timeouts and crashes, scores that are not numbers, and scoring much slower on long replies |
| D14 | Wrong answers paid in part | the row of that name |
| D15 | Same input, different scores | S1, the same reply sent again |
| D16 | Reference answer rejected | G1, G2 |
| D17 | Units ignored | M23 |
| D18 | Score does not follow the answer | any wrong answer earning the reference answer's score, or a reply that should earn nothing scoring above it |
| D19 | Close answers accepted | M5, M6, M7 |
| D20 | Parts of a structured answer ignored | M8, M9, M10, M11, M12, M13, M17, M28 |
| D21 | A different answer accepted | M16, M19, M20 |
| D22 | Reference formatting changes the verdict | G3 |

**Findings.** Each finding is one probe, with an example reply and a rate; `--full` prints them, and `report.md` always has them.
Where a probe sends distinct kinds of reply, the rate is that of the kind that went that way most often, and the example is
one of that kind. Severity is a rule of thumb for what to look at first:

| Severity | Meaning |
|---|---|
| High | A reply's text reaching a Python import, on one call. Non-answers, wrong answers or hedged answers accepted on about half of calls or more, a number 5% off accepted on half of rows, or right answers losing the score just for the reasoning before them on one call in five or more. |
| Medium | Wrong answers accepted or right answers rejected on one call in ten or more, hedged answers accepted on one in five, non-answers accepted more often than chance would give, crashes and timeouts, scores that are not numbers, wrong answers paid in part as the row above says, a number 1% off accepted on half of rows, a right answer lost to 4k characters of text or fewer. |
| Low | The same less often, or about what chance would give, and errors the checker raised. Listed for completeness. |

**A checker that compares the whole reply.** If the bare answer passes and text around it mostly loses the score, or you
say so with `--answer-field`, the checker takes a reply to be its answer field, as a script scoring a file of predictions
does. The report says so at the top, the replies with text around the answer stay out of Rejects right answers, and their
findings stay low. Put the extraction step of your harness in front of the checker if you want those replies measured.

A few calls say little, so a finding stays low, with the note "too few calls to rate", below ten scored calls or on one row only.
Failed calls and rejected reference answers need two rows. A defect looks at its probes together: on a dataset with few numbers,
each probe on numbers has only a few calls, and between them they may have enough. Q23 stays low, since few checkers read a
number in words.

**A finding is about behaviour.** rewardaudit does not know what you intended. If your task demands an exact format, then
`34.0` being rejected may be right. Read the example, and run `repro.py` if you want to re-score it.

**Composite scores.** If a checker returns sub-scores (correctness plus format, say), rewardaudit works out which of them
separate right answers from wrong ones and judges acceptance on those. The report names them. A reply that earns only the
format credit does not count as accepted.

**Saving the report.** `--out DIR` writes:

- `report.md`: the summary, the defects, every finding with an example, what to fix, accepted formats, the dataset checks, how it was run.
- `report.json`: the same results for scripts: `defects`, `metrics.findings`, `metrics.head` for the summary rows, and
  `metrics.probes` for the rate of every probe.
- `repro.py`: re-scores the reply behind each finding (not written for a function passed from Python).

**Probes.** `rewardaudit probes` lists all 83; `rewardaudit probes Q6` shows what one sends and which row it counts toward. They come in families: N (empty and
placeholder replies), E (many candidates in one reply), L (copying the prompt, fixed answers), O (two answers, which one is read),
M (definite errors), Q (right answers rewritten), G (the row's other accepted answers, and its reference answer written another
way; G1, the finding of a reference answer rejected in every format, is listed with them), P (malformed and extreme input, a
reply run as code).

**Dataset checks.** Rows whose reference answer is empty (blank, or an empty list, as SQuAD 2.0 keeps its unanswerable
questions) are left out before the draw, and a line says how many; the other row counts in the report are of the rows left.
Then the checks: empty answers, which an environment's own rows can still have, duplicate prompts, answers that appear in
the prompt, skewed answers, and multiple-choice answers that are not among the options. These read the data only, and they do
not affect the exit code. The line after them,
`Checker imports`, names the third-party packages imported as the checker loaded and scored, with their versions, the first
8 by name. Packages imported to read the dataset, such as pyarrow or datasets, do not count, nor does cloudpickle, which
carries a function over from Python. Another version of one can change what the checker decides, so these are the versions
that reproduce the result; `environment` in `report.json` lists them all, with the Python version and the platform.

## When the audit stops

rewardaudit does not fit every checker. What it cannot measure it does not audit: it stops, says why in a line, and exits with 2.

| Message says | Meaning | What to do |
|---|---|---|
| No reference answer earned full score | The checker rejects its own reference answers in every format tried | Check `--answer-col`; name the [reply format](#the-reply-format); [wrap the checker](#the-reference-answer-is-not-what-the-checker-expects) |
| Reference answers are empty, ids, labels or specifications | The column does not hold answers | Name the right column, or the checker needs no reference and is out of scope |
| Calls tried to use the network | The checker probably calls a model | Out of scope: network access is blocked while scoring |
| Calls timed out | The checker is slow | Raise `--timeout` |
| Multi-turn, tools, container, judges | The environment is not a single reply, or asks a model | Out of scope |

If a checker gives right and wrong answers the same score, the audit still finishes, says the rates could not be measured, and
exits with 2. Replies that score above the reference answer are still reported.

## Audit a verifiers environment

```bash
rewardaudit audit my-env --n 200           # installed in this Python
rewardaudit audit owner/name --n 200       # from the Prime Intellect Hub
rewardaudit audit owner/name@1.2.0         # a specific version
```

An environment brings its own dataset and rubric, so there is no `--dataset`. For `owner/name`, rewardaudit uses
[uv](https://docs.astral.sh/uv/) to install the package into its own venv under `~/.cache/rewardaudit/envs` (set
`REWARDAUDIT_HOME` to move it). It picks the verifiers version for the package's API.

**This runs someone else's code on your machine**, while installing and while scoring, with your user's permissions and
environment variables. Only audit packages you would run anyway, or run the tool in a container or a throwaway machine.

If an environment was written for an older verifiers, its reference answers may fail for no reason of its own. The message
then suggests a version:

```bash
rewardaudit audit owner/name --verifiers '<0.1.11'
```

The report says which version the results were measured on. Delete the environment's directory under
`~/.cache/rewardaudit/envs` to install it afresh.

## Compare two checkers

Two checkers for one dataset, such as a strict one and a lenient one, or two versions of one checker, before and after a
change, can be compared reply by reply. Audit each with `--calls`, on the same dataset with the same `--n` and `--seed`, so
that both draw the same rows, then compare the two files:

```bash
rewardaudit audit reward.py --dataset data.jsonl --calls v1.jsonl
rewardaudit audit reward_v2.py --dataset data.jsonl --calls v2.jsonl
rewardaudit compare v1.jsonl v2.jsonl
```

**The calls file.** One JSON object a line, and nothing else: `task_id`, `probe_id`, `variant`, `completion` (the reply),
`kind` (`score`, `timeout`, `error` or `crash`), `score`, and `accepted`, whether the audit counted the reply as accepted;
those two are `null` for a call with no score. The reference answer's calls have `probe_id` `reference` and the format as
`variant`, or `wrong_answer` for the wrong answer they are checked against. They are written even when the audit stops
after them, as when the checker rejects its own reference answers. Calls that only check other calls are not in the file:
the repeats that see whether a score stays the same, those that confirm a high defect, and the reference answer put into a
reply that scored above it. The file holds every reply in full, about 8 MB for 50 rows:
most of that is the long replies of P1, P17, Q28 and Q29, which go to five rows only, and each further row adds some 25 KB.
From Python it is `audit(..., calls="v1.jsonl")`.

**Reading the result.** Here `reward.py` is the quickstart checker, which reads the last run of digits, and `reward_v2.py`
reads the number in the last `\boxed{}`:

```
A is v1.jsonl, B is v2.jsonl.

Aligned 4,222 calls; agree on 85.2%. Only in A: 440, only in B: 0.

  reference    A accepts 240, B accepts 0
               [task 0, bare] 34  A 1.0  B 0.0
               ...
  O11          A accepts 80, B accepts 0
               [task 0, either] The answer is either 35 or 34.  A 1.0  B 0.0
               [task 0, inside_wrapper] \boxed{35 \text{ or } 34}  A 1.0  B 0.0
               [task 1, either] The answer is either 35 or 34.  A 1.0  B 0.0
  Q8           A accepts 0, B accepts 80
               [task 0, decimal] \boxed{34.0}  A 0.0  B 1.0
               ...

                              A      B
  Accepts non-answers         0.3%   0.3%
  Accepts wrong answers       14.0%  0.0%
  Accepts hedged answers      26.3%  11.8%
  Rejects right answers       18.8%  6.2%
  Pays wrong answers in part  0.0%   0.0%
  Calls with no score         0.0%   0.0%
```

- Calls are lined up by row, probe and variant. They agree where `accepted` is the same, and two calls with no score agree.
- A call only in A or only in B was sent in one audit only: the formats tried for the reference answer depend on the checker,
  a row whose reference answer a checker rejects gets no probes, some probes need two accepted formats, and a `--budget` can
  leave probes out. These are counted, not compared.
- Each audit writes its replies in a format its checker accepts. If the two accept different formats, the replies differ
  too: the first paragraph then says on how many calls, and the examples show both replies. To compare the two on the same
  replies, give both audits the same `--format`, one both checkers accept, such as `--format '#### {answer}'`.
- Then each probe where the two differ, the most differences first: how often A accepted a reply B did not, and the other
  way round, with up to three examples of each: the row, the variant, the reply and both scores. `no score on one side`
  counts calls where one checker gave no score and the other rejected the reply. `reference` lists the formats of the
  reference answer one accepts and the other does not.
- Which side is right is for you to read from the examples. Most probes send replies that should earn nothing, so there the
  side that accepts is the one to look at; Q, G and O4 send right answers, and there it is the side that rejects.
  `rewardaudit probes O4` says what a probe sends and what a sound checker does with it.

**The rates in the table** are worked out from the calls in each file alone: for each row of the summary, the share of the
calls behind it that were accepted (rejected, for Rejects right answers; for Pays wrong answers in part, wrong answers with a
score above zero that were not accepted), and last the share of all calls with no score. They are not the audit's own
numbers, which see more than the file holds. In the audit, Accepts non-answers is the best reply built without reading the
reference answer, chosen on part of the rows and measured on the others, where this table pools every call of N, E and L;
Rejects right answers leaves out replies with text around the answer when the checker takes only the answer; Pays wrong
answers in part looks only at the parts of a score that separate right answers from wrong ones, where this table takes the
whole score, so a bonus for the format counts here; and each rate has a 95% interval. Accepts wrong answers, Accepts hedged
answers and Rejects right answers otherwise come out the same as in the audit.

`compare` exits with 0, or with 2 when the two files cannot be compared, as when they have no `task_id` in common. From
Python, `compare("v1.jsonl", "v2.jsonl")` returns all of this as a dict: `aligned`, `agreement`, `different_replies`,
`only_in_a`, `only_in_b`, `by_probe` and `heads`.

## Use it in CI

| Exit code | Meaning |
|---|---|
| 0 | No high or medium finding |
| 1 | At least one high defect, or a rate above `--fail-on` |
| 2 | The audit could not be done, or the rates could not be measured |

```yaml
- run: pip install rewardaudit
- run: rewardaudit audit reward.py --dataset data.jsonl --n 200 --out report --fail-on "wrong_accepted>0.01,right_rejected>0.2"
```

With `--fail-on`, only the rates named decide the exit code. The names are `non_solving_accepted`, `wrong_accepted`,
`hedging_accepted`, `right_rejected` and `wrong_partial`.

## Use it from Python

```python
from rewardaudit import audit, AuditError, NotAuditable

def score(completion, answer):
    return float(answer in completion)

rows = [{"prompt": "1 + 1?", "answer": "2"}, ...]

try:
    result = audit(score, rows, n=200, seed=0)
except NotAuditable as e:   # the checker cannot be audited; e.cls, e.reason, e.advice
    print(e)
except AuditError as e:     # something to fix in the input
    print(e)

result.print()              # the summary; print(full=True) for every finding
result.defects              # a list of dicts, most severe first: id, name, explanation, severity, evidence, example, confirmation
result.findings             # a list of dicts: probe_id, name, category, severity, rate, ci, n, example (the reply is in example["completion"])
result.rates                # {"non_solving_accepted": 0.0, "wrong_accepted": 0.0, ...}
result.environment          # what the checker ran under: python, platform and packages, such as {"sympy": "1.14.0"}
result.save("out/")         # report.md and report.json
```

`checker` can be a function (closures and lambdas work), a file (`"reward.py"`, `"reward.py:fn"`), or an environment
(`"my-env"`, `"owner/name"`). `data` can be a path, `"hf:org/name"`, a list of dicts, a DataFrame or a Hugging Face Dataset.

If right and wrong answers score the same, `audit()` still returns. `result.rates` is then empty and `result.findings` holds only
replies that scored above the reference answer, so an empty `findings` means nothing unless `rates` has values.

A finding's `category` is its summary row: `non_solving_accepted`, `wrong_accepted`, `hedging_accepted`, `right_rejected` or
`robustness`.

Other arguments: `n` (a number, or `"all"`), `seed`, `prompt_col`, `answer_col`, `workers`, `timeout`, `probes` (such as
`["N", "O"]`), `reply_format` (such as `"<solution>{answer}</solution>"`), `answer_field` (`True` for a checker that takes
only the answer; see [above](#the-reply-format)), `budget_s` (seconds the audit may take; see
[above](#choose-how-many-rows)), `calls` (a file to write the calls to; see [above](#compare-two-checkers)), and
`verifiers`. For a function, also `style` (how it is
called, found from its parameters if not given: `bare`, `verl`, `trl`, `openrlhf`, `areal`, `rllm`, `easyr1` or `positional`),
`kwargs` (extra keyword arguments), `reference_col` (a column to pass as the reference), `reference_template` (such as
`"\\boxed{{{answer}}}"`), `ground_truth_json`, and `prompt_in_completion` (pass prompt and reply as one string).

The checker always runs in separate worker processes, so that a hang or crash in it cannot take your program down.

## What stays the same across versions

These change only with a note in the [changelog](../CHANGELOG.md):

- the command-line options and exit codes;
- the arguments of `audit()`;
- the top-level keys of `report.json` and the fields of a finding (`probe_id`, `name`, `category`, `severity`, `rate`, `ci`, `n`,
  `example`), with `schema_version` going up if they change;
- the five categories;
- the probe ids. New probes take new ids, and an id never moves to another probe.

Findings themselves can change, since new versions add probes and refine severities. Pin the version in CI if your builds
depend on the result.

## Troubleshooting

**Many timeouts.** `--timeout` allows more per call (10 seconds by default). A call that times out counts as failed, never as
accepted; your training framework may treat it differently.

**Crashes that mention the memory limit.** A worker that grows 4 GB beyond what it held when scoring started is stopped and the
call counts as a crash. Probes with huge numbers cause this in some checkers, which is itself a finding.

**`network attempts` in the summary.** Network access is blocked while scoring. If your checker needs a local service, rewardaudit
cannot audit it as it stands; put the logic in a function that does not need it.

**Slow audits.** Use a smaller `--n`, more `--workers`, or `--probes N,M,O,Q`. A large dataset also takes time and memory to
load, once per audit, and once per worker for environments.

**The checker prints a lot.** Output from the checker is discarded while scoring.
