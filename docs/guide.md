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
12. [What the audit does not see](#what-the-audit-does-not-see)
13. [Troubleshooting](#troubleshooting)

Commands using `reward.py` and `data.jsonl` assume the [quickstart files](../examples/quickstart) are in the current directory.
Commands starting with `examples/` run from the repository root.

## What you need

- **A checker:** a Python function that takes a model's reply and returns a score. It must run locally and must not
  call a model. Repeated inputs are checked for consistent scores; inconsistency is reported as S1 and the audit continues.
- **A dataset:** rows with a prompt and a reference answer, the final answer a right reply should give. A worked solution is
  not one; see [below](#the-reference-answer-is-not-what-the-checker-expects).

Try `rewardaudit demo` to audit three built-in checkers without any files of yours.

**Math equivalence.** `pip install 'rewardaudit[math]'` installs math-verify. When it is importable, RewardAudit uses it to
judge whether two answers are equivalent, which can change the probes sent and the findings. This affects the tool's own
judgement even if your checker does not use it. Keep the tool's extras and dependency versions the same when comparing runs;
the report's `environment` records the checker's imports, not all packages used by the tool.

## Prepare the dataset

| Source | How to pass it |
|---|---|
| JSON Lines | `--dataset data.jsonl` |
| JSON (a list of rows, or an object with a `data`, `rows`, `examples` or `tasks` list) | `--dataset data.json` |
| CSV | `--dataset data.csv` |
| Parquet (needs `rewardaudit[data]`) | `--dataset data.parquet` |
| Hugging Face (needs `rewardaudit[data]`) | `--dataset hf:openai/gsm8k:main:test`, that is `hf:name:config:split` |

Without a split, `hf:org/name` reads `train`; the report names that default. Use `hf:org/name::test` to choose `test`
without a config, or `hf:org/name:config:test` with one. From Python, `split="test"` supplies the default when the name has
no split; an explicit split in the name takes precedence.

**Columns.** The prompt is taken from the first column present among `prompt`, `question`, `problem`, `input`, `query`,
`instruction`, `text`. The reference answer is taken from `answer`, `solution`, `ground_truth`, `target`, `label`,
`final_answer`, `output`, `reference`, `reward_model.ground_truth`. If yours are named differently:

```bash
rewardaudit audit reward.py --dataset data.jsonl --prompt-col q --answer-col a
```

Dotted names reach into nested values (`reward_model.ground_truth`, `answers.text.0`). A prompt can be a string or a list of chat messages. An
answer that is a dict or list is passed on as it is. An `id` column names the rows in reports, if its values are unique.

## Connect your checker

RewardAudit calls your checker through one function: it takes a reply and the reference answer and returns the score.
Put that function in a file, name it `score`, and pass the file (`audit_me.py:my_function` names a function with another
name):

```bash
rewardaudit audit audit_me.py --dataset data.jsonl
```

If your checker is a class or a method, or needs more than the reply and the reference, a few lines of your own connect
it. Here the checker is a class in another module:

```python
# audit_me.py, next to the code it imports
from rewards import Grader

grader = Grader(strict=True)

def score(completion, answer):
    return grader.grade(response=completion, expected=answer)["points"]
```

RewardAudit passes each reply it writes as `completion` and the row's reference answer as `answer`, and takes the number
the function returns as the score.

A wrapper like this may import the checker, build the object it needs as your evaluation builds it, call it for one
reply and pick the score out of what it returns. It must not change what the checker decides: no cleaning or parsing of
the reply that your real evaluation does not do, and no arguments other than those your real evaluation passes.
Otherwise the audit describes the wrapper and not the checker.

**Functions that work as they are.** A checker that already takes the reply and the reference needs no wrapper: pass the
file it is in. The shapes below are shortcuts for the same interface. Given a file without a function name, RewardAudit
tries `score`, `compute_score`, `reward`, `reward_fn`, `reward_func`, `accuracy_reward` and `verify`, in that order.
If none exists, it takes the first public function defined in the file whose name contains `score` or `reward`, in definition
order, then the only public function if there is just one. `file.py:function` selects explicitly; `pkg.module:function`
loads an importable module.
Parameters are matched by name, so these work as they are:

```python
def score(completion, answer): ...                       # also response, prediction, gold, target, label, ...
def score(prompt, completion, answer): ...
def compute_score(solution_str, ground_truth): ...       # verl; data_source and extra_info are filled in too
def reward(prompts, completions, **columns): ...         # TRL and open-r1; returns a list, other columns are passed
```

An unrecognised two-argument signature gets the reply and reference by position; one with three or more gets the prompt,
reply and reference. Prefer recognised names. `prompt`, `messages`, `info`, `metadata`, `data_source` and `extra_info`
receive values from the row.

A function can return a number, a bool, a one-element list, tensor or array, a tuple (the first numeric element), an object
with a numeric `score`, `reward` or `overall` attribute (tried in that order), or a dict. From a dict the main value is taken from
`score`, `overall`, `reward`, `rewards`, `total` or `accuracy`, and other numbers count as sub-scores. A dict with none of
those keys counts all its numbers, bools included, as parts of a total, so return one number if that is not what they
are.

For a single-number score, the full score is the highest score earned by a reference reply across the rows and formats tried.
For a composite score, it uses the sum of the parts that distinguish reference answers from wrong answers when it can
identify them, and finds a full score for each row.
Rows whose reference does not reach the full score get no probes.

### A verl-style reward

Each training reward example has ten rows and needs no training library installed. Run from the repository root.

The [verl-style checker](../examples/verl) takes `solution_str`, `ground_truth`, `data_source` and `extra_info`, and returns
a dict of scores. It compares absolute values, so the sign is ignored:

```bash
rewardaudit audit examples/verl/reward.py --dataset examples/verl/data.jsonl --n all
```

Excerpt:

```
rewardaudit 0.3.0 · reward:compute_score · 10 of 10 rows (seed 0) · 1,342 calls · 1 s

  HIGH  D1  Signs and punctuation ignored  M2 100%
            -17
            Sent 5 more times: 1.0 each.
```

### A TRL-style batch reward

The [TRL-style checker](../examples/trl) takes lists of prompts, plain-text completions and answer columns. The audit calls
it with one reply per batch. It reads the first option letter even when the reply gives another answer:

```bash
rewardaudit audit examples/trl/reward.py --dataset examples/trl/data.jsonl --n all
```

Excerpt:

```
rewardaudit 0.3.0 · reward:reward · 10 of 10 rows (seed 0) · 1,275 calls · 1 s

  HIGH    D6  Hedged answers accepted  O7 100%, O16 60%, E2 30%
              The answer is either A or B.
              Sent 5 more times: 1.0 each. Shortest reply still accepted (1.0): A or
```

### The reply format

RewardAudit first finds out how the checker wants a reply written: it sends each reference answer as it is, in `\boxed{}`, in
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
answers, its findings low. Without it, RewardAudit infers this from Q5 and Q6, when the bare answer passes and text before
or after it mostly loses the score; where it infers right, the result is the same.

### Several accepted answers

Many QA benchmarks accept any of several answers and keep them in a list. RewardAudit writes each reply from one answer and
compares scores, so name one for the replies and hand the whole list to the checker:

```bash
rewardaudit audit reward.py --dataset data.jsonl --answer-col answers.text.0 --reference-col answers.text
```

A number in a column name picks an item of a list, and a row whose list is empty has no answer and is left out. Your
function's reference parameter gets the whole list. Replies are written from the answer you name, and G2 sends each of the
others as well, one per reply and up to 8 per row, since a checker given the list should accept any of them.

The [QA example](../examples/qa) deliberately checks only the first accepted name:

```bash
rewardaudit audit examples/qa/reward.py --dataset examples/qa/data.jsonl --n all --answer-col answers.text.0 --reference-col answers.text --answer-field
```

Excerpt:

```
rewardaudit 0.3.0 · reward:score · 10 of 10 rows (seed 0) · 1,092 calls · 1 s

  MEDIUM  D16  Reference answer rejected  G2 100%
               Victor-Marie Hugo
```

### The reference answer is not what the checker expects

Some checkers want the reference in a particular form. Wrap them as above and pass the reference in the form your
evaluation passes it, for example in `\boxed{}` or parsed from JSON.
If your dataset holds a worked solution, extract the final answer into a column first: RewardAudit rewrites the reference
answer to build its replies, so it has to be the answer itself.

G3 also checks whether changing the reference's writing changes the score of the same right reply (D22).
It runs only when the checker receives the answer as a plain string, including a `reference_template` with `{answer}`;
structured references, `--reference-col`, `ground_truth_json` and environments are excluded.
See the [reference](reference.md#g3-the-reference-answer-written-another-way) for the rewrites it tries.

## Choose how many rows

`--n` is yours to choose. RewardAudit always reads the whole dataset and draws that many rows at random, without replacement,
using `--seed` (0 by default). The default is 100; `--n all` uses every row. The report states how many rows were drawn from
how many. A row whose reference answer is empty is left out first, since a reply to it could not be scored against anything.

**What more rows give you.** Call counts depend on the answer types and probes that apply: the ten-row verl-style example
makes 1,342 calls, about 134 per row. A behaviour that affects 5% of rows shows up in a sample
of 100 with a probability of 99.4%, and in a sample of 30 with 79%. Larger samples narrow the intervals and reach rarer rows.

**Time.** The quickstart took 1 second on a laptop with 4 workers; a slower checker sets the pace. `--workers` runs that
many processes (4 by default, or your CPU count if lower). A progress bar shows the time left.
`--probes N,O,M` runs only those families.

**A slow checker.** `--budget 10m` (or `90s`, `1h`; `budget_s=600` from Python) limits sending probes, using elapsed time from the
audit's start. Reference calls, the determinism check, confirmation and environment installation are not limited by it.
Once that time has passed no further probe is sent: the probe whose calls are under way finishes, so the audit can
run over by about as long as one probe takes, and every call made counts as usual. A line at the top of the report then
names the probes that did not run, and `report.json` lists them under `meta.probes_not_run`. The summary rows are measured
on the probes that ran, so a probe left out can hide a finding; to keep every probe, draw fewer rows instead. Under a budget
the progress bar of the probes has no total, since counting their calls ahead means building every reply first.

## Read the result

Save the quickstart's report and print the findings behind each defect:

```bash
rewardaudit audit reward.py --dataset data.jsonl --full --out out
```

One finding from that run:

```
rewardaudit 0.3.0 · reward:score · 40 of 40 rows (seed 0) · 4,776 calls · 1 s

  HIGH    O11  hedge between two answers, the right one last                                                        100%
               The answer is either 68 or 34.
               95% interval [97%, 100%], 120 calls
```

Each row answers a question about the checker. On the first five rows, `yes` means a high or medium problem. **Runs reliably**
reads the other way round: `yes` is good.

The percentage is the share of calls that went that way, with a 95% interval. These are shares of the replies RewardAudit sends,
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

The [reference](reference.md) lists all 22 defects, their explanations and their probes, plus G1, S1, S2 and the evidence ids.
`rewardaudit probes D6` looks up a defect; `rewardaudit probes O11` explains a probe and shows examples.

**Findings.** A probe can yield more than one finding: acceptance or rejection, and robustness. Each records a rate,
with an example where available; `--full` prints them, and `report.md` always has them.
Most acceptance or rejection finding rates pool the probe's variants, counting only calls that returned a score.
For example, O11 reports 33% when one variant is accepted on 99 of 100 calls and the other two on 0 of 100 each:
99 accepted calls out of 300.

The [reference](reference.md) marks probes that instead report the highest problem rate of one variant; their denominator
is that variant's scored calls, and the example comes from it. Scale probes M24, M28, Q28 and Q29 report one selected size:
the largest error accepted on at least half of scored calls, or the shortest text rejected on at least half. If no size
meets that threshold, they use the highest problem rate. Their notes give the acceptance rate at every size.
Errors and timeouts are separate findings over all of that probe's calls. On prompts requiring the answer at the end,
Q28 still measures failures and scoring time, but its calls stay out of right-answer rejection findings.
Severity is a rule of thumb for what to look at first:

| Severity | Meaning |
|---|---|
| High | A reply's text reaching a Python import, on one call. Non-answers, wrong answers or hedged answers accepted on about half of calls or more, a number 5% off accepted on half of rows, or right answers losing the score just for the reasoning before them on one call in five or more. |
| Medium | Wrong answers accepted or right answers rejected on one call in ten or more, hedged answers accepted on one in five, non-answers accepted more often than chance would give, crashes and timeouts, scores that are not numbers, wrong answers paid in part as the row above says, a number 1% off accepted on half of rows, a right answer lost to 4k characters of text or fewer. |
| Low | The same less often, or about what chance would give, and errors the checker raised. Listed for completeness. |

Q23 rejection findings always stay low because few checkers read every number spelling; O13 acceptance findings stay low
when the accepted reply format has no thinking block, so reading an answer inside `<think>` does not violate a required
boundary. For an extracted-answer checker, prose rejections (Q5, Q6, Q21, Q22, Q28, Q29, O4) stay low because surrounding
text is outside its input. M24 stays low for numeric errors below 1%, and Q28/Q29 for text longer than 4k characters,
which may reflect rounding or unusually long replies.

**A checker that compares the whole reply.** If the bare answer passes and text around it mostly loses the score, or you
say so with `--answer-field`, the checker takes a reply to be its answer field, as a script scoring a file of predictions
does. The report says so at the top, the replies with text around the answer stay out of Rejects right answers, and their
findings stay low. Put the extraction step of your harness in front of the checker if you want those replies measured.

A few calls say little, so a finding stays low, with the note "too few calls to rate", below ten scored calls or on one row only.
The result says it could not rate some findings and suggests more rows; this alone still exits with 0.
Failed calls and rejected reference answers need two rows. A defect looks at its probes together: on a dataset with few numbers,
each probe on numbers has only a few calls, and between them they may have enough.

**A finding is about behaviour.** RewardAudit does not know what you intended. If your task demands an exact format, then
`34.0` being rejected may be right. Read the example, and run `repro.py` if you want to re-score it.

**Composite scores.** If a checker returns sub-scores (correctness plus format, say), RewardAudit works out which of them
separate right answers from wrong ones and judges acceptance on those. The report names them. A reply that earns only the
format credit does not count as accepted.

**Saving the report.** `--out DIR` writes:

- `report.md`: the summary, defects, findings with examples, what to fix, accepted formats, reliability, dataset checks and how it ran.
- `report.json`: the same results for scripts: `defects`, `metrics.findings`, `metrics.head` for the summary rows, and
  `metrics.probes` for the rate of every probe.
- `repro.py`: re-scores the reply behind each finding (not written for a function passed from Python).

**Probes.** `rewardaudit probes` lists all 79; `rewardaudit probes Q6` shows what one sends and which row it counts toward. They come in families: N (empty and
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
8 by name. Packages imported by the tool to read or check the dataset, such as pyarrow, datasets or math-verify, do not
count, nor does cloudpickle, which carries a function over from Python. Another version of one can change what the checker decides, so these are the versions
that reproduce the result; `environment` in `report.json` lists them all, with the Python version and the platform.

## When the audit stops

RewardAudit does not fit every checker. What it cannot measure it does not audit: it stops, says why in a line, and exits with 2.

| Message contains | What to do |
|---|---|
| Could not load the checker | Fix the named import or exception in the checker environment; check the file path |
| No reference answer could be scored / No reference answer earned full score / Only … reference answers earned full score | Check `--answer-col` holds final answers; give the [reply format](#the-reply-format) or [connect the reference correctly](#the-reference-answer-is-not-what-the-checker-expects) |
| The checker raised an error on … calls with the reference answer | Fix the named exception; check the reference type and reply format your checker expects |
| The checker returned a value that is not a score | Return one of the [supported types](#connect-your-checker), rather than a string or `None` |
| Reference answers are empty / placeholders / identifiers / specifications, or earn only a small share of the reward | Select the answer column; tasks without reference answers are out of scope |
| … calls with the reference answer tried to use the network | Use a checker that scores locally; remote services and model judges are out of scope |
| … calls with the reference answer timed out | Raise `--timeout` |
| … calls with the reference answer crashed | Inspect the checker failure; fewer `--workers` may help with memory |
| … rollouts with the reference answer failed on this verifiers | Read the rollout error; use the suggested `--verifiers` constraint if one is given |
| The rubric has no reward function with a nonzero weight | Use a rubric with a scoring reward; metrics alone cannot be audited |
| Multi-turn / tools / container / judges / own environment | The environment is outside the scope of a deterministic checker for one reply |

For an environment, the dataset and rubric come with it: column flags and wrapper functions do not apply. When its reference
answers fail, check `--format` and any suggested verifiers version instead.

If a checker gives right and wrong answers the same score, the audit still finishes, says the rates could not be measured, and
exits with 2. Replies that score above the reference answer are still reported.

## Audit a verifiers environment

```bash
rewardaudit audit my-env --n 200           # installed in this Python
rewardaudit audit owner/name --n 200       # from the Prime Intellect Hub
rewardaudit audit owner/name@1.2.0         # a specific version
```

An environment brings its own dataset and rubric, so there is no `--dataset`. For `owner/name`, RewardAudit uses
[uv](https://docs.astral.sh/uv/) to install the package into its own venv under `~/.cache/rewardaudit/envs` (set
`REWARDAUDIT_HOME` to move it). It picks the verifiers version for the package's API.

A checker runs in worker processes with your user's permissions and environment variables. During scoring, the worker
replaces Python's `socket` connection functions to block network connections and record attempts;
Unix-domain sockets for local communication remain allowed. A forked child inherits this guard, but a program started with
exec or native code opening sockets itself can go around it. It is not a security sandbox, and loading and installing code
happen outside the scoring guard. Run only code you trust, or use a container or throwaway machine for isolation.

If an environment was written for an older verifiers, its reference answers may fail for no reason of its own. The message
then suggests a version:

```bash
rewardaudit audit owner/name --verifiers '<0.1.11'
```

The report says which version the results were measured on. Delete the environment's directory under
`~/.cache/rewardaudit/envs` to install it afresh.

## Compare two checkers

Audit both checkers on the same dataset with the same `--n` and `--seed`, then compare their calls files. For a runnable
example, put this beside the quickstart's `reward.py` and `data.jsonl`. It reads the number in the last `\boxed{}`:

```python
# reward_v2.py
import re


def score(completion, answer):
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", completion)
    try:
        return float(bool(boxes) and float(boxes[-1]) == float(answer))
    except ValueError:
        return 0.0
```

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
reply that scored above it. The file holds every reply in full, 7.8 MB for the 40-row quickstart:
most of that is the long replies of P1, P17, Q28 and Q29, which go to five rows only.
From Python it is `audit(..., calls="v1.jsonl")`.

**Reading the result.** Excerpt from those runs:

```
A is v1.jsonl, B is v2.jsonl.

Aligned 4,262 calls; agree on 84.4%. Only in A: 440, only in B: 0.

  reference    A accepts 240, B accepts 0
               [task 0, bare] 34  A 1.0  B 0.0
               [task 1, bare] 34  A 1.0  B 0.0
               [task 2, bare] 45  A 1.0  B 0.0
  O11          A accepts 80, B accepts 0
               [task 0, either] The answer is either 68 or 34.  A 1.0  B 0.0
               [task 0, inside_wrapper] \boxed{68 \text{ or } 34}  A 1.0  B 0.0
               [task 1, either] The answer is either 68 or 34.  A 1.0  B 0.0
  Q8           A accepts 0, B accepts 80
               [task 0, decimal] \boxed{34.0}  A 0.0  B 1.0
               [task 0, two_decimals] \boxed{34.00}  A 0.0  B 1.0
               [task 1, decimal] \boxed{34.0}  A 0.0  B 1.0
```

- Calls are lined up by row, probe and variant. They agree where `accepted` is the same, and two calls with no score agree.
- A call only in A or only in B was sent in one audit only: the formats tried for the reference answer depend on the checker,
  a row whose reference answer a checker rejects gets no probes, some probes need two accepted formats, and a `--budget` can
  leave probes out. These are counted, not compared.
- Each audit writes its replies in a format its checker accepts. If the two accept different formats, the replies differ
  too: the first paragraph then says on how many calls, and the examples show both replies. To use the same main reply
  format, give both audits a `--format` they accept, such as `--format '\boxed{{{answer}}}'`. Probes that depend on other
  accepted formats may still write different replies.
- Then each probe where the two differ, the most differences first: how often A accepted a reply B did not, and the other
  way round, with up to three examples of each: the row, the variant, the reply and both scores. `no score on one side`
  counts calls where one checker gave no score and the other rejected the reply. `reference` lists the formats of the
  reference answer one accepts and the other does not.
- Which side is right is for you to read from the examples. Most probes send replies that should earn nothing, so there the
  side that accepts is the one to look at; Q, G and O4 send right answers, and there it is the side that rejects.
  `rewardaudit probes O4` says what a probe sends and what a sound checker does with it.

**The rates in the table** use the calls in each file alone. They pool all non-answer replies, include prose around right
answers even for a checker of answer fields, and count partial credit from the whole score. The audit instead selects a
non-answer reply on separate rows, excludes prose where appropriate, and judges the parts that distinguish right answers.
Agreement compares acceptance, so two different partial scores or two calls with no score can agree. Read the examples and
scores as well as the percentages.

`compare` exits with 0, or with 2 when the two files cannot be compared, as when they have no `task_id` in common. From
Python, `compare("v1.jsonl", "v2.jsonl")` returns all of this as a dict: `aligned`, `agreement`, `different_replies`,
`only_in_a`, `only_in_b`, `by_probe` and `heads`.

## Use it in CI

| Exit code | Meaning |
|---|---|
| 0 | No high defect, or no named rate above its `--fail-on` threshold; too few calls to rate findings can still exit 0 |
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
from reward import score

try:
    result = audit(score, "data.jsonl", n=200, seed=0)
except NotAuditable as e:   # e.cls, e.reason, e.advice describe why it cannot be measured
    raise SystemExit(str(e))
except AuditError as e:     # input or setup to fix
    raise SystemExit(str(e))

result.print()              # the summary; print(full=True) for every finding
result.defects              # dicts: id, name, explanation, severity, evidence, example, confirmation
result.findings             # dicts: probe_id, name, category, severity, rate, ci, n, example
result.rates                # the measured summary rates, keyed by the names --fail-on takes
result.environment          # python, platform and the checker's imported package versions
result.save("out/")         # report.md and report.json; a checker passed by function has no repro.py
```

`checker` can be a function (closures and lambdas work), a file (`"reward.py"`, `"reward.py:fn"`), `"pkg.module:fn"`, or an environment
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
`kwargs` (extra keyword arguments), `split` (the Hugging Face split when absent from the name),
`reference_col` (a column to pass as the reference), `reference_template` (such as
`"\\boxed{{{answer}}}"`), `ground_truth_json`, and `prompt_in_completion` (pass prompt and reply as one string).

`log` takes a callback `log(message: str)` for loading, sampling and other status messages. `progress` takes
`progress(stage: str, done: int, total: int)`: counts within each stage; the total is an estimate, or 0 when unknown.
Both default to `None`, so `audit()` prints no progress or status unless you supply callbacks. Call `result.print()` for
the report. The checker runs in separate worker processes.

`result.meta` holds run details such as `probes_not_run`; `dataset_issues` holds checks of the data; `tasks` holds sampled
rows; `contract` describes accepted formats and scores; `determinism` and `latency_ms` describe repeated calls and timing.
See [examples/python_api.py](../examples/python_api.py) for a complete in-memory example.

## What stays the same across versions

These change only with a note in the [changelog](../CHANGELOG.md):

- the command-line options and exit codes;
- the Python API: `audit`, `compare`, `AuditError` and `NotAuditable`, imported from `rewardaudit`, with their documented arguments;
- the top-level keys of `report.json` and the documented finding fields, with `schema_version` going up if they change;
- the five category names;
- probe, defect and finding ids, including G1, S1, S2 and evidence ids such as `format_only`. New ids are appended;
  retired or skipped ids are never reused, and an id never moves to a different meaning.

A 0.x minor release may change the command line, Python API or report schema with a changelog entry. Findings and severities
can change as probes and judgement improve. Pin the tool and dependencies when CI depends on results. Use a saved `repro.py`
with the RewardAudit version recorded in its report; other package modules are not part of the Python API.

## What the audit does not see

Read findings against the rules of the evaluation you actually run:

- The audit sends replies directly to the checker. If your pipeline cuts a reply at the first `</answer>`, check whether
  a flagged reply with a second answer would ever reach it before changing the checker.
- A required format shown only by examples or non-English instructions may be missed. If replies must end with
  `Antwort: 42`, give `--format 'Antwort: {answer}'` and read each flagged reply against that instruction.
- Probe prose is English; a letter parser may read its filler as option letters even on tasks in another language.
- G2 expands only native lists of strings, not JSON text or structured annotation objects.
- P16 tests specific import forms; execution reached only through other reply shapes or a fresh interpreter can be missed.
- Q27 adds an English article only when the prompt contains the word `the`; this does not establish the answer's language.
  Check that a rewritten answer is valid in the task's language before treating its refusal as a defect.
- The tool cannot decide whether a scoring rule is intentional. If the benchmark allows a 5% numeric tolerance or uses
  F1 partial credit, compare a flagged reply with its documented scoring rules before changing the checker.
- A clean result covers only the rows and probes tested. Empty references, such as a correct abstention, are omitted;
  check the reported rows left out, reference acceptance and any budget notice before relying on the result.
- This audit needs a local checker for one reply and a reference answer. For a checker that executes SQL,
  scores a conversation or asks a model, test its own evaluation pipeline instead.

## Troubleshooting

**Many timeouts.** `--timeout` allows more per call (10 seconds by default). A call that times out counts as failed, never as
accepted; your training framework may treat it differently.

**Crashes that mention the memory limit.** A worker that grows 4 GB beyond what it held when scoring started is stopped and the
call counts as a crash. Probes with huge numbers cause this in some checkers, which is itself a finding.

**`network attempts` in the summary.** Network access is blocked while scoring. If your checker needs a local service, RewardAudit
cannot audit it as it stands; put the logic in a function that does not need it.

**Slow audits.** Use a smaller `--n`, more `--workers`, or `--probes N,M,O,Q`. A large dataset also takes time and memory to
load, once per audit, and once per worker for environments.

**The checker prints a lot.** Output from the checker is discarded while scoring.
