# Usage guide

1. [What you need](#what-you-need)
2. [Prepare the dataset](#prepare-the-dataset)
3. [Connect your checker](#connect-your-checker)
4. [Choose how many rows](#choose-how-many-rows)
5. [Read the result](#read-the-result)
6. [When the audit stops](#when-the-audit-stops)
7. [Audit a verifiers environment](#audit-a-verifiers-environment)
8. [Use it in CI](#use-it-in-ci)
9. [Use it from Python](#use-it-from-python)
10. [What stays the same across versions](#what-stays-the-same-across-versions)
11. [Troubleshooting](#troubleshooting)

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

Parameters named `prompt`, `messages`, `info`, `metadata`, `data_source` or `extra_info` get those values from the row. A
function can return a number, a bool, a dict (the main value from `score`, `overall`, `reward`, `rewards`, `total` or
`accuracy`; other numbers count as sub-scores), or a one-element list or tensor.

**Anything else: wrap it.** rewardaudit needs only a function that takes a reply and a reference answer and returns a score,
so a few lines in your own file connect any benchmark. Call your checker from it without changing what the checker decides:

```python
# audit_me.py
from my_project.rewards import Grader

grader = Grader(strict=True)

def score(completion, answer):
    return grader.grade(response=completion, expected="\\boxed{%s}" % answer)
```

### The reply format

rewardaudit first finds out how the checker wants a reply written: it sends each reference answer as it is, in `\boxed{}`, in
`<answer>` tags, after `Final Answer:`, after `####`, and in other common forms, along with any tag or answer line the prompts
ask for. Every probe then writes its replies in a form the checker accepts. If yours is none of these, name it:

```bash
rewardaudit audit reward.py --dataset data.jsonl --format '<solution>{answer}</solution>'
```

### Several accepted answers

Many QA benchmarks accept any of several answers and keep them in a list. rewardaudit writes each reply from one answer and
compares scores, so name one for the replies and hand the whole list to the checker:

```bash
rewardaudit audit reward.py --dataset data.jsonl --answer-col answers.text.0 --reference-col answers.text
```

A number in a column name picks an item of a list. A row whose list is empty counts as having no answer.

### The reference answer is not what the checker expects

Some checkers want the reference in a particular form. Wrap them as above, for example to add `\boxed{}`, or to parse JSON.
If your dataset holds a worked solution, extract the final answer into a column first: rewardaudit rewrites the reference
answer to build its replies, so it has to be the answer itself.

## Choose how many rows

`--n` is yours to choose. rewardaudit always reads the whole dataset and draws that many rows at random, without replacement,
using `--seed` (0 by default). The default is 100; `--n all` uses every row. The report states how many rows were drawn from
how many.

**What more rows give you.** Each row costs about 90 checker calls. A behaviour that affects 5% of rows shows up in a sample
of 100 with a probability of 99.4%, and in a sample of 30 with 79%. Larger samples narrow the intervals and reach rarer rows.

**Time.** A fast checker takes a few minutes per thousand rows at most; a checker that takes 10 ms per call needs about 15
minutes per thousand rows on one worker. `--workers` runs that many processes (4 by default, or your CPU count if lower). A
progress bar shows the time left. `--probes N,O,M` runs only those families.

## Read the result

```
Result: problems found (1 high, 2 medium)

  Accepts non-answers     no     0%  [0%, 14%]
  Accepts wrong answers   no     1%  [0%, 3%]
  Accepts hedged answers  yes   23%  [20%, 27%]
  Rejects right answers   yes   19%  [16%, 22%]
  Runs reliably           yes  no errors, timeouts or crashes in 3,727 calls
```

Each row answers a question about the checker. `yes` or `no` says whether that row has a high or medium finding. The percentage
is the share of calls that went that way, with a 95% interval.

- **Accepts non-answers:** the best reply built without reading the reference answer. It is chosen on part of the rows and
  measured on the others, so the choice does not inflate the rate. A reply that always says the most common answer is right
  on some rows by chance, so this counts only above that level.
- **Accepts wrong answers:** the reference answer with a definite error.
- **Accepts hedged answers:** a reply stating two answers, of which one is right, or stating the right one and taking it back.
- **Rejects right answers:** the reference answer written another way, still correct.
- **Runs reliably:** errors, timeouts, crashes and network attempts, and whether repeating a call gives the same score.

**Findings.** Each finding is one probe with its rate and an example reply. Severity is a rule of thumb for what to look at
first:

| Severity | Meaning |
|---|---|
| High | Happens on most calls, or right answers lose the score just for the reasoning around them. |
| Medium | Happens often enough to shape what a model learns: roughly one call in ten or more. |
| Low | Rare, or about what chance would give. Listed for completeness. |

**A finding is about behaviour.** rewardaudit does not know what you intended. If your task demands an exact format, then
`34.0` being rejected may be right. Read the example, and run `repro.py` if you want to re-score it.

**Composite scores.** If a checker returns sub-scores (correctness plus format, say), rewardaudit works out which of them
separate right answers from wrong ones and judges acceptance on those. The report names them. A reply that earns only the
format credit does not count as accepted.

**Saving the report.** `--out DIR` writes:

- `report.md`: the summary, every finding with an example, what to fix, accepted formats, the dataset checks, how it was run.
- `report.json`: the same results for scripts, with the rate of every probe.
- `repro.py`: re-scores the reply behind each finding (not written for a function passed from Python).

**Probes.** `rewardaudit probes` lists them; `rewardaudit probes Q6` shows what one sends. They come in families: N (empty and
placeholder replies), E (many candidates in one reply), L (copying the prompt), O (two answers), M (definite errors), Q (right
answers rewritten), P (malformed and extreme input).

**Dataset checks.** Empty answers, duplicate prompts, answers that appear in the prompt, skewed answers, and multiple-choice
answers that are not among the options. These read the data only, and they do not affect the exit code.

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

## Use it in CI

| Exit code | Meaning |
|---|---|
| 0 | No high or medium finding |
| 1 | At least one high or medium finding, or a rate above `--fail-on` |
| 2 | The audit could not be done, or the rates could not be measured |

```yaml
- run: pip install rewardaudit
- run: rewardaudit audit reward.py --dataset data.jsonl --n 200 --out report --fail-on "wrong_accepted>0.01,right_rejected>0.2"
```

With `--fail-on`, only the rates named decide the exit code. The names are `non_solving_accepted`, `wrong_accepted`,
`hedging_accepted` and `right_rejected`.

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
result.findings             # a list of dicts: probe_id, name, category, severity, rate, ci, n, example
result.rates                # {"non_solving_accepted": 0.0, "wrong_accepted": 0.0, ...}
result.save("out/")         # report.md and report.json
```

`checker` can be a function (closures and lambdas work), a file (`"reward.py"`, `"reward.py:fn"`), or an environment
(`"my-env"`, `"owner/name"`). `data` can be a path, `"hf:org/name"`, a list of dicts, a DataFrame or a Hugging Face Dataset.

A finding's `category` is its summary row: `non_solving_accepted`, `wrong_accepted`, `hedging_accepted`, `right_rejected` or
`robustness`.

Other arguments: `n` (a number, or `"all"`), `seed`, `prompt_col`, `answer_col`, `workers`, `timeout`, `probes` (such as
`["N", "O"]`), `reply_format` (such as `"<solution>{answer}</solution>"`), and `verifiers`. For a function, also `style`,
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
