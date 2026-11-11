# rewardaudit

Test the checker before you train on it.

A checker is the function that turns a model's reply into a reward. If it gives full score to a reply that does not solve
the task, the model learns to write that reply. If it gives no score to a right answer written differently, the model is
punished for being right.

rewardaudit finds these problems without a model. You give it a checker and a dataset with reference answers. It writes
replies whose quality is known in advance, such as an empty reply, a near-miss wrong answer, two answers at once, or the right
answer in another form. It sends them to the checker and reports what the checker did with them.

## Install

```bash
pip install rewardaudit
```

Python 3.11 or newer, on Linux or macOS. To read parquet files or Hugging Face datasets: `pip install 'rewardaudit[data]'`.
The [math extra](https://github.com/xzx34/rewardaudit/blob/main/docs/guide.md#what-you-need) changes how the tool judges equivalent answers.

## Quick start

You need a scoring function and a dataset with reference answers. This example, with all its 40 rows, is in
[`examples/quickstart`](https://github.com/xzx34/rewardaudit/tree/main/examples/quickstart). Run the command from that directory.

```python
# reward.py
import re

def score(completion, answer):
    numbers = re.findall(r"-?\d+", completion)
    return 1.0 if numbers and numbers[-1] == answer.strip() else 0.0
```

Two of the rows:

```json lines
{"question": "Gus has 23 apples and gets 11 more. How many apples does Gus have now?", "answer": "34"}
{"question": "Ben has 41 cards and gives away 7. How many cards does Ben have left?", "answer": "34"}
```

```bash
rewardaudit audit reward.py --dataset data.jsonl --n 100
```

Excerpt (elapsed time varies):

```
rewardaudit 0.2.0 · reward:score · 40 of 40 rows (seed 0) · 4,779 calls · 1 s

Result: problems found (2 high, 4 medium)

  Accepts non-answers         no     0%  [0%, 14%]
  Accepts wrong answers       yes   14%  [11%, 17%]
  Accepts hedged answers      yes   26%  [23%, 30%]
  Rejects right answers       yes   19%  [16%, 22%]
  Pays wrong answers in part  no     0%  [0%, 1%]
  Runs reliably               yes  no errors, timeouts or crashes in 4,662 calls

Rates are shares of calls, with 95% intervals. Yes on the first five rows: a high or medium problem.

Reference answers accepted on 100% of rows. Replies were sent in the format \boxed{34}, one of 22 it accepts.

Defects (7)
  HIGH    D1   Signs and punctuation ignored  M22 100%, M3 4%
               \boxed{−34}
               Sent 5 more times: 1.0 each.
  HIGH    D6   Hedged answers accepted  O11 100%
               The answer is either 68 or 34.
               Sent 5 more times: 1.0 each. Shortest reply still accepted (1.0): 68 34.
```

Here the checker takes the last run of digits in the reply, so `either 68 or 34` and `−34` earn full score and `34.0` earns nothing.
Under each high defect, its reply is sent five more times. When possible, it is cut down to the shortest reply still
accepted: `68 34.` shows that only the last number counts.
`rewardaudit demo` shows three more examples without any files of yours.

## What it checks

| Summary row | Question | Example of a reply it sends |
|---|---|---|
| Accepts non-answers | Can a reply that ignores the task earn full score? | an empty reply, `\boxed{}`, the prompt copied back |
| Accepts wrong answers | Does a definite error still earn full score? | off by one, sign flipped, another task's answer |
| Accepts hedged answers | Does giving two answers, one of them right, earn full score? | `either 68 or 34`, the right answer retracted |
| Rejects right answers | Does a correct answer lose the score when written differently? | `34.0`, `\dfrac{1}{2}`, an explanation before it |
| Pays wrong answers in part | Does a definite error earn more than nothing? | off by one, one word changed |
| Runs reliably | Does the checker run without errors, hangs, crashes or network use? | huge numbers, malformed input, repeated calls |

`rewardaudit probes` lists all 84 probes; `rewardaudit probes O11` explains one.
The [reference](https://github.com/xzx34/rewardaudit/blob/main/docs/reference.md) describes every probe and all 22 defects.
`rewardaudit compare v1.jsonl v2.jsonl` shows where two checkers, or two versions of one, decide differently, probe by
probe, from two audits of the same rows run with `--calls`.

A finding says how the checker behaved, not whether that was intended. A strict format rule may be what you want, so read
the example before you change anything.

## How many rows

rewardaudit reads your whole dataset and draws `--n` rows from it at random, without replacement. The rows you get depend
only on the dataset and `--seed`.

```bash
rewardaudit audit reward.py --dataset data.jsonl --n 500            # up to 500 random rows (default: 100)
rewardaudit audit reward.py --dataset data.jsonl --n all            # every row
rewardaudit audit reward.py --dataset data.jsonl --n 500 --seed 1   # another draw
```

The 40-row quickstart makes 4,779 checker calls, about 119 per row; the count depends on which probes apply.
Checks on the dataset itself, such as duplicate prompts and empty answers, always cover every row.

## From Python

```python
from rewardaudit import audit

result = audit(score, "data.jsonl", n=200)  # or a list of dicts, a DataFrame, a Hugging Face Dataset
result.print()
result.defects       # what the checker does wrong, most severe first
result.findings      # every finding, one per probe
result.rates         # the rate behind each row of the summary
result.save("out/")  # report.md, report.json
```

## Scoring functions and environments

rewardaudit needs a function that takes a reply and a reference answer and returns a score. Parameters are matched by name,
so `score(completion, answer)`, verl's `compute_score(solution_str, ground_truth)` and TRL's
`reward(prompts, completions, **columns)` work as they are. For any other benchmark, a few lines that call its checker are
enough, `--format '<solution>{answer}</solution>'` names a reply format the tool does not try by itself, and
`--answer-col answers.text.0 --reference-col answers.text` handles a list of accepted answers. Datasets can be
jsonl, json, csv, parquet or `hf:org/name[:config[:split]]`.

A [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment works too, and brings its own dataset:

```bash
rewardaudit audit my-env --n 200               # installed in this Python
rewardaudit audit primeintellect/gsm8k         # from the Prime Intellect Hub, into its own venv (needs uv)
```

An environment from the Hub is someone else's code, and this installs and runs it on your machine, as installing it yourself
would. Only do this with packages you would run anyway.

## In CI

The exit code is 0 with no high defect, 1 with one, and 2 if the audit could not be done or the rates could not
be measured.
`--fail-on "right_rejected>0.1,wrong_accepted>0.01"` fails only above the rates you choose.

## What it does not do

rewardaudit does not fit every checker. It audits those that score one reply and give the same score every time, on a dataset
with reference answers. Multi-turn and tool environments and checkers that call a model are out of scope, and it stops and
says so instead of giving misleading numbers. It tests a fixed set of replies, so a clean result means none of these replies
got through, not that the checker is sound.

The [usage guide](https://github.com/xzx34/rewardaudit/blob/main/docs/guide.md) covers everything in more detail.

## License

Apache-2.0
