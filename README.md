<p align="center">
  <img src="https://github.com/xzx34/RewardAudit/raw/main/docs/assets/banner.svg" alt="RewardAudit: test the checker before you train on it" width="100%">
</p>

<p align="center">
  <b>English</b> · <a href="https://github.com/xzx34/RewardAudit/blob/main/README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/rewardaudit/"><img src="https://img.shields.io/pypi/v/rewardaudit?style=flat-square&color=14b8a6&label=pypi" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-334155?style=flat-square" alt="Python 3.11 to 3.13">
  <a href="https://github.com/xzx34/RewardAudit/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/xzx34/RewardAudit/ci.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://github.com/xzx34/RewardAudit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-334155?style=flat-square" alt="Apache-2.0"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-tests">What it tests</a> ·
  <a href="#results-on-real-checkers">Results on real checkers</a> ·
  <a href="https://github.com/xzx34/RewardAudit/blob/main/docs/guide.md">Guide</a> ·
  <a href="https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md">Reference</a>
</p>

Reinforcement learning with verifiable rewards trusts one small piece of code: the **checker** that compares the
model's answer with the reference answer and returns a score. When the checker is wrong, training learns its mistakes.
A checker that pays for `either 68 or 34` teaches the model to hedge. One that scores `34.0` as wrong teaches it to
avoid a right answer.

RewardAudit tests a checker the way you would test any function: it sends replies whose correct score is known in
advance and reports the ones the checker scores wrongly. You give it the checker and a dataset with reference answers.
It needs no model and no GPU, and a small audit takes seconds.

## Quick start

Install it:

```bash
pip install rewardaudit
```

Watch an audit before writing anything. The demo audits three small checkers that ship with the package:

```bash
rewardaudit demo
```

Then audit your own. A checker is a Python function that takes the model's reply and the reference answer and returns
a score. This one takes the last number in the reply:

```python
# reward.py
import re

def score(completion, answer):
    numbers = re.findall(r"-?\d+", completion)
    return 1.0 if numbers and numbers[-1] == answer.strip() else 0.0
```

The dataset holds one task per line, with its reference answer:

```json
{"question": "Gus has 23 apples and gets 11 more. How many apples does Gus have now?", "answer": "34"}
{"question": "Ben has 41 cards and gives away 7. How many cards does Ben have left?", "answer": "34"}
```

Run the audit:

```bash
rewardaudit audit reward.py --dataset data.jsonl
```

Both files, with all 40 tasks, are in [`examples/quickstart`](https://github.com/xzx34/RewardAudit/tree/main/examples/quickstart).

## Reading the result

<p align="center">
  <img src="https://github.com/xzx34/RewardAudit/raw/main/docs/assets/demo.svg" alt="The audit's output: problems found, six summary rows, and two high defects with the replies that show them" width="100%">
</p>

- **Result** says whether the checker has problems, and how many are high or medium.
- **The six summary rows** each ask one question about the checker. *Yes* marks a problem; the percentage is how often
  it happened, with a 95% interval.
- **Defects** say what is wrong, most severe first, each with a reply that shows it.

This checker has two high defects:

- `\boxed{−34}` gets full score. The pattern does not know the Unicode minus sign, so it reads 34.
- `The answer is either 68 or 34.` gets full score, because only the last number counts. Cut down to the shortest
  reply that still gets full score, it is just `68 34.`

Every high defect is confirmed by sending its reply five more times. `--full` prints every finding, and `--out DIR`
saves the report with a script that reproduces it.

## What it tests

RewardAudit writes its replies from each task's reference answer. Some leave the task unsolved, so a correct checker
gives them zero. Others are right but written differently, so they deserve full score. There are 79 probes in eight
families:

| Family | The reply holds | Example, reference answer `34` | Should get |
|---|---|---|---|
| **N** | no answer | an empty reply, `\boxed{}` | 0 |
| **E** | many answers at once | `Possible answers: -20, -19, …, 200` | 0 |
| **L** | text copied from the prompt | the question itself | 0 |
| **O** | a right and a wrong answer | `The answer is either 68 or 34.` | 0 |
| **M** | the answer with a definite error | `\boxed{−34}`, `\boxed{340}` | 0 |
| **Q** | the right answer, written differently | `\boxed{34.0}`, `34` after a long explanation | full score |
| **G** | another accepted answer for the row | an alias listed in the dataset | full score |
| **P** | malformed or extreme input | `9**9**9`, an unclosed `\boxed{` | no error, no hang |

An audit runs in four steps:

1. **Learn the format.** It sends every reference answer in each format it knows, from `\boxed{}` to answer tags,
   and keeps the formats the checker accepts.
2. **Probe.** It sends the probes in those formats. The checker runs in separate processes with time and memory
   limits, so a crash or a hang becomes a finding, not a stopped audit.
3. **Compare.** It compares every score with the score the reply should get, and groups what it finds into 22
   defects such as *Hedged answers accepted* or *Equivalent rewrites rejected*.
4. **Confirm.** It sends each high defect's reply five more times and cuts it down to the shortest reply the checker
   still accepts.

`rewardaudit probes` lists every probe, and `rewardaudit probes D6` explains one defect. The
[reference](https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md) describes them all.

## Results on real checkers

We audited 185 checkers taken from open-source evaluation suites and RL training frameworks. 92 had at least one
high-severity finding. We then checked each of those 234 findings against the benchmark's own code, with a model's
help, and 213 held up. **81 of the 185 checkers have at least one confirmed high-severity defect.**

How many checkers have each defect:

```text
Negation ignored                ██████████████████████████████  39
Hedged answers accepted         ██████████████████████████▏     34
Signs and punctuation ignored   ███████████████████████▏        30
Numeric tolerance too wide      ██████████████▋                 19
Substring match                 ███████████▌                    15
Reply text run as code          ██████████                      13
Reads the thinking block        ████████▌                       11
One of several answers is read  ██████▉                          9
```

We do not say which checker has which defect.

## Use it in your setup

**From Python.**

```python
from rewardaudit import audit

result = audit(score, "data.jsonl", n=200)  # or a list of dicts, a DataFrame, a Hugging Face Dataset
result.print()
result.defects       # what is wrong, most severe first
result.findings      # every finding
result.save("out/")  # report.md and report.json
```

**verl, TRL and other reward functions.** Arguments are matched by name, so verl's
`compute_score(solution_str, ground_truth)` and TRL's `reward(prompts, completions, **columns)` work unchanged. For any
other benchmark, write a few lines that call its checker. Datasets can be JSONL, JSON, CSV, Parquet or a Hugging Face
dataset (`hf:org/name`). If the checker wants a format the tool does not try by itself, name it with
`--format '<solution>{answer}</solution>'`; if a row has several accepted answers, point to them with
`--answer-col answers.text.0 --reference-col answers.text`.

**verifiers environments.** An environment brings its own dataset:

```bash
rewardaudit audit my-env --n 200          # installed in this Python
rewardaudit audit primeintellect/gsm8k    # from the Prime Intellect Hub, in its own venv (needs uv)
```

A Hub environment is someone else's code, and auditing it installs and runs that code on your machine. Only do this
with packages you would run anyway.

**In CI.** The exit code is 0 when there is no high defect, 1 when there is, and 2 when the audit could not be done or
its rates could not be measured.
`--fail-on "right_rejected>0.1,wrong_accepted>0.01"` sets your own thresholds. `rewardaudit compare` shows where two
versions of a checker decide differently.

## Limits

RewardAudit audits checkers that score one reply against a reference answer. Multi-turn and tool-use environments and
checkers that ask a model are out of scope; it stops and says so rather than print misleading numbers. If the same
input gets different scores, it reports that and carries on. It tests a fixed set of replies, so a clean result means
none of them got a wrong score, not that the checker is correct.

## Documentation

- [Usage guide](https://github.com/xzx34/RewardAudit/blob/main/docs/guide.md): every option, reading a report, the Python API, limits.
- [Reference](https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md): all 79 probes and 22 defects, with examples.
- [Examples](https://github.com/xzx34/RewardAudit/tree/main/examples): verl, TRL and question-answering rewards.
- [Contributing](https://github.com/xzx34/RewardAudit/blob/main/CONTRIBUTING.md) · [Changelog](https://github.com/xzx34/RewardAudit/blob/main/CHANGELOG.md)

## Citation

```bibtex
@software{rewardaudit,
  author = {Xu, Zixiang},
  title  = {RewardAudit: test the checker before you train on it},
  year   = {2026},
  url    = {https://github.com/xzx34/RewardAudit}
}
```

## License

Apache-2.0
