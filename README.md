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
  <img src="https://img.shields.io/badge/needs-no%20model%20%C2%B7%20no%20GPU-6366f1?style=flat-square" alt="No model, no GPU">
</p>

<p align="center">
  <a href="#what-it-catches">What it catches</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#tested-on-real-checkers">Real checkers</a> ·
  <a href="https://github.com/xzx34/RewardAudit/blob/main/docs/guide.md">Guide</a> ·
  <a href="https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md">Reference</a>
</p>

A checker is the function that turns a model's reply into a reward. If it pays a reply that does not solve the task,
the model learns to write that reply. If it refuses a right answer written another way, the model is punished for
being right.

**RewardAudit** finds these problems before training does. Give it a checker and a dataset with reference answers.
It writes replies whose correctness is known in advance, sends them to the checker, and reports every place where the
score and the truth disagree, with the reply that shows it. No model and no GPU: a 40-row audit takes seconds.

## What it catches

Replies the probes send, and what a faulty checker does with them. The reference answer is `34`.

| Reply | Truth | A faulty checker gives | Defect |
|---|---|---|---|
| `The answer is not \boxed{34}. The answer is \boxed{68}.` | wrong | **1.0** | D7 Negation ignored |
| `The answer is either 68 or 34.` | hedged | **1.0** | D6 Hedged answers accepted |
| `<think>My first guess is 34…</think> \boxed{68}` | wrong | **1.0** | D5 Reads the thinking block |
| `\boxed{−34}` | wrong | **1.0** | D1 Signs and punctuation ignored |
| `\boxed{340}` | wrong | **1.0** | D2 Substring match |
| `\boxed{3400}` | wrong | **1.0** | D3 Numeric tolerance too wide |
| `\boxed{34.0}` | right | **0.0** | D9 Equivalent rewrites rejected |
| `\boxed{__import__('rewardaudit_probe_marker')}` | not an answer | runs it | D12 Reply text run as code |

79 probes in eight families write replies like these, and 22 defect types group what they find.
`rewardaudit probes` lists them; `rewardaudit probes D7` explains one.

<details>
<summary><b>The eight probe families</b></summary>
<br>

| Family | What the reply holds | Probes | A sound checker gives |
|---|---|---:|---|
| **N** | no answer at all: empty, a placeholder, a fixed answer | 6 | zero |
| **E** | many candidate answers at once | 8 | zero |
| **L** | text copied from the prompt | 6 | zero |
| **O** | a right and a wrong answer together: hedged, retracted, in the thinking block | 12 | zero |
| **M** | the reference answer with a definite error: sign, digit, scale, an extra character | 21 | zero |
| **Q** | the reference answer rewritten but still correct, or surrounded by long text | 18 | full score |
| **G** | the row's other accepted answers, and its reference answer written another way | 2 | full score |
| **P** | malformed or extreme input, and text that must never run as code | 6 | no error, no hang |

</details>

## Quick start

```bash
pip install rewardaudit
```

A checker that takes the last number in the reply, and a dataset of rows such as
`{"question": "Gus has 23 apples and gets 11 more. How many apples does Gus have now?", "answer": "34"}`:

```python
# reward.py
import re

def score(completion, answer):
    numbers = re.findall(r"-?\d+", completion)
    return 1.0 if numbers and numbers[-1] == answer.strip() else 0.0
```

```bash
rewardaudit audit reward.py --dataset data.jsonl
```

<p align="center">
  <img src="https://github.com/xzx34/RewardAudit/raw/main/docs/assets/demo.svg" alt="The audit's output: two high defects, D1 Signs and punctuation ignored and D6 Hedged answers accepted, each with the reply that shows it" width="100%">
</p>

Two high defects in a four-line checker. `\boxed{−34}` earns full score because the pattern does not know the Unicode
minus, and `either 68 or 34` does because only the last number counts. Each high defect's reply is sent five more times
and cut down to the shortest reply still accepted: `68 34.` shows that the words never mattered.

The example and its 40 rows are in [`examples/quickstart`](https://github.com/xzx34/RewardAudit/tree/main/examples/quickstart), and
`rewardaudit demo` runs three more without any files of yours. An audit draws up to 100 rows at random
(`--n`, `--seed`); checks on the dataset itself, such as duplicate prompts and empty answers, cover every row.

<details>
<summary><b>Reading the summary</b></summary>
<br>

| Summary row | Question | A reply it sends |
|---|---|---|
| Accepts non-answers | Can a reply that ignores the task earn full score? | an empty reply, `\boxed{}`, the prompt copied back |
| Accepts wrong answers | Does a definite error still earn full score? | off by one, sign flipped, another task's answer |
| Accepts hedged answers | Does giving two answers, one of them right, earn full score? | `either 68 or 34`, the right answer retracted |
| Rejects right answers | Does a correct answer lose the score when written differently? | `34.0`, `\dfrac{1}{2}`, an explanation before it |
| Pays wrong answers in part | Does a definite error earn more than nothing? | off by one, one word changed |
| Runs reliably | Does the checker run without errors, hangs, crashes or network use? | huge numbers, malformed input, repeated calls |

Rates are shares of calls with 95% intervals. A finding says how the checker behaved, not whether that was intended:
a strict format rule may be what you want, so read the example before you change anything.

</details>

## How it works

<table>
<tr>
<td width="25%" valign="top"><b>1 · Learn the format</b><br><br>Sends every reference answer in each format it knows, from <code>\boxed{}</code> to answer tags, and keeps the formats the checker pays. Rows whose reference is refused get no probes.</td>
<td width="25%" valign="top"><b>2 · Probe</b><br><br>Writes replies whose verdict is known in advance: non-answers, hedges, near-miss numbers, rewrites of the right answer, malformed and huge input.</td>
<td width="25%" valign="top"><b>3 · Measure</b><br><br>Compares every score with the known verdict and reports rates with 95% intervals, grouped into defects from high to low severity.</td>
<td width="25%" valign="top"><b>4 · Confirm</b><br><br>Sends each high defect's reply five more times and cuts it down to the shortest reply the checker still accepts.</td>
</tr>
</table>

The checker runs in separate worker processes with time and memory limits; the report notes errors, hangs, network
use and the modules the checker imports. `--out DIR` saves `report.md`, `report.json` and a `repro.py` that scores
the reply behind each finding again.

## Tested on real checkers

We ran RewardAudit on **185 checkers** copied from open-source evaluation suites and RL training frameworks.
92 had at least one high-severity finding. A model-assisted review of those 234 findings against each benchmark's own
code kept 213, leaving **81 checkers with at least one real high-severity defect**. Checkers with each defect:

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

We do not publish which checker has which defect.

## Connect your checker

RewardAudit needs a function that takes a reply and a reference answer and returns a score. Parameters are matched by
name, so `score(completion, answer)`, verl's `compute_score(solution_str, ground_truth)` and TRL's
`reward(prompts, completions, **columns)` work as they are. For any other benchmark, a few lines that call its checker
are enough. `--format '<solution>{answer}</solution>'` names a reply format the tool does not try by itself, and
`--answer-col answers.text.0 --reference-col answers.text` handles a list of accepted answers. Datasets can be jsonl,
json, csv, parquet or `hf:org/name[:config[:split]]`.

**From Python**

```python
from rewardaudit import audit

result = audit(score, "data.jsonl", n=200)  # or a list of dicts, a DataFrame, a Hugging Face Dataset
result.print()
result.defects       # what the checker does wrong, most severe first
result.findings      # every finding; a probe can give more than one
result.rates         # the rate behind each row of the summary
result.save("out/")  # report.md, report.json
```

**Environments.** A [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment works too, and brings its
own dataset:

```bash
rewardaudit audit my-env --n 200               # installed in this Python
rewardaudit audit primeintellect/gsm8k         # from the Prime Intellect Hub, into its own venv (needs uv)
```

An environment from the Hub is someone else's code, and this installs and runs it on your machine, as installing it
yourself would. Only do this with packages you would run anyway.

**In CI.** The exit code is 0 with no high defect, 1 with one, and 2 if the audit could not be done or the rates could
not be measured. `--fail-on "right_rejected>0.1,wrong_accepted>0.01"` fails only above the rates you choose, and
`rewardaudit compare v1.jsonl v2.jsonl` shows where two versions of a checker decide differently, from two audits
run with `--calls`.

## What it does not do

RewardAudit audits checkers that score one reply on a dataset with reference answers. If identical inputs give
different scores, it reports S1 and continues. Multi-turn and tool environments and checkers that call a model are out
of scope, and it stops and says so instead of giving misleading numbers. It tests a fixed set of replies, so a clean
result means none of these replies got through, not that the checker is sound.

## Documentation

- [Usage guide](https://github.com/xzx34/RewardAudit/blob/main/docs/guide.md): every option, how to read a report, the Python API and the limits.
- [Reference](https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md): all 79 probes and 22 defects, with examples.
- [Examples](https://github.com/xzx34/RewardAudit/tree/main/examples): verl, TRL and question answering rewards, each with ten rows.
- [Contributing](https://github.com/xzx34/RewardAudit/blob/main/CONTRIBUTING.md) and the [changelog](https://github.com/xzx34/RewardAudit/blob/main/CHANGELOG.md).

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
