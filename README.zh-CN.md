<p align="center">
  <img src="https://github.com/xzx34/rewardaudit/raw/main/docs/assets/banner.svg" alt="rewardaudit：先测判分函数，再用它训练" width="100%">
</p>

<p align="center">
  <a href="https://github.com/xzx34/rewardaudit/blob/main/README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/rewardaudit/"><img src="https://img.shields.io/pypi/v/rewardaudit?style=flat-square&color=14b8a6&label=pypi" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-334155?style=flat-square" alt="Python 3.11 至 3.13">
  <a href="https://github.com/xzx34/rewardaudit/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/xzx34/rewardaudit/ci.yml?style=flat-square&label=tests" alt="测试"></a>
  <a href="https://github.com/xzx34/rewardaudit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-334155?style=flat-square" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/needs-no%20model%20%C2%B7%20no%20GPU-6366f1?style=flat-square" alt="不需要模型和 GPU">
</p>

<p align="center">
  <a href="#能发现什么">能发现什么</a> ·
  <a href="#快速上手">快速上手</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#在真实-checker-上的结果">真实 checker</a> ·
  <a href="https://github.com/xzx34/rewardaudit/blob/main/docs/guide.md">使用指南（英文）</a> ·
  <a href="https://github.com/xzx34/rewardaudit/blob/main/docs/reference.md">参考（英文）</a>
</p>

checker（判分函数）把模型的回复变成奖励。它要是给没有解出题目的回复打了分，模型就会学着写这种回复；它要是因为写法不同而拒绝了正确答案，模型就会因为答对而受罚。

**rewardaudit** 在训练之前把这些问题找出来。你给它一个 checker 和一份带参考答案的数据集，它构造一批事先知道对错的回复发给 checker，凡是分数和事实不一致的地方都报告出来，并附上能说明问题的那条回复。不需要模型，也不需要 GPU，40 行数据几秒钟就能审完。

## 能发现什么

下面是探针实际发出的回复，以及一个有缺陷的 checker 会怎么判。参考答案是 `34`。

| 回复 | 实际 | 有缺陷的 checker 给出 | 缺陷 |
|---|---|---|---|
| `The answer is not \boxed{34}. The answer is \boxed{68}.` | 错 | **1.0** | D7 否定被忽略 |
| `The answer is either 68 or 34.` | 两头下注 | **1.0** | D6 两头下注的回答被接受 |
| `<think>My first guess is 34…</think> \boxed{68}` | 错 | **1.0** | D5 读取思考过程中的答案 |
| `\boxed{−34}` | 错 | **1.0** | D1 符号和标点被忽略 |
| `\boxed{340}` | 错 | **1.0** | D2 子串匹配 |
| `\boxed{3400}` | 错 | **1.0** | D3 数值容差过宽 |
| `\boxed{34.0}` | 对 | **0.0** | D9 等价改写被拒 |
| `\boxed{__import__('rewardaudit_probe_marker')}` | 不是答案 | 当作代码执行 | D12 回复文本被当作代码执行 |

79 个探针分成八族，构造的就是这类回复；发现的问题归为 22 种缺陷。`rewardaudit probes` 列出全部探针，`rewardaudit probes D7` 解释其中一项。

<details>
<summary><b>八个探针族</b></summary>
<br>

| 族 | 回复里是什么 | 探针数 | 正确的 checker 应该给 |
|---|---|---:|---|
| **N** | 根本没有答案：空回复、占位文字、固定答案 | 6 | 零分 |
| **E** | 一次给出许多候选答案 | 8 | 零分 |
| **L** | 从题目里照抄的文字 | 6 | 零分 |
| **O** | 对的和错的答案同时出现：两头下注、先给出再撤回、写在思考过程里 | 12 | 零分 |
| **M** | 带有明确错误的参考答案：符号、某一位数字、数量级、多出的字符 | 21 | 零分 |
| **Q** | 改写后仍然正确的参考答案，或者前后有很长的文字 | 18 | 满分 |
| **G** | 这一行的其他可接受答案，以及换一种写法的参考答案 | 2 | 满分 |
| **P** | 畸形或极端的输入，以及绝不能被当作代码执行的文本 | 6 | 不报错、不卡住 |

</details>

## 快速上手

```bash
pip install rewardaudit
```

一个取回复里最后一个数字的 checker，加上一份数据集，每行形如
`{"question": "Gus has 23 apples and gets 11 more. How many apples does Gus have now?", "answer": "34"}`：

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
  <img src="https://github.com/xzx34/rewardaudit/raw/main/docs/assets/demo.svg" alt="审计的输出：两个高严重度缺陷，D1 符号和标点被忽略、D6 两头下注的回答被接受，各附一条能说明问题的回复" width="100%">
</p>

四行代码的 checker，查出两个高严重度缺陷。`\boxed{−34}` 拿了满分，因为正则不认识 Unicode 减号；`either 68 or 34` 也拿了满分，因为它只看最后一个数字。每个高严重度缺陷的回复都会再发五次，并且缩短到仍然被接受的最短形式：`68 34.` 说明那些文字从来没起过作用。

这个例子和它的 40 行数据在 [`examples/quickstart`](https://github.com/xzx34/rewardaudit/tree/main/examples/quickstart)，`rewardaudit demo` 不需要你提供任何文件就能再演示三个。每次审计最多随机抽 100 行（`--n`、`--seed`）；对数据集本身的检查，比如重复的题目和空答案，总是覆盖全部行。

<details>
<summary><b>怎么读汇总</b></summary>
<br>

| 汇总行 | 问的是什么 | 发出的回复举例 |
|---|---|---|
| Accepts non-answers | 不理会题目的回复能拿满分吗？ | 空回复、`\boxed{}`、照抄题目 |
| Accepts wrong answers | 明确错误的答案还能拿满分吗？ | 差一、符号反了、另一道题的答案 |
| Accepts hedged answers | 同时给出两个答案、其中一个对，能拿满分吗？ | `either 68 or 34`、先给出正确答案再撤回 |
| Rejects right answers | 正确答案换个写法就丢分吗？ | `34.0`、`\dfrac{1}{2}`、前面加一段解释 |
| Pays wrong answers in part | 明确错误的答案能拿到部分分吗？ | 差一、改掉一个词 |
| Runs reliably | checker 运行时会报错、卡住、崩溃或者联网吗？ | 极大的数、畸形输入、重复调用 |

比例是调用次数的占比，附 95% 区间。一条发现说的是 checker 实际怎么做的，不是它是否有意为之：严格的格式要求可能正是你想要的，改代码之前先看一下例子。

</details>

## 工作原理

<table>
<tr>
<td width="25%" valign="top"><b>1 · 摸清格式</b><br><br>把每个参考答案用它认识的各种格式发一遍，从 <code>\boxed{}</code> 到答案标签，留下 checker 给分的格式。参考答案本身拿不到满分的行不发探针。</td>
<td width="25%" valign="top"><b>2 · 发探针</b><br><br>构造事先知道对错的回复：没有答案的、两头下注的、差一点的数字、正确答案的改写、畸形和超大的输入。</td>
<td width="25%" valign="top"><b>3 · 计量</b><br><br>把每个分数和已知的对错比对，给出带 95% 区间的比例，按严重度从高到低归为缺陷。</td>
<td width="25%" valign="top"><b>4 · 确认</b><br><br>每个高严重度缺陷的回复再发五次，并缩短到 checker 仍然接受的最短形式。</td>
</tr>
</table>

checker 运行在独立的工作进程里，有时间和内存限制；报告会记下报错、卡住、联网，以及 checker 导入了哪些模块。`--out DIR` 会保存 `report.md`、`report.json`，以及一个把每条发现背后的回复重新打分的 `repro.py`。

## 在真实 checker 上的结果

我们用 rewardaudit 审了 **185 个 checker**，它们取自开源的评测套件和强化学习训练框架。其中 92 个至少有一条高严重度发现。对这 234 条发现逐条对照各个 benchmark 自己的代码，用模型辅助复核，保留了 213 条，**81 个 checker 至少有一个经过复核的高严重度缺陷**。各缺陷涉及的 checker 数：

```text
██████████████████████████████  39  否定被忽略
██████████████████████████▏     34  两头下注的回答被接受
███████████████████████▏        30  符号和标点被忽略
██████████████▋                 19  数值容差过宽
███████████▌                    15  子串匹配
██████████                      13  回复文本被当作代码执行
████████▌                       11  读取思考过程中的答案
██████▉                          9  多个答案只读其中一个
```

我们不公开具体哪个 checker 有哪个缺陷。

## 接入你的 checker

rewardaudit 需要一个函数，接收回复和参考答案，返回分数。参数按名字匹配，所以 `score(completion, answer)`、verl 的 `compute_score(solution_str, ground_truth)` 和 TRL 的 `reward(prompts, completions, **columns)` 都能直接用。其他 benchmark 写几行调用它的 checker 即可。`--format '<solution>{answer}</solution>'` 指定工具不会自己尝试的回复格式，`--answer-col answers.text.0 --reference-col answers.text` 用来处理多个可接受答案。数据集可以是 jsonl、json、csv、parquet，或者 `hf:org/name[:config[:split]]`。

**在 Python 里用**

```python
from rewardaudit import audit

result = audit(score, "data.jsonl", n=200)  # 也可以是字典列表、DataFrame 或 Hugging Face Dataset
result.print()
result.defects       # checker 的缺陷，最严重的在前
result.findings      # 全部发现；一个探针可以给出不止一条
result.rates         # 汇总里每一行背后的比例
result.save("out/")  # report.md、report.json
```

**环境。** [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) 环境也可以审，它自带数据集：

```bash
rewardaudit audit my-env --n 200               # 已安装在当前 Python 里
rewardaudit audit primeintellect/gsm8k         # 从 Prime Intellect Hub 装进单独的 venv（需要 uv）
```

Hub 上的环境是别人的代码，这样做会在你的机器上安装并运行它，和你自己安装一样。只对你本来就愿意运行的包这样做。

**在 CI 里用。** 没有高严重度缺陷时退出码是 0，有则是 1，无法审计或无法计量比例时是 2。`--fail-on "right_rejected>0.1,wrong_accepted>0.01"` 只在超过你设定的比例时失败。`rewardaudit compare v1.jsonl v2.jsonl` 对比两次用 `--calls` 跑出的审计，告诉你同一个 checker 的两个版本在哪里判得不一样。

## 不做什么

rewardaudit 审的是对单条回复、按参考答案打分的 checker。同样的输入给出不同分数时，它报告 S1 并继续。多轮对话、工具调用环境和调用模型打分的 checker 不在范围内，遇到时它会停下来说明原因，而不是给出误导的数字。它测试的是一组固定的回复，所以审计结果干净只说明这些回复都被判对了，不说明 checker 没有问题。

## 文档

- [使用指南](https://github.com/xzx34/rewardaudit/blob/main/docs/guide.md)（英文）：所有选项、怎么读报告、Python 接口和局限。
- [参考](https://github.com/xzx34/rewardaudit/blob/main/docs/reference.md)（英文）：全部 79 个探针和 22 种缺陷，附例子。
- [例子](https://github.com/xzx34/rewardaudit/tree/main/examples)：verl、TRL 和问答的奖励函数，每个 10 行数据。
- [贡献指南](https://github.com/xzx34/rewardaudit/blob/main/CONTRIBUTING.md)和[更新日志](https://github.com/xzx34/rewardaudit/blob/main/CHANGELOG.md)。

## 引用

```bibtex
@software{rewardaudit,
  author = {Xu, Zixiang},
  title  = {rewardaudit: test the checker before you train on it},
  year   = {2026},
  url    = {https://github.com/xzx34/rewardaudit}
}
```

## 许可证

Apache-2.0
