<p align="center">
  <img src="https://github.com/xzx34/RewardAudit/raw/main/docs/assets/banner.svg" alt="RewardAudit：先测 checker，再用它训练" width="100%">
</p>

<p align="center">
  <a href="https://github.com/xzx34/RewardAudit/blob/main/README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/rewardaudit/"><img src="https://img.shields.io/pypi/v/rewardaudit?style=flat-square&color=14b8a6&label=pypi" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-334155?style=flat-square" alt="Python 3.11 至 3.13">
  <a href="https://github.com/xzx34/RewardAudit/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/xzx34/RewardAudit/ci.yml?style=flat-square&label=tests" alt="测试"></a>
  <a href="https://github.com/xzx34/RewardAudit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-334155?style=flat-square" alt="Apache-2.0"></a>
</p>

<p align="center">
  <a href="#快速上手">快速上手</a> ·
  <a href="#结果解读">结果解读</a> ·
  <a href="#检测内容">检测内容</a> ·
  <a href="#在真实-checker-上的结果">在真实 checker 上的结果</a> ·
  <a href="https://github.com/xzx34/RewardAudit/blob/main/docs/guide.md">使用指南</a> ·
  <a href="https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md">参考文档</a>
</p>

在基于可验证奖励的强化学习中，模型学到什么，很大程度上取决于 **checker**：一段把模型输出与参考答案进行比较并打分的代码。checker 一旦判错，训练就会朝错误的方向优化。比如，`either 68 or 34` 这样两头下注的输出如果能拿到满分，模型就会学会两头下注；正确答案写成 `34.0` 如果被判为错误，模型就会学会回避这种本来正确的写法。

RewardAudit 用测试普通函数的方式来测试 checker：构造一批应得分数已知的输出交给 checker 打分，再找出其中判错的部分。使用时只需提供 checker 和带参考答案的数据集，不需要模型，也不需要 GPU，小规模的审计几秒钟即可完成。

## 快速上手

安装：

```bash
pip install rewardaudit
```

审计你自己的 checker：在这里，checker 是一个封装好的 Python 函数，接收模型的输出和参考答案，并返回分数。下面这个 checker 会取出输出中的最后一个数，与参考答案比较并打分：

```python
# reward.py
import re

def score(completion, answer):
    numbers = re.findall(r"-?\d+", completion)
    return 1.0 if numbers and numbers[-1] == answer.strip() else 0.0
```

数据集的每一行包含一道题目及其参考答案：

```json
{"question": "Gus has 23 apples and gets 11 more. How many apples does Gus have now?", "answer": "34"}
{"question": "Ben has 41 cards and gives away 7. How many cards does Ben have left?", "answer": "34"}
```

运行审计：

```bash
rewardaudit audit reward.py --dataset data.jsonl
```

完整的示例文件（共 40 道题）见 [`examples/quickstart`](https://github.com/xzx34/RewardAudit/tree/main/examples/quickstart)。

## 结果解读

<p align="center">
  <img src="https://github.com/xzx34/RewardAudit/raw/main/docs/assets/demo.svg" alt="审计输出：问题概况、六行汇总，以及两个高严重度缺陷及其示例输出" width="100%">
</p>

输出分为三部分：

- **Result**：checker 是否存在问题，以及高、中严重度的问题各有几个。
- **六行汇总**：每一行检查 checker 的一类行为。*yes* 表示存在问题，百分比是问题出现的频率，方括号内是 95% 置信区间。
- **Defects**：checker 的缺陷，按严重度从高到低排列，每个缺陷附有一条示例输出。

这个 checker 有两个高严重度缺陷：

- `\boxed{−34}` 得到了满分：正则表达式不识别 Unicode 减号，只读到了 34。
- `The answer is either 68 or 34.` 得到了满分：checker 只看最后一个数。把这条输出删减到仍能得满分的最短形式，只剩下 `68 34.`。

每个高严重度缺陷的示例输出都会再发送五次，以确认结果稳定。加上 `--full` 可以查看全部发现，加上 `--out DIR` 会保存完整报告和一个复现脚本。

## 检测内容

RewardAudit 根据每道题的参考答案构造输出。其中一部分没有解出题目，正确的 checker 应当给零分；另一部分答案正确，只是写法不同，应当给满分。这些输出由 79 个探针生成，分为八族：

| 族 | 输出内容 | 示例（参考答案为 `34`） | 应得分数 |
|---|---|---|---|
| **N** | 不含答案 | 空输出、`\boxed{}` | 0 |
| **E** | 同时列出大量候选答案 | `Possible answers: -20, -19, …, 200` | 0 |
| **L** | 从题目中照抄的文字 | 题目原文 | 0 |
| **O** | 同时包含正确答案和错误答案 | `The answer is either 68 or 34.` | 0 |
| **M** | 带有明确错误的答案 | `\boxed{−34}`、`\boxed{340}` | 0 |
| **Q** | 写法不同的正确答案 | `\boxed{34.0}`、长篇解释之后的 `34` | 满分 |
| **G** | 该题的其他可接受答案 | 数据集中列出的别名 | 满分 |
| **P** | 畸形或极端的输入 | `9**9**9`、未闭合的 `\boxed{` | 不报错、不卡死 |

一次审计分为四步：

1. **确定格式。** 把每个参考答案以各种常见格式（从 `\boxed{}` 到答案标签）发送给 checker，保留它接受的格式。
2. **发送探针。** 以这些格式发送探针生成的输出。checker 在独立进程中运行，并受时间和内存限制，崩溃或卡死会被记为一条发现，不会中断审计。
3. **比对分数。** 将每个分数与应得分数比对，并把发现归入 22 类缺陷，例如“两头下注的答案被接受”“等价改写被拒绝”。
4. **确认结果。** 将每个高严重度缺陷的示例输出再发送五次，并删减为 checker 仍然接受的最短形式。

`rewardaudit probes` 列出全部探针，`rewardaudit probes D6` 查看某一类缺陷的说明，完整说明见[参考文档](https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md)（英文）。

## 在真实 checker 上的结果

我们审计了 185 个 checker，它们来自开源的评测套件和强化学习训练框架，其中 92 个至少有一条高严重度发现。随后，我们借助模型，将这 234 条发现逐条对照相应 benchmark 的源代码进行核实，其中 213 条成立。**185 个 checker 中，有 81 个至少存在一个经过核实的高严重度缺陷。**

各类缺陷涉及的 checker 数量：

```text
██████████████████████████████  39  否定被忽略
██████████████████████████▏     34  两头下注的答案被接受
███████████████████████▏        30  符号和标点被忽略
██████████████▋                 19  数值容差过宽
███████████▌                    15  子串匹配
██████████                      13  输出文本被当作代码执行
████████▌                       11  读取了思考过程中的答案
██████▉                          9  多个答案只读取其中一个
```

其中几个例子：

- **lm-evaluation-harness**（MGSM）：开启去除标点后，参考答案为 `3` 时，`Die Antwort lautet -3.` 被判为正确。
- **OpenAI grade-school-math**（GSM8K）：参考答案为 `30` 时，`#### 30 … Wait, that is not right. … The correct result is #### 60` 被判为正确，因为它只保留第一个标记出的数字。
- **Inspect AI**（`match_str`，数值匹配）：参考答案为 `30` 时，`The answer is either 60 or 30.` 被判为正确。
- **TheoremQA**：数字解析失败时，会把输出文本当作 Python 代码执行。

[已确认的缺陷](https://github.com/xzx34/RewardAudit/blob/main/docs/findings.md)列出了 30 个项目中的 35 个此类问题，每条都附有被审计版本的源码链接（英文）。

## 接入你的项目

**在 Python 中调用。**

```python
from rewardaudit import audit

result = audit(score, "data.jsonl", n=200)  # 也可以传入字典列表、DataFrame 或 Hugging Face Dataset
result.print()
result.defects       # 缺陷，按严重度从高到低排列
result.findings      # 全部发现
result.save("out/")  # 保存 report.md 和 report.json
```

**verl、TRL 及其他奖励函数。** 参数按名称匹配，因此 verl 的 `compute_score(solution_str, ground_truth)` 和 TRL 的 `reward(prompts, completions, **columns)` 无需修改即可审计。对于其他 benchmark，只需写几行代码调用它的 checker。数据集支持 JSONL、JSON、CSV、Parquet 和 Hugging Face 数据集（`hf:org/name`）。如果 checker 要求的格式不在工具默认尝试的范围内，可以用 `--format '<solution>{answer}</solution>'` 指定；如果一道题有多个可接受答案，可以用 `--answer-col answers.text.0 --reference-col answers.text` 指定。

**verifiers 环境。** 环境自带数据集：

```bash
rewardaudit audit my-env --n 200          # 已安装在当前 Python 环境中
rewardaudit audit primeintellect/gsm8k    # 从 Prime Intellect Hub 安装到独立的 venv（需要 uv）
```

Hub 上的环境是第三方代码，审计时会在你的机器上安装并运行。请只对你本来就愿意运行的包这样做。

**在 CI 中使用。** 没有高严重度缺陷时退出码为 0，存在时为 1，无法完成审计或无法计算比例时为 2。`--fail-on "right_rejected>0.1,wrong_accepted>0.01"` 可以自定义阈值，`rewardaudit compare` 可以对比同一个 checker 的两个版本在哪些输出上判分不同。

## 局限

RewardAudit 适用于针对单条输出、按参考答案打分的 checker。多轮对话、工具调用环境以及调用模型打分的 checker 不在适用范围内；遇到这类情况时，它会停止并说明原因，而不会给出误导性的数字。如果同一输入得到不同的分数，它会报告这一情况并继续审计。由于测试的是一组固定的输出，审计结果干净只说明这些输出都得到了正确的分数，并不代表 checker 没有问题。

## 文档

- [使用指南](https://github.com/xzx34/RewardAudit/blob/main/docs/guide.md)：全部选项、报告解读、Python 接口和局限。
- [参考文档](https://github.com/xzx34/RewardAudit/blob/main/docs/reference.md)：全部 79 个探针和 22 类缺陷，附示例。
- [已确认的缺陷](https://github.com/xzx34/RewardAudit/blob/main/docs/findings.md)：具名开源 checker 中的缺陷，每条附源码链接。
- [示例](https://github.com/xzx34/RewardAudit/tree/main/examples)：verl、TRL 和问答任务的奖励函数。
- [贡献指南](https://github.com/xzx34/RewardAudit/blob/main/CONTRIBUTING.md) · [更新日志](https://github.com/xzx34/RewardAudit/blob/main/CHANGELOG.md)

以上文档目前只有英文版。

## 引用

```bibtex
@software{rewardaudit,
  author = {Xu, Zixiang},
  title  = {RewardAudit: test the checker before you train on it},
  year   = {2026},
  url    = {https://github.com/xzx34/RewardAudit}
}
```

## 许可证

Apache-2.0
