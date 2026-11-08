"""The function adapter's calling conventions, reference handling and score coercion."""

import json
import sys
import textwrap
import types
from dataclasses import dataclass

import pytest

from rewardaudit.adapters.base import RawScore
from rewardaudit.adapters.dataset import _render_prompt, load_tasks
from rewardaudit.adapters.function_adapter import FunctionAdapter, _coerce, detect_style
from rewardaudit.errors import AuditError
from rewardaudit.types import Task

TASK = Task("t1", "What is 6 times 7?", "42", info={"full_solution": "6 * 7 = 42. #### 42", "data_source": "toy",
                                                     "reference": '{"target": 42, "numbers": [6, 7]}'})


def _adapter(tmp_path, code: str, func: str, **spec) -> FunctionAdapter:
    path = tmp_path / f"m_{func}_{abs(hash(code)) % 10**8}.py"
    path.write_text(textwrap.dedent(code))
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"prompt": TASK.prompt, "answer": TASK.answer}) + "\n")
    return FunctionAdapter({"path": f"{path}:{func}", "dataset": str(data), **spec})


def _echo(tmp_path, signature: str, body: str = "return 1.0", **spec) -> FunctionAdapter:
    """A function that records its arguments in CALLS and returns `body`."""
    code = f"CALLS = []\n\ndef f({signature}):\n    CALLS.append(dict(locals()))\n    {body}\n"
    return _adapter(tmp_path, code, "f", **spec)


def _calls(a: FunctionAdapter) -> list[dict]:
    return a.fn.__globals__["CALLS"]


# --- style detection ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("signature,style", [
    ("prompt, completion, answer", "bare"),
    ("solution_str, ground_truth", "verl"),
    ("data_source, solution_str, ground_truth, extra_info=None", "verl"),
    ("predict_str, ground_truth, use_boxed=True", "verl"),
    ("model_output, ground_truth, timeout_score=0", "verl"),
    ("model_response, gt_answer, fast=False", "bare"),
    ("response, label, metadata=None", "bare"),
    ("prompts, completions, **kw", "trl"),
    ("queries, prompts, labels, **kwargs", "openrlhf"),
    ("prompt, completions, prompt_ids, completion_ids, answer, **kwargs", "areal"),
    ("task_info, action", "rllm"),
    ("reward_input, format_weight=0.5", "easyr1"),
    ("reward_inputs, format_weight=0.1", "easyr1"),
    ("x, y", "positional"),
])
def test_detect_style(tmp_path, signature, style):
    assert detect_style(_echo(tmp_path, signature).fn) == style


# --- binding by name and by position ------------------------------------------------------------------------

@pytest.mark.parametrize("comp_name", ["solution_str", "completion", "response", "predict_str", "model_output",
                                       "model_response", "prediction"])
def test_completion_aliases(tmp_path, comp_name):
    a = _echo(tmp_path, f"{comp_name}, ground_truth")
    a.score_raw(TASK, "C")
    assert _calls(a)[-1] == {comp_name: "C", "ground_truth": "42"}


@pytest.mark.parametrize("ref_name", ["ground_truth", "answer", "gold", "label", "gt_answer", "target"])
def test_reference_aliases(tmp_path, ref_name):
    a = _echo(tmp_path, f"response, {ref_name}")
    a.score_raw(TASK, "C")
    assert _calls(a)[-1] == {"response": "C", ref_name: "42"}


def test_unrecognised_leading_parameters_bind_by_position(tmp_path):
    a = _echo(tmp_path, "data_source, text, gt, extra_info=None")
    a.score_raw(TASK, "C")
    assert _calls(a)[-1] == {"data_source": "toy", "text": "C", "gt": "42", "extra_info": {}}
    b = _echo(tmp_path, "pred_text, answer, strict=True")
    b.score_raw(TASK, "C")
    assert _calls(b)[-1] == {"pred_text": "C", "answer": "42", "strict": True}


def test_metadata_and_info_are_copies(tmp_path):
    a = _echo(tmp_path, "response, label, metadata=None", body="metadata['seen'] = True\n    return 1.0")
    a.score_raw(TASK, "C")
    a.score_raw(TASK, "C")
    assert "seen" not in TASK.info and _calls(a)[-1]["metadata"]["full_solution"] == TASK.info["full_solution"]


def test_kwargs_reach_named_parameters(tmp_path):
    a = _echo(tmp_path, "model_response, gt_answer, fast=False", kwargs={"fast": True, "unused": 1})
    a.score_raw(TASK, "C")
    assert _calls(a)[-1] == {"model_response": "C", "gt_answer": "42", "fast": True}


# --- framework conventions ----------------------------------------------------------------------------------

def test_openrlhf(tmp_path):
    a = _echo(tmp_path, "queries, prompts, labels, **kwargs",
              body="return {'rewards': [1.0], 'scores': [1.0], 'extra_logs': {'acc': 1.0}}")
    assert a.score_raw(TASK, " C").score == 1.0
    call = _calls(a)[-1]
    assert call["queries"] == [TASK.prompt + " C"] and call["prompts"] == [TASK.prompt] and call["labels"] == ["42"]


def test_areal(tmp_path):
    a = _echo(tmp_path, "prompt, completions, prompt_ids, completion_ids, answer, **kwargs",
              reference_col="full_solution")
    a.score_raw(TASK, "C")
    call = _calls(a)[-1]
    assert (call["prompt"], call["completions"], call["prompt_ids"], call["completion_ids"]) == (TASK.prompt, "C", None, None)
    assert call["answer"] == TASK.info["full_solution"] and call["kwargs"]["data_source"] == "toy"


def test_rllm(tmp_path):
    code = """
    from dataclasses import dataclass, field
    @dataclass
    class RewardOutput:
        reward: float
        is_correct: bool = False
        metadata: dict = field(default_factory=dict)
    SEEN = []
    def math_reward_fn(task_info, action):
        SEEN.append((task_info, action))
        return RewardOutput(reward=1.0 if action == task_info["ground_truth"] else 0.0, is_correct=True)
    """
    a = _adapter(tmp_path, code, "math_reward_fn")
    assert a.style == "rllm" and a.score_raw(TASK, "42").score == 1.0 and a.score_raw(TASK, "41").score == 0.0
    info, action = a.fn.__globals__["SEEN"][-1]
    assert action == "41" and info["problem"] == TASK.prompt and info["data_source"] == "toy"


def test_easyr1_batch_and_sequential(tmp_path):
    body = "return {'overall': 0.9, 'format': 1.0, 'accuracy': 0.0}"
    seq = _echo(tmp_path, "reward_input, format_weight=0.1", body=body)
    batch = _echo(tmp_path, "reward_inputs, format_weight=0.1", body=f"return [{body[7:]}]")
    for a in (seq, batch):
        r = a.score_raw(TASK, "the answer is 42")
        assert r.score == 0.9 and r.sub_scores == {"overall": 0.9, "format": 1.0, "accuracy": 0.0}
    item = {"response": "the answer is 42", "response_length": 4, "ground_truth": "42"}
    assert _calls(seq)[-1]["reward_input"] == item and _calls(batch)[-1]["reward_inputs"] == [item]


def test_trl_messages_from_a_typing_annotation(tmp_path):
    """list[Dict[str, str]] (open-r1's len_reward) asks for chat messages as list[dict] does; list[str] for plain text."""
    body = "    return [float(completions[0][0]['content'] == 'C')]\n"
    chat = _adapter(tmp_path, f"from typing import Dict\n\ndef f(completions: list[Dict[str, str]], **kwargs):\n{body}", "f")
    assert chat.conversational is True and chat.score_raw(TASK, "C").score == 1.0
    text = _adapter(tmp_path, "from typing import List\n\ndef f(completions: List[str], **kwargs):\n    return [1.0]\n", "f")
    assert text.conversational is False


def test_trl_columns_include_the_reference(tmp_path):
    a = _echo(tmp_path, "completions, solution, **kwargs", body="return [1.0]", reference_col="full_solution")
    assert a.score_raw(TASK, "C").score == 1.0
    call = _calls(a)[-1]
    assert call["solution"] == [TASK.info["full_solution"]] and call["kwargs"]["data_source"] == ["toy"]


# --- reference ---------------------------------------------------------------------------------------------

def test_ground_truth_json(tmp_path):
    a = _echo(tmp_path, "solution_str, ground_truth", reference_col="reference", ground_truth_json=True)
    a.score_raw(TASK, "C")
    assert _calls(a)[-1]["ground_truth"] == {"target": 42, "numbers": [6, 7]}
    plain = _echo(tmp_path, "solution_str, ground_truth", reference_col="reference")
    plain.score_raw(TASK, "C")
    assert _calls(plain)[-1]["ground_truth"] == TASK.info["reference"]  # a JSON string stays a string by default
    bad = _echo(tmp_path, "solution_str, ground_truth", ground_truth_json=True)
    with pytest.raises(ValueError, match="not JSON"):
        bad.score_raw(Task("t", "q", "forty-two"), "C")


def test_annotation_asks_for_structured_ground_truth(tmp_path):
    a = _echo(tmp_path, "solution_str: str, ground_truth: dict[str, int]", reference_col="reference")
    a.score_raw(TASK, "C")
    assert _calls(a)[-1]["ground_truth"] == {"target": 42, "numbers": [6, 7]}
    union = _echo(tmp_path, "solution_str, ground_truth: str | list[str]", reference_col="reference")
    union.score_raw(TASK, "C")
    assert isinstance(_calls(union)[-1]["ground_truth"], str)


def test_structured_columns_pass_as_they_are(tmp_path):
    t = Task("t", "q", '{"target": ["a"]}', info={"reward_model": {"ground_truth": {"target": ["a"]}, "style": "rule"}})
    a = _echo(tmp_path, "solution_str, ground_truth")
    a.score_raw(t, "C")
    assert _calls(a)[-1]["ground_truth"] == {"target": ["a"]}  # verl parquet layout, found without configuration
    b = _echo(tmp_path, "solution_str, ground_truth", reference_col="reward_model.style")
    b.score_raw(t, "C")
    assert _calls(b)[-1]["ground_truth"] == "rule"


def test_reference_template_reads_columns(tmp_path):
    t = Task("t", "q", "1998", info={"golden_answers": '["1998", "in 1998"]', "ids": [3, 4]})
    a = _echo(tmp_path, "solution_str, ground_truth", reference_template='{{"target": {golden_answers}}}',
              ground_truth_json=True)
    a.score_raw(t, "C")
    assert _calls(a)[-1]["ground_truth"] == {"target": ["1998", "in 1998"]}
    b = _echo(tmp_path, "solution_str, ground_truth", reference_template="{answer} {ids}")
    b.score_raw(t, "C")
    assert _calls(b)[-1]["ground_truth"] == "1998 [3, 4]"


def test_prompt_in_completion(tmp_path):
    a = _echo(tmp_path, "solution_str, ground_truth", prompt_in_completion=True)
    a.score_raw(TASK, " <answer>42</answer>")
    assert _calls(a)[-1]["solution_str"] == TASK.prompt + " <answer>42</answer>"


# --- dataset ----------------------------------------------------------------------------------------------

def test_dataset_dotted_columns_and_structured_gold(tmp_path):
    rows = [{"prompt": [{"role": "user", "content": "Find the year."}], "data_source": "nq",
             "reward_model": {"ground_truth": {"target": ["1998"]}, "style": "rule"}, "extra_info": {"index": 0}}]
    path = tmp_path / "verl.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (t,), _ = load_tasks(str(path))
    assert t.prompt == "Find the year." and json.loads(t.answer) == {"target": ["1998"]}
    assert t.info["reward_model"]["style"] == "rule" and t.info["extra_info"] == {"index": 0}
    (u,), _ = load_tasks(str(path), answer_col="reward_model.style")
    assert u.answer == "rule"
    with pytest.raises(AuditError, match="no column"):
        load_tasks(str(path), answer_col="reward_model.missing")


ROWS = [{"prompt": "p0", "answer": "a0"}, {"prompt": "p1", "answer": "a1"}]


def _csv(path):
    path.write_text("prompt,answer\np0,a0\np1,a1\n", encoding="utf-8")


def _parquet(path):
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet").write_table(pa.Table.from_pylist(ROWS), path)


@pytest.mark.parametrize("name, write", [
    ("rows.jsonl", lambda p: p.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n\n", encoding="utf-8")),  # a blank line is no row
    ("rows.json", lambda p: p.write_text(json.dumps(ROWS), encoding="utf-8")),
    ("wrapped.json", lambda p: p.write_text(json.dumps({"version": 1, "data": ROWS}), encoding="utf-8")),
    ("rows.csv", _csv),
    ("rows.parquet", _parquet),
    ("ROWS.JSONL", lambda p: p.write_text(json.dumps(ROWS[0]) + "\n" + json.dumps(ROWS[1]), encoding="utf-8")),
])
def test_dataset_files(tmp_path, name, write):
    write(tmp_path / name)
    assert [(t.id, t.prompt, t.answer) for t in load_tasks(str(tmp_path / name))[0]] == [("0", "p0", "a0"), ("1", "p1", "a1")]


@pytest.mark.parametrize("source, split, expected", [
    ("hf:org/name", None, ("org/name", None, "train")),
    ("hf:org/name", "test", ("org/name", None, "test")),
    ("hf:org/name:main", None, ("org/name", "main", "train")),
    ("hf:org/name:main:validation", "test", ("org/name", "main", "validation")),
    ("hf:org/name::validation", None, ("org/name", None, "validation")),
])
def test_a_hugging_face_name_gives_its_config_and_split(monkeypatch, source, split, expected):
    asked = []

    def load_dataset(name, config, split):
        asked.append((name, config, split))
        return ROWS

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    assert [t.answer for t in load_tasks(source, split=split)[0]] == ["a0", "a1"] and asked == [expected]


def test_chat_prompts_and_a_system_prompt():
    assert _render_prompt([{"role": "system", "content": "s"}, {"role": "user", "content": "first"},
                           {"role": "assistant", "content": "a"}, {"role": "user", "content": "last"}])[0] == "last"
    assert _render_prompt("plain") == ("plain", None)


def test_a_row_without_a_column_says_which(tmp_path):
    (tmp_path / "rows.jsonl").write_text(json.dumps(ROWS[0]) + "\n" + json.dumps({"prompt": "p1"}), encoding="utf-8")
    with pytest.raises(AuditError, match="Row 1 of .* has no column 'answer'"):
        load_tasks(str(tmp_path / "rows.jsonl"))


def test_rows_with_an_empty_reference_answer_are_left_out_and_counted(tmp_path):
    answers = ["a0", "", "  \n", None, [], 0, "a6", ["x"], {}]
    rows = [{"prompt": f"p{i}", "answer": a} for i, a in enumerate(answers)]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tasks, left_out = load_tasks(str(tmp_path / "rows.jsonl"))
    assert left_out == 4 and [(t.id, t.answer) for t in tasks] == [("0", "a0"), ("5", "0"), ("6", "a6"), ("7", '["x"]'), ("8", "{}")]
    squad = [{"question": "q0", "answers": {"text": ["Paris"]}}, {"question": "q1", "answers": {"text": []}}]
    (tmp_path / "squad.jsonl").write_text("\n".join(json.dumps(r) for r in squad), encoding="utf-8")
    tasks, left_out = load_tasks(str(tmp_path / "squad.jsonl"), answer_col="answers.text.0")  # an empty list: no item 0
    assert [t.answer for t in tasks] == ["Paris"] and left_out == 1
    pairs = [{"question": "q0", "answers": [["yes", ["c1"]]]}, {"question": "q1", "answers": []}]  # [answer, conditions]
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in pairs), encoding="utf-8")
    tasks, left_out = load_tasks(str(tmp_path / "pairs.jsonl"), answer_col="answers.0.0")  # nothing below a missing item
    assert [t.answer for t in tasks] == ["yes"] and left_out == 1
    with pytest.raises(AuditError, match="The dataset has no column 'answers.0.text'"):  # a column absent is still absent
        load_tasks(str(tmp_path / "pairs.jsonl"), answer_col="answers.0.text")


def test_a_dataset_whose_answers_are_all_empty_says_so(tmp_path):
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps({"prompt": f"p{i}", "answer": ""}) for i in range(3)), encoding="utf-8")
    with pytest.raises(AuditError, match=r"All 3 rows of .*rows.jsonl have an empty reference answer in 'answer'\. .*--answer-col"):
        load_tasks(str(tmp_path / "rows.jsonl"))


# --- coercion ----------------------------------------------------------------------------------------------

@dataclass
class _Out:
    reward: float
    is_correct: bool = True


class _Tensor:
    def __init__(self, *v: float) -> None:
        self.v = v

    def item(self) -> float:
        if len(self.v) != 1:
            raise ValueError("more than one element")
        return self.v[0]


@pytest.mark.parametrize("result,score", [
    (1, 1.0), (True, 1.0), ([0.5], 0.5), ((True, False, "13"), 1.0), (({"formatted": True}, 1.0), 1.0),
    ({"overall": 0.9, "format": 1.0, "accuracy": 0.0}, 0.9), ({"score": 1.0, "acc": True, "pred": "x"}, 1.0),
    ({"reward": 0.5}, 0.5), ({"rewards": [1.0], "scores": [0.0], "extra_logs": {}}, 1.0),
    ({"format": 1.0, "accuracy": 1.0}, 1.0), (_Out(0.25), 0.25), (_Tensor(0.75), 0.75), ({"score": _Tensor(1.0)}, 1.0),
])
def test_coerce(result, score):
    assert _coerce(result).score == score


def test_coerce_sub_scores_and_errors():
    assert _coerce({"score": 1.0, "acc": True, "pred": "13"}) == RawScore(1.0, {"score": 1.0, "acc": 1.0})
    for bad in (None, [1.0, 0.0], ("a", "b"), {"pred": "x"}, "1.0", _Tensor(1.0, 2.0)):
        with pytest.raises(ValueError):
            _coerce(bad)
