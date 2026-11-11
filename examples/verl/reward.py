"""A numeric reward with the compute_score interface used by verl.

This checker has one deliberate defect: it compares absolute values,
so a reply with the wrong sign earns full score.
The dataset supplies data_source and reward_model.ground_truth.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return a total and a correctness part for one reply."""
    text = solution_str.strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        text = text[7:-1].strip()

    try:
        predicted = float(text)
        expected = float(ground_truth)
    except ValueError:
        return {"score": 0.0, "correctness": 0.0}

    # Deliberate defect: the sign is part of the answer.
    correct = float(abs(predicted) == abs(expected))
    return {"score": correct, "correctness": correct}
