"""A QA checker given a list of accepted answers.

This checker has one deliberate defect: it reads only the first answer
in the list, so the other accepted names lose the score.
The audit writes replies from answers.text.0 and passes answers.text whole.
"""


def score(completion, answer: list[str]):
    """Compare one answer field with the accepted names."""
    text = completion.strip()
    if text.startswith("\\boxed{") and text.endswith("}"):
        text = text[7:-1].strip()

    text = text.casefold().rstrip(".")
    accepted = [name.strip().casefold().rstrip(".") for name in answer]

    # Deliberate defect: every name in accepted should be eligible.
    expected = accepted[0]
    if text == expected:
        return 1.0
    return 0.0
