"""A small checker of the kind many projects start with: the last number in the reply must equal the answer."""

import re


def score(completion: str, answer: str) -> float:
    numbers = re.findall(r"-?\d+", completion)
    return 1.0 if numbers and numbers[-1] == answer.strip() else 0.0
