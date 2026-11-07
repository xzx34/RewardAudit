"""Audit a function from Python: python examples/python_api.py"""

from rewardaudit import audit


def score(completion: str, answer: str) -> float:
    return 1.0 if answer in completion else 0.0


rows = [{"prompt": f"What is {a} + {a + 1}?", "answer": str(2 * a + 1)} for a in range(10, 60)]

result = audit(score, rows, n=20)
result.print()

print(result.rates)  # the share of calls on each headline
for finding in result.findings[:3]:
    print(finding["severity"], finding["name"], f"{finding['rate']:.0%}")
