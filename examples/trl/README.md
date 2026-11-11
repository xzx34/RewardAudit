# A TRL-style batch reward

Run from the repository root:

```bash
rewardaudit audit examples/trl/reward.py --dataset examples/trl/data.jsonl --n all
```

The function takes lists of prompts, plain-text completions and answers, and returns a list of scores.
It deliberately reads only the first option letter. Look for D6 and D7 in the result.
