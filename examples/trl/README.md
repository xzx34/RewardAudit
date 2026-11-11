# A TRL-style batch reward

Run from the repository root:

```bash
rewardaudit audit examples/trl/reward.py --dataset examples/trl/data.jsonl --n all
```

The function takes lists of prompts, plain-text completions and answers, and returns a list of scores.
It deliberately reads only the first option letter. Look for D6 and D7 in the result.

Excerpt (elapsed time varies):

```
rewardaudit 0.3.0 · reward:reward · 10 of 10 rows (seed 0) · 1,275 calls · 1 s

  HIGH    D6  Hedged answers accepted  O7 100%, O16 60%, E2 30%
              The answer is either A or B.
              Sent 5 more times: 1.0 each. Shortest reply still accepted (1.0): A or
```
