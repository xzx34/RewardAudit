# A verl-style numeric reward

Run from the repository root:

```bash
rewardaudit audit examples/verl/reward.py --dataset examples/verl/data.jsonl --n all
```

The ten rows use `reward_model.ground_truth`. The function returns a dict with a total and a correctness part.
It deliberately compares absolute values, so a wrong sign earns full score. Look for D1 in the result.

Excerpt (elapsed time varies):

```
rewardaudit 0.3.0 · reward:compute_score · 10 of 10 rows (seed 0) · 1,342 calls · 1 s

  HIGH  D1  Signs and punctuation ignored  M2 100%
            -17
            Sent 5 more times: 1.0 each.
```
