# A verl-style numeric reward

Run from the repository root:

```bash
rewardaudit audit examples/verl/reward.py --dataset examples/verl/data.jsonl --n all
```

The ten rows use `reward_model.ground_truth`. The function returns a dict with a total and a correctness part.
It deliberately compares absolute values, so a wrong sign earns full score. Look for D1 in the result.
