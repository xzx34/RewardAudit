# Several accepted QA answers

Run from the repository root:

```bash
rewardaudit audit examples/qa/reward.py --dataset examples/qa/data.jsonl --n all --answer-col answers.text.0 --reference-col answers.text --answer-field
```

The checker receives the whole list of names, and each reply is written from one name.
It deliberately checks only the first name. Look for G2 under D16: the other accepted names lose the score.

Excerpt (elapsed time varies):

```
rewardaudit 0.3.0 · reward:score · 10 of 10 rows (seed 0) · 1,092 calls · 1 s

  MEDIUM  D16  Reference answer rejected  G2 100%
               Victor-Marie Hugo
```
