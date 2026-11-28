# RESULTS (eval-analyst; final tables for the paper)

Every number with its episode count and a 95 % Wilson interval. Never a number that is not here.

## PutItBack, simulation

| Model | Checkpoint | Episodes | Success | 95 % CI | Notes |
|---|---|---|---|---|---|
| pi0.5, no memory (T1-none) | 12k | 4 (+3 with 10-step plans) | 0 | 0-49 % (n=4) | never grasps; pilot only |
| pi0.5, memory (T1-mem) | 12k | 6 | 0 | 0-39 % (n=6) | never grasps; pilot only |

The memory comparison is not possible yet: both fail before the memory part of the task.
