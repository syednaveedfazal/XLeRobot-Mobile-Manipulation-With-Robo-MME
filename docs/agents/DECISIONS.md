# DECISIONS (experiment-planner adds a decision only after the user agrees)

Format: date | decision | reason

- 2026-11-20 | Task: PutItBack with K = 3 identical stations, sandbox scene, scripted-expert demos | user's choice.
- 2026-11-21 | Stations and work table 70 cm high; work table is a four-legged table | matches the user's lab.
- 2026-11-21 | Object: small box-like object instead of the cup | user's choice; became a 2x2x4 cm block because the SO-101 jaws only hold ~2 cm-wide parts in sim.
- 2026-11-23 | Train both models 12k steps, batch 16, identical settings | 5 s/step on one A40; 30k steps would not fit the 24 h job limit.
- 2026-11-23 | Pin `lerobot/pi05_base` to revision a538eb27 | later revisions name a processor step our LeRobot 0.4.2 lacks; weights identical.
- 2026-11-24 | Nothing is uploaded to Hugging Face; read-only token | user's rule.
- _Pending user confirmation:_ object -> 6 cm box with the 2 cm handle; look-first expert; 360 demos with recovery noise.

- **2026-11-26** | Re-record (D3) instead of training on D2 | user: "re record now and do the timing fix" | D2's turn-back
  time predicts the station 65 % (B12); a memory result on D2 would be confounded with time. Cost: ~1 day.
- **2026-11-26** | Table wait until a random clock time (60-75 s), not a random length | removes the shortcut by
  construction instead of drowning it in noise; costs ~25 s more of idle frames per episode.
- **2026-11-26** | 9 episodes per chunk x 20 (not 12 x 15) | episodes are ~45 % longer; recordings must fit the 1 h queue.

