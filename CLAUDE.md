# CLAUDE.md

XLeRobot in Isaac Sim / Isaac Lab / LeIsaac on the Marvin cluster, memory tasks (PutItBack) and
pi0.5 with and without memory. See README.md for running things and docs/agents/STATUS.md for
where the project stands.

## Ground rules
1. Do not commit or push. No `git commit`, `git push`, `git reset --hard`, `git clean`, or branch
   deletion. Leave all changes uncommitted; the user reviews and commits.
2. Nothing leaves Marvin: no uploads to Hugging Face, W&B public projects or anywhere else.
3. Delete only what is really not needed any more (the user allows it), e.g. raw HDF5 recordings
   after a successful conversion, or space-hungry scratch files; say what was deleted. Datasets,
   checkpoints, results and logs that matter go to an `_archive/` folder instead.
4. GPU only through Slurm; never run Isaac Sim or training on the login node. Ask the user before
   any job over 2 GPU-hours.
5. One change per experiment; the log says what that one change is.
6. Real robot: no command that moves motors without the user's explicit "yes" in the chat.
7. Plain language in all notes and reports.

## Agents
Project subagents live in .claude/agents/. Shared notes live in docs/agents/.
- Start a session: experiment-planner reads STATUS.md and proposes the next steps.
- Literature or citations: research-scout.
- Anything wrong in simulation: sim-debugger.
- New dataset: data-auditor before training-ops.
- Training: training-ops. Evaluation and numbers: eval-analyst.
- Real robot and calibration: sim2real-engineer (no motor commands without the user's yes).
- Paper: paper-editor.
Ground rules: no commits or pushes, nothing uploaded, delete only what is really not needed, GPU only via Slurm,
ask before jobs over 2 GPU-hours, one change per experiment.
