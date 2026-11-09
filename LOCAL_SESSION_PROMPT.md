# Prompt for the local session: run the ablations and validate the results

Copy everything below the line into the local Claude Code session (repo `Learning2Rearrange`, branch `claude/affectionate-babbage-ainipv`).

---

You are continuing work on this repository (UR5e rearrangement in PyBullet, double-DQN with dense pixel-wise Q-maps). Everything is
implemented; your job is to **run the experiments, validate that the results are trustworthy, and report them**. Do not change the
algorithms unless a validation step below fails and you can show why. Work on the current branch, commit small, push to the same branch,
do not open a PR. Read `README.md`, `ABLATIONS.md` and `SIMULATION_NOTES.md` first (10 minutes); every flag is documented there.

## 0. Environment and sanity (must pass before any training)
1. `pip install -r requirements.txt`; check `python -c "import torch;print(torch.cuda.is_available())"`. Use the GPU if there is one.
2. `python -m pytest tests -q` -> everything passes (one test is skipped without `L2R_GOLDEN=1`).
3. `L2R_GOLDEN=1 python -m pytest tests/test_sim_regression.py -q` -> the golden scenarios of the real simulator match
   (`tests/golden/{legacy,modern}.json`). If PyBullet differs from the version that recorded them, re-record **only** after confirming the
   differences are tiny (see `tests/sim_scenario.py`) and say so in the commit.
4. `python profile_sim.py` -> note the simulator steps/s with 1 and `--num_envs 4`; use it to size the runs (below).
5. `python scripts/run_experiments.py run --suite smoke --out runs/smoke` -> whole chain (train, evaluate, summary) works.

## 1. Perception networks (train before the perception ablations)
Type network (existing) and the new **object-agnostic** network (background / interior / touching-boundary; no type catalogue):
```bash
python train_perception.py collect --scenes 600 --out data/perception
python train_perception.py train   --data data/perception --minutes 20 --out assets/perception/segnet.pth
python train_perception.py evaluate --weights assets/perception/segnet.pth --scenes 40

python train_perception.py train    --task agnostic --data data/perception --minutes 20 --out assets/perception/agnostic.pth
python train_perception.py evaluate --task agnostic --weights assets/perception/agnostic.pth --scenes 40
```
**Generalisation test of the agnostic network** (the point of it): the catalogue has 4 YCB types, an episode draws 3. Train without one type,
test with it:
```bash
python train_perception.py collect --scenes 600 --out data/perception_holdout --object_types PottedMeatCan CrackerBox ChipsCan   # MustardBottle held out
python train_perception.py train    --task agnostic --data data/perception_holdout --minutes 20 --out /tmp/agnostic_holdout.pth
python train_perception.py evaluate --task agnostic --weights /tmp/agnostic_holdout.pth --scenes 40 --object_types MustardBottle CrackerBox ChipsCan
```
Repeat holding out each type in turn. Report interior/boundary IoU, instance recall/precision, class agreement and free-space IoU of the
observation. **Pass criteria:** top-view instance recall >= 0.9 and precision >= 0.9 on seen types; on a held-out type the agnostic network
should lose less than 10 points of recall, whereas the type network (`--perception network`) cannot handle it at all (show this
once). If the agnostic net fails the criteria, look at `match_instances` failures (merged touching objects? thin boundary after the 2x
downsampling?) and fix the cause; consider a wider boundary target before changing the architecture.

## 2. Ablations (all through `scripts/run_experiments.py`)
`python scripts/run_experiments.py list` shows every suite. Size the training from step 0.4: the default `--steps 20000` is about the
smallest budget where differences are visible; use 3 seeds (`--seeds 0 1 2`) for everything that is reported.
```bash
python scripts/run_experiments.py run --suite core         --seeds 0 1 2 --steps 30000 --out runs/core
python scripts/run_experiments.py run --suite belief       --seeds 0 1 2 --steps 30000 --out runs/belief
python scripts/run_experiments.py run --suite perception   --seeds 0 1 2 --steps 30000 --out runs/perception
python scripts/run_experiments.py run --suite end_effector --seeds 0 1 2 --steps 30000 --out runs/end_effector
python scripts/run_experiments.py run --suite physics      --seeds 0 1 2 --steps 30000 --out runs/physics
python scripts/run_experiments.py summarize --out runs/core      # table, mean +- std over seeds
```
Order matters: `core` first (the reward is **shaped (`progress`) by default now**; `sparse_reward` is the paper's reward and the reference
for the reward claim). Add `reward_noop` and a longer run of the best two configurations if the budget allows.

## 3. Validation of the results (do not skip; this is the actual task)
For every claim in `ABLATIONS.md` ("X should help"), state **supported / not supported / inconclusive** with the numbers:
* **Seeds and significance.** 3 seeds, mean +- std, per object count. A difference smaller than 1 std of the baseline is *inconclusive*, not an
  improvement. Look at `logs/<run>/episodes.csv` learning curves, not only the final evaluation: a variant that learns faster but ends equal is a result.
* **Reward.** Shaping must not change the evaluation metric's ranking pathologically: evaluate with success rate / mean actions (not the
  training return, which has a different scale per reward mode). If `progress` is worse than `sparse`, check the Q-range clipping
  (`Config.q_range`) and `analyze_rewards.py` before concluding anything.
* **Hierarchy.** Verify the continuous push length is actually used: histogram the executed lengths (`info['actions']`, `pix[3]`) - if they
  collapse to one value the actor is not learning (compare with the fixed multi-length `push_lengths` run).
* **Gripper.** Report grasp success of the *learned* policy next to the oracle bound (12/12 on reachable front objects, see `ABLATIONS.md`);
  inspect failures with `demo_sim.py --end_effector gripper` (GUI): walls at |y| > 17 cm, objects wider than the 20 cm opening, tipping.
* **Perception.** The policy trained on ground truth, evaluated with `--perception network|agnostic`, shows the sim-to-perception gap; the
  policy trained *with* the network perception shows what is recoverable. Both rows are needed.
* **Physics.** `modern` vs `legacy` compare success *and* the rate of unexecuted actions (`force_stops`, failed grasps): the former physics
  may simply be more forgiving.
* **Determinism.** Re-run one finished experiment with the same seed: the evaluation must agree within the statistical noise of the
  episode count (if not, find the nondeterminism: `--sim_quality modern` has `deterministic` contacts).
* **Suspicious results first.** An ablation that is much better than the baseline is more likely a bug (leaky observation, privileged
  ids reaching the policy, relabelling error) than a discovery: check that `gt_idmap` / `instance_to_object` are never read by the
  network input (`Trainer.prepare`) and that the evaluation uses fresh seeds (`1000 + seed`).

## 4. Deliverables
1. `runs/*/summary.md` per suite, plus a consolidated results section appended to `ABLATIONS.md` (table + 3-6 lines per claim: supported /
   not / inconclusive and why). Keep raw logs out of git except the summaries (`runs/` is data; commit only `summary.md/csv`).
2. The trained perception weights (`assets/perception/segnet.pth`, `agnostic.pth`) if their criteria pass.
3. A short list of bugs or surprises found, each with the fix commit or an issue text.
4. Commit and push to `claude/affectionate-babbage-ainipv`. Commit messages end with the usual `Co-Authored-By` / `Claude-Session` trailer lines.
