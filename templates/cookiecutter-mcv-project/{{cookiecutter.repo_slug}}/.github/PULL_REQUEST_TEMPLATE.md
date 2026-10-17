# MCV Lab Pull Request Checklist (Experimentation Standard)

## Summary
**What does this PR add/change?**
- [ ] New dataset pipeline
- [ ] New model / architecture
- [ ] Training loop updates
- [ ] Evaluation updates
- [ ] Experiment config updates (Argparse/Hydra)
- [ ] SLURM/HPC scripts
- [ ] Documentation only

**Short description (2–5 sentences):**
> ...

---

## Reproducibility & Experiment Tracking (Required)
- [ ] Training/eval runs end-to-end **without editing source code**
- [ ] All experiment settings are controlled via:
  - [ ] CLI args (argparse) OR
  - [ ] Hydra config groups + overrides
- [ ] Every run saves:
  - [ ] resolved config (`config.json` or `config_resolved.yaml`)
  - [ ] checkpoints (`last` and optionally `best`)
  - [ ] `cmd.txt` (recommended)
- [ ] Seed is supported and set (`seed`)
- [ ] Deterministic flag exists (`--deterministic` or config field)

---

## Config & Interface Quality
- [ ] Names are consistent across scripts and configs
- [ ] Defaults allow a baseline run immediately
- [ ] Important args have constrained choices (model/dataset/optimizer)
- [ ] Dataset paths are NOT hardcoded in code
- [ ] New configs are documented briefly in README

---

## Logging & Metrics
- [ ] Metrics are logged periodically
- [ ] Run naming is meaningful and sweep-friendly
- [ ] Optional: logs to W&B / MLflow / TensorBoard
- [ ] Metrics saved in machine-readable format (recommended: JSONL)

---

## HPC / SLURM (if applicable)
- [ ] SLURM logs go to `slurm_logs/`
- [ ] Run outputs go to `runs/`
- [ ] SLURM script requests GPU/CPU/RAM/time explicitly
- [ ] Supports sweeps:
  - [ ] SLURM arrays OR
  - [ ] Hydra multirun (small sweeps)

---

## Testing
- [ ] Smoke test verified:
  - [ ] `trainer.epochs=1` works
  - [ ] saves resolved config
  - [ ] writes checkpoint(s)

---

## Notes to Reviewer
**What should the reviewer pay attention to?**
> ...

**How to run**
```bash
python train.py
python train.py model=resnet50 optim.lr=5e-5 seed=2
```
