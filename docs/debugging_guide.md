# Debugging Your ML Experiments (PyTorch)

Companion guide to `pytorch_debug_drills.py`. Run the script first, scenario
by scenario, and try to name the bug from the printed symptom before reading
the fix. This doc is the "why," a framework you can follow whenever a run
misbehaves, and a cheat sheet to keep open while you debug your own
experiments.

## Why a systematic process matters

Most training failures look alike from the outside: the loss is NaN, or not
decreasing, or the validation numbers are strange. The fastest way to find the
cause is rarely to stare at the code and guess. It is to work through a fixed
sequence of cheap checks, each of which rules out a whole class of bugs.
Writing down what you checked and what you saw as you go (in your lab notebook
or experiment log) also makes it much easier to ask for help, because someone
else can see what has already been ruled out.

## The debugging framework

When a model won't train, work outward in this order. Each step is cheap
and rules out a large class of bugs before you go looking for something
subtle.

**1. Reproducibility first.** Fix every seed (`torch`, `numpy`, `random`,
CUDA) before you start changing anything. If you can't get the same failure
twice, you can't confirm you've fixed it.

**2. Overfit one tiny batch.** Take 8–16 examples, turn off regularization,
and try to drive the loss to ~0 in a few hundred steps. If a model with
enough capacity can't memorize 8 examples, the bug is upstream of
optimization — bad labels, wrong loss function, a frozen or disconnected
layer, data not reaching the model, and so on. This single check rules out
more bugs than anything else on this list, and it's the first thing an
experienced practitioner does. See `overfit_single_batch_sanity_check()` in the
script.

**3. Check shapes at the loss.** `assert pred.shape == target.shape` (or the
expected relationship) immediately before every loss call. Broadcasting bugs
don't crash — they silently compute the wrong thing and are the hardest
class of bug to catch by staring at code (Scenario 6).

**4. Print the first-iteration loss value and sanity-check its scale.**
For `k`-way classification with random-ish predictions,
`-log(1/k)` is the expected loss at step 0 (e.g. ~1.10 for 3 classes). Wildly
off from that means labels, loss function, or logits/targets are mismatched
before training even starts.

**5. Check train/eval mode switching.** `model.train()` before the training
loop, `model.eval()` before validation/inference, always paired with
`torch.no_grad()` at inference. Forgetting `eval()` corrupts BatchNorm
statistics and leaves Dropout active — validation metrics become noisy and
non-reproducible (Scenario 3).

**6. Check gradients, not just loss.** Right after `.backward()`, before
`.step()`:
```python
for name, p in model.named_parameters():
    if p.grad is None:
        print("no grad reaching:", name)   # frozen, or not in the graph
    elif torch.isnan(p.grad).any():
        print("NaN grad in:", name)
    else:
        print(name, p.grad.norm().item())
```
A `None` grad means that parameter never affected the loss (typo in the
forward pass, `requires_grad=False` left on by accident, or the parameter
just isn't wired into the computation graph). A grad norm that's zero
everywhere means dead ReLUs / a learning rate near-zero / a detached graph.
A grad norm that's enormous is your exploding-gradient early warning, before
it turns into a NaN loss.

**7. If it's NaN specifically**, see the walkthrough below — it's common
enough to deserve its own checklist.

## Reproducibility, in full

```python
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Two things people typically miss:

- **DataLoader workers.** With `num_workers > 0`, each worker process forks
  *after* you called `set_seed()` in the main process, so their own RNGs
  (used by e.g. random augmentations, random sampling) are not seeded by it.
  Pass `worker_init_fn` and a `generator` to the `DataLoader` (see
  `make_reproducible_loader()` in the script).
- **`cudnn.benchmark`.** It auto-tunes convolution algorithms for your input
  shapes for speed, but the chosen algorithm's numerics aren't guaranteed
  identical across runs. Off (`False`) for debugging/exact reproducibility,
  on (`True`) once you trust the code and want throughput on fixed-size
  inputs.
- Reproducibility is necessary but not sufficient — it makes a bug
  *reliably reproducible*, which is what actually lets you bisect it.

## The NaN-loss walkthrough (the most common failure)

1. **Confirm it, and find exactly when.** `if torch.isnan(loss): print(step); break`.
   "It goes NaN" and "it goes NaN at step 20, after climbing from 1.1 to
   1e24 by step 10" are very different amounts of signal — the second tells
   you it's a slow explosion, not an instant NaN.
2. **Bisect data vs. model.** Assert no NaNs in the input batch and labels
   before they hit the model. If the *data* is already NaN, the bug is in
   your preprocessing/normalization, not the model.
3. **Check gradients right after `backward()`**, before `step()` — see step
   6 of the framework above.
4. **`torch.autograd.set_detect_anomaly(True)`** at the top of the script.
   Autograd will raise at the exact forward op that produced the first
   NaN/Inf, with a stack trace pointing at the line. It's slow (don't leave
   it on for real training), but it turns "NaN somewhere in a 50-layer
   model" into "NaN at this exact `log()` call" in one run.
5. **Root causes, roughly by likelihood:**
   - Learning rate too high (Scenario 4) — the most common one by far.
   - No gradient clipping on an architecture prone to exploding gradients
     (RNNs/LSTMs, deep transformers without warmup). Fix:
     `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
     right after `backward()`, before `step()`.
   - `log(0)` or division by zero in a hand-rolled loss (Scenario 5). Fix:
     use the built-in `F.cross_entropy` / `F.binary_cross_entropy_with_logits`
     instead of reimplementing them — they use the log-sum-exp trick
     internally specifically to avoid this. If you must hand-roll,
     `torch.clamp(x, min=eps)` before any `log`.
   - Unnormalized inputs with large magnitude feeding into an unstable loss
     landscape.
   - Mixed precision (fp16) underflow/overflow. Use
     `torch.cuda.amp.GradScaler`, which exists precisely to keep small
     gradients from underflowing to zero and large ones from overflowing to
     inf in fp16.
   - A learning-rate scheduler warming up too fast, or one term in a
     multi-term loss dominating because its weight is too large.

## Quick-reference: symptom → likely cause

| Symptom | Most likely cause |
|---|---|
| Loss is `nan` after a few steps, climbing fast before that | LR too high / exploding gradients — add clipping, lower LR |
| Loss is `nan` or `inf` from step 0–1 | `log(0)`, division by zero, or bad/unnormalized input data |
| Loss decreases, then jumps up periodically | LR too high for some batches, or a data-loading bug feeding a corrupt batch |
| Loss barely moves at all | LR too low, a frozen layer, `zero_grad()` never called so grads accumulate destructively, or gradients not reaching the parameters (check step 6) |
| Loss looks fine, but accuracy is much worse than expected | Shape/broadcast bug in the loss (Scenario 6), or a train/val label mismatch |
| Val metrics noisy / worse than expected / not reproducible | Forgot `model.eval()` (Scenario 3) |
| Model trains but a re-run of the *same* code gives different results | Missing seed somewhere — often DataLoader workers, or `cudnn.benchmark=True` |
| Loss looks great locally but degrades right after resuming from a checkpoint | Optimizer state not saved/restored, or a LR scheduler restarting from step 0 |

## Running the drills

```bash
python pytorch_debug_drills.py             # run all 6 scenarios + bonus
python pytorch_debug_drills.py --scenario 4 # just the NaN-loss one
```

To build the habit: cover the `_fixed()` function for a scenario, read only
the `_buggy()` code and its printed symptom, and write down what you would
check first, second, third. Then reveal the fix and compare. When a run of
your own misbehaves, apply the same sequence to your real training script.
