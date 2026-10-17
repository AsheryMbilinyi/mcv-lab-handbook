"""
PyTorch Debugging Drills for ML Experiments
===========================================

Six runnable "spot the bug" scenarios. Each has:
  - a `_buggy()` version that reproduces a real symptom (NaN loss, stuck
    loss, silently wrong loss, etc.)
  - a `_fixed()` version with the fix and a comment explaining *why*.

Run all of them:
    python pytorch_debug_drills.py

Run one scenario (e.g. #4, the NaN-loss one):
    python pytorch_debug_drills.py --scenario 4

Read top to bottom WITHOUT looking at the fixed version first — try to
name the bug from the symptom alone. That builds the pattern recognition
you'll use on your own experiments.
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. Reproducibility — do this before every scenario, and in every real
#    training script. If you can't reproduce a bug, you can't fix it, and
#    if a collaborator or reviewer re-runs your code and gets different
#    numbers, that's its own red flag.
# ---------------------------------------------------------------------------
def set_seed(seed: int = 0):
    """Seed every random number generator used in a run.

    Args:
        seed: Seed shared by Python, NumPy and PyTorch (CPU and CUDA).

    Side effects:
        Makes cuDNN deterministic and disables its autotuner, which can
        slow down training on fixed-size inputs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU, safe to call always
    # Determinism vs speed tradeoff. cudnn's autotuner picks the fastest
    # algorithm for your input shapes, but that algorithm can be
    # non-deterministic across runs. Turn this off for debugging, back on
    # (benchmark=True) once you trust the code and want speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed the RNGs of one DataLoader worker process.

    Pass this as ``worker_init_fn`` so that augmentations that use
    NumPy or ``random`` are reproducible when ``num_workers > 0``.

    Args:
        worker_id: Index of the worker, supplied by the DataLoader.
    """
    # DataLoader workers are forked *after* you seed the main process, so
    # each worker's own RNG (used by e.g. random augmentations) is NOT
    # seeded by set_seed() above. This is the #1 cause of "I set the seed
    # but my results still aren't reproducible" when num_workers > 0.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_reproducible_loader(dataset, batch_size, seed=0, **kwargs):
    """Build a DataLoader with seeded shuffling and workers.

    Args:
        dataset: Any map-style PyTorch dataset.
        batch_size: Number of examples per batch.
        seed: Seed for the shuffling generator.
        **kwargs: Extra arguments passed to ``DataLoader``.

    Returns:
        A ``torch.utils.data.DataLoader`` that yields the same batches in
        the same order on every run.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        **kwargs,
    )


def _toy_classification_data(n=256, d=20, classes=3, seed=0):
    """Create a small, learnable classification problem.

    Args:
        n: Number of examples.
        d: Number of input features.
        classes: Number of classes.
        seed: Seed for the data generator.

    Returns:
        A tuple ``(X, y)`` of features and integer class labels.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    true_w = torch.randn(d, classes, generator=g)
    y = (X @ true_w).argmax(dim=1)
    return X, y


def _banner(title):
    """Print a title between two rows of ``=`` characters."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Scenario 1 — missing optimizer.zero_grad()
#   Symptom: the loss drops for the first few steps, then turns erratic and
#   spikes, while the correct run falls smoothly. Gradients from every step
#   keep accumulating on top of each other (PyTorch grads are additive by
#   design, to support gradient accumulation over multiple micro-batches),
#   so each update is no longer the gradient of the current batch.
#   Beware: on a tiny convex problem the pile-up just acts like a larger
#   step size and can even look *better*, which is why this bug is easy to
#   miss. That is why this demo uses a small MLP, mini-batches and a
#   learning rate that is fine when gradients are cleared.
# ---------------------------------------------------------------------------
def _scenario_1_run(zero_grad: bool):
    """Train a small MLP on mini-batches, with or without ``zero_grad()``.

    Both variants draw the same mini-batches, so the only difference is
    whether stale gradients are cleared.

    Args:
        zero_grad: Whether to call ``opt.zero_grad()`` before every step.

    Returns:
        The training loss at every step.
    """
    set_seed(0)
    X, y = _toy_classification_data(n=256)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.3)
    batches = torch.Generator().manual_seed(1)
    losses = []
    for step in range(50):
        idx = torch.randint(0, len(y), (32,), generator=batches)
        if zero_grad:
            opt.zero_grad()  # clear stale gradients from the previous step
        # else: opt.zero_grad() is MISSING here
        loss = F.cross_entropy(model(X[idx]), y[idx])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def scenario_1_buggy():
    """Train without ``zero_grad()`` so gradients pile up.

    Returns:
        The training loss at every step.
    """
    losses = _scenario_1_run(zero_grad=False)
    shown = losses[::10] + [losses[-1]]
    print("buggy   (no zero_grad):", [round(v, 2) for v in shown])
    return losses


def scenario_1_fixed():
    """Train with ``zero_grad()`` called every step.

    Returns:
        The training loss at every step.
    """
    losses = _scenario_1_run(zero_grad=True)
    shown = losses[::10] + [losses[-1]]
    print("fixed   (with zero_grad):", [round(v, 2) for v in shown])
    return losses


# ---------------------------------------------------------------------------
# Scenario 2 — softmax fed into CrossEntropyLoss ("double softmax")
#   Symptom: loss doesn't reach near-zero even when the model can trivially
#   fit the data (e.g. overfitting one batch, see the sanity check below).
#   nn.CrossEntropyLoss already applies log_softmax internally — it expects
#   *raw logits*. Feeding it post-softmax probabilities squashes the
#   gradient signal and biases the loss landscape.
# ---------------------------------------------------------------------------
def scenario_2_buggy():
    """Feed softmax probabilities into ``cross_entropy``.

    Returns:
        The final training loss, which stays well above zero.
    """
    set_seed(0)
    X, y = _toy_classification_data(n=64)
    model = nn.Linear(20, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for step in range(300):
        opt.zero_grad()
        probs = F.softmax(model(X), dim=1)  # BUG: softmax applied here...
        loss = F.cross_entropy(probs, y)  # ...and CE applies it again
        loss.backward()
        opt.step()
    print(
        f"buggy   (double softmax) final loss on 64 samples: {loss.item():.4f}"
    )
    return loss.item()


def scenario_2_fixed():
    """Feed raw logits into ``cross_entropy``.

    Returns:
        The final training loss, which gets close to zero.
    """
    set_seed(0)
    X, y = _toy_classification_data(n=64)
    model = nn.Linear(20, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for step in range(300):
        opt.zero_grad()
        logits = model(X)  # pass raw logits
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    print(
        f"fixed   (raw logits) final loss on 64 samples:     "
        f" {loss.item():.4f}"
    )
    return loss.item()


# ---------------------------------------------------------------------------
# Scenario 3 — forgetting model.eval() at validation time
#   Symptom: validation metrics are noisy / worse than training metrics in
#   a way that's too large to be normal generalization gap, and they
#   *change between identical validation runs*. Dropout is still randomly
#   zeroing activations and BatchNorm is still using per-batch statistics
#   instead of its running averages.
# ---------------------------------------------------------------------------
def scenario_3_buggy():
    """Evaluate a model that is still in ``train()`` mode.

    Returns:
        The largest difference between two identical forward passes.
    """
    set_seed(0)
    model = nn.Sequential(
        nn.Linear(20, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 3),
    )
    X, y = _toy_classification_data(n=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

    # BUG: no model.eval() before "validation" -> dropout/batchnorm still
    # behave as in training. Same input, different output each call.
    with torch.no_grad():
        out1 = model(X)
        out2 = model(X)
    diff = (out1 - out2).abs().max().item()
    print(
        f"buggy   (still in train() mode) max diff between two identical "
        f"eval passes: {diff:.4f}  (should be ~0)"
    )
    return diff


def scenario_3_fixed():
    """Evaluate a model after calling ``eval()``.

    Returns:
        The largest difference between two identical forward passes.
    """
    set_seed(0)
    model = nn.Sequential(
        nn.Linear(20, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 3),
    )
    X, y = _toy_classification_data(n=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

    model.eval()  # disable dropout, freeze BatchNorm to running stats
    with torch.no_grad():
        out1 = model(X)
        out2 = model(X)
    diff = (out1 - out2).abs().max().item()
    print(
        f"fixed   (eval() mode) max diff between two identical eval "
        f"passes:      {diff:.6f}  (now ~0)"
    )
    return diff


# ---------------------------------------------------------------------------
# Scenario 4 — the classic NaN loss: learning rate too high + no clipping
#   Symptom: loss is finite for a handful of steps, then becomes `nan` and
#   stays there forever, because once a weight becomes NaN, every
#   downstream computation involving it is also NaN (NaN is "poisonous").
#   This is one of the most common failures you'll hit in real experiments.
# ---------------------------------------------------------------------------
def scenario_4_buggy():
    """Train with a learning rate so high that the loss diverges.

    Returns:
        The first step at which the loss was NaN or infinite, or ``None``
        if it never was.
    """
    set_seed(0)
    X, y = _toy_classification_data(n=128)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.SGD(model.parameters(), lr=50.0)  # BUG: way too high
    first_nan_step = None
    for step in range(30):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        if (torch.isnan(loss) or torch.isinf(loss)) and first_nan_step is None:
            first_nan_step = step
        if step in (0, 1, 2, 5, 10, 15, 20, 29):
            print(f"  step {step:2d}  loss={loss.item()}")
    print(f"buggy   (lr=50.0) first NaN/Inf at step: {first_nan_step}")
    return first_nan_step


def scenario_4_fixed():
    """Train with a sane learning rate and gradient clipping."""
    set_seed(0)
    X, y = _toy_classification_data(n=128)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)  # sane LR
    for step in range(30):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        # Belt-and-suspenders even at a sane LR: cap the gradient norm so a
        # rare bad batch can't blow the weights up.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if step in (0, 1, 2, 5, 10, 29):
            print(f"  step {step:2d}  loss={loss.item():.4f}")
    print(
        f"fixed   (lr=0.05 + grad clipping) "
        f"final loss={loss.item():.4f}, no NaN"
    )
    return None


# How you'd actually *locate* this in your own code, beyond just eyeballing
# the loss curve. Work through these steps in order:
def scenario_4_diagnostic_walkthrough():
    """Print the steps for locating a NaN loss."""
    _banner("Scenario 4 diagnostic steps (use these on your own NaN losses)")
    print("""\
1. Confirm it's really NaN and find the first occurrence:
       if torch.isnan(loss): print(step); break
2. Bisect: is it the DATA or the MODEL/OPTIMIZATION that's NaN first?
       assert not torch.isnan(x).any(), "NaN in input batch"
       assert not torch.isnan(y).any(), "NaN in labels"
3. If inputs are clean, check gradients right after backward(), before step():
       for n, p in model.named_parameters():
           if p.grad is not None and torch.isnan(p.grad).any():
               print("NaN grad in", n)
4. torch.autograd.set_detect_anomaly(True) at the top of the script makes
   autograd raise immediately at the *op* that produced the first NaN/Inf,
   with a stack trace pointing at the forward call that created it. Slow —
   only turn it on while debugging, never in production training.
5. Common root causes, roughly in order of likelihood:
     - learning rate too high (this scenario)
     - no gradient clipping on a model prone to exploding grads (RNNs, deep
       transformers without warmup)
     - log(0) / division by zero in a custom loss (see Scenario 5)
     - unnormalized inputs with large magnitude
     - mixed precision (fp16) underflow/overflow -> use
       torch.amp.GradScaler, which exists specifically to prevent this
     - a learning rate scheduler warming up too fast, or a loss weight
       that's too large in a multi-term loss
""")


# ---------------------------------------------------------------------------
# Scenario 5 — log(0) in a hand-rolled loss function
#   Symptom: NaN appears almost immediately (often step 0-1), independent
#   of learning rate, and often coincides with the model predicting a
#   probability of exactly 0.0 or 1.0 for some example.
# ---------------------------------------------------------------------------
def scenario_5_buggy():
    """Compute a hand-written loss that takes ``log(0)``.

    Returns:
        The loss value, which is infinite.
    """
    set_seed(0)
    logits = torch.tensor(
        [[10.0, -10.0]], requires_grad=True
    )  # very confident
    target = torch.tensor([0])

    probs = F.softmax(logits, dim=1)
    p_correct = probs[range(1), target]
    # BUG: hand-rolled negative log likelihood, no numerical safeguard.
    # softmax(10, -10) rounds to exactly 1.0 in fp32 -> log(1.0 - eps) is
    # fine, but log(0.0) (the *wrong*-class prob for a confident wrong
    # prediction) is -inf, and -inf * grad → NaN during backward.
    loss = -torch.log(1 - p_correct + 0 * p_correct)  # forces the log(~0) path
    print(f"buggy   (log(0) risk) loss={loss.item()}")
    loss.backward()
    print(f"        grad on logits: {logits.grad}")
    return loss.item()


def scenario_5_fixed():
    """Clamp the argument of ``log`` to avoid ``log(0)``.

    Returns:
        The loss value, which is finite.
    """
    set_seed(0)
    logits = torch.tensor([[10.0, -10.0]], requires_grad=True)
    target = torch.tensor([0])

    probs = F.softmax(logits, dim=1)
    p_correct = probs[range(1), target]
    eps = 1e-8
    loss = -torch.log(torch.clamp(1 - p_correct, min=eps))
    print(f"fixed   (clamped) loss={loss.item():.4f}")
    loss.backward()
    print(f"        grad on logits: {logits.grad}")
    print(
        "        (in practice: just use the built-in F.cross_entropy / "
        "F.binary_cross_entropy_with_logits — they use the log-sum-exp "
        "trick internally and never hit this.)"
    )
    return loss.item()


# ---------------------------------------------------------------------------
# Scenario 6 — silent shape broadcast bug
#   Symptom: no crash, no NaN, model trains and loss even goes down — but
#   final accuracy is much worse than it should be. These are the worst
#   bugs precisely because nothing errors. (F.mse_loss does print a
#   UserWarning for this case, but it is easy to miss in training logs,
#   and a hand-written (pred - target) ** 2 prints nothing at all.)
#   MSE between predictions shaped (N, 1) and targets shaped (N,) broadcasts
#   to an (N, N) matrix instead of elementwise (N,) — every prediction gets
#   compared against every target.
# ---------------------------------------------------------------------------


def scenario_6_buggy():
    """Compute MSE between tensors of shape (N, 1) and (N,).

    Returns:
        The loss, silently averaged over an N by N matrix.
    """
    set_seed(0)
    N = 8
    pred = torch.randn(N, 1, requires_grad=True)
    target = torch.randn(N)  # BUG: shape (N,), not (N, 1)
    loss = F.mse_loss(pred, target)
    print(
        f"buggy   pred.shape={tuple(pred.shape)} "
        f"target.shape={tuple(target.shape)}"
        f" -> broadcast shape used internally: {(pred - target).shape}"
    )
    print(
        f"        loss={loss.item():.4f} (silently averaging over an "
        f"{N}x{N} matrix instead of {N} elementwise diffs)"
    )
    return loss.item()


def scenario_6_fixed():
    """Compute MSE after reshaping the target to (N, 1).

    Returns:
        The loss, averaged over N elementwise differences.
    """
    set_seed(0)
    N = 8
    pred = torch.randn(N, 1, requires_grad=True)
    target = torch.randn(N).unsqueeze(1)  # reshape to (N, 1) to match
    assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
    loss = F.mse_loss(pred, target)
    print(
        f"fixed   pred.shape={tuple(pred.shape)} "
        f"target.shape={tuple(target.shape)}"
        f" -> elementwise, loss={loss.item():.4f}"
    )
    print(
        "        Rule of thumb: assert shapes right before every loss call. "
        "PyTorch often will not warn you about a broadcast that "
        "'succeeds' but means something different than what you "
        "intended."
    )
    return loss.item()


# ---------------------------------------------------------------------------
# Bonus — the single highest-value sanity check in the whole drill:
# overfit one tiny batch before you trust anything about your full pipeline.
# If a model with enough capacity can't drive the loss on 8 examples to
# ~0 within a few hundred steps, something upstream is broken (bad labels,
# wrong loss, frozen layers, data not actually reaching the model, etc.) —
# and it's far cheaper to find that out on 8 examples than after a 3-hour
# run on the full dataset.
# ---------------------------------------------------------------------------


def overfit_single_batch_sanity_check():
    """Check that a model can memorize eight examples.

    Returns:
        A tuple ``(loss, accuracy)`` after training. A healthy pipeline
        reaches a loss near zero and an accuracy of 1.0.
    """
    set_seed(0)
    X, y = _toy_classification_data(n=8, classes=3)
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    acc = (model(X).argmax(1) == y).float().mean().item()
    print(
        f"overfit-one-batch check: final loss={loss.item():.5f}, "
        f"accuracy={acc:.2f}  (expect loss -> ~0, accuracy -> 1.0)"
    )
    return loss.item(), acc


SCENARIOS = {
    1: ("missing zero_grad()", scenario_1_buggy, scenario_1_fixed),
    2: (
        "softmax fed into CrossEntropyLoss",
        scenario_2_buggy,
        scenario_2_fixed,
    ),
    3: ("forgot model.eval() at val time", scenario_3_buggy, scenario_3_fixed),
    4: (
        "NaN loss: LR too high, no clipping",
        scenario_4_buggy,
        scenario_4_fixed,
    ),
    5: ("NaN loss: log(0) in custom loss", scenario_5_buggy, scenario_5_fixed),
    6: ("silent shape-broadcast bug", scenario_6_buggy, scenario_6_fixed),
}


def main():
    """Run every scenario, or only the one chosen with ``--scenario``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=int,
        choices=SCENARIOS.keys(),
        help="run just one scenario (default: run all)",
    )
    args = parser.parse_args()

    to_run = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    for i in to_run:
        title, buggy_fn, fixed_fn = SCENARIOS[i]
        _banner(f"Scenario {i}: {title}")
        buggy_fn()
        print("-" * 78)
        fixed_fn()
        if i == 4:
            scenario_4_diagnostic_walkthrough()

    if not args.scenario:
        _banner("Bonus: overfit-a-single-batch sanity check")
        overfit_single_batch_sanity_check()


if __name__ == "__main__":
    main()
