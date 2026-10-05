"""A minimal PyTorch training loop, three optimisers, and the bugs that bite RL code.

Chapter 00, Section 7.
  (a) Fit an MLP to noisy 1-D data with the canonical loop
         zero_grad -> forward -> loss -> backward -> step        (Algorithm 7.1)
      using SGD, SGD+momentum and Adam.
  (b0) autograd reproduces the hand-computed derivative of Section 3.1.
  (b) Live demonstrations of common bugs (the rows of the table in Section 7.7):
        1. the (N,1) vs (N,) broadcasting bug in the loss (and what it does to training),
        2. forgetting zero_grad (gradients accumulate),
        3. detach(): semi-gradient vs full gradient through a bootstrapped target,
        4. torch.no_grad() for evaluation / acting (shown via requires_grad / grad_fn),
        5. float64 numpy arrays meeting a float32 network,
        6. gather() to pick Q(s, a) for the taken actions,
        7. arithmetic on a boolean `terminated` tensor,
        8. in-place modification of a tensor that backward() needs.

Run:  python code/ch00_math_toolkit/pytorch_regression.py [--quick]
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def make_data(rng, n):
    x = rng.uniform(-3, 3, size=n)
    y = np.sin(2 * x) + 0.3 * x + 0.1 * rng.normal(size=n)
    # float32 + explicit column shape (N, 1): the two conventions that avoid bugs 5 and 1
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def make_mlp(hidden=64):
    return nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


def train(model, opt, x, y, epochs, batch_size, gen, target_shape_bug=False):
    """Algorithm 7.1. Returns the per-epoch mean training loss."""
    loss_fn = nn.MSELoss()
    n = x.shape[0]
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)          # reshuffle every epoch
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x[idx], y[idx]
            if target_shape_bug:
                yb = yb.squeeze(1)                        # (B,) instead of (B, 1): BUG
            pred = model(xb)                              # forward: shape (B, 1)
            loss = loss_fn(pred, yb)                      # scalar
            opt.zero_grad()                               # clear old gradients (they accumulate!)
            loss.backward()                               # reverse-mode autodiff fills p.grad
            opt.step()                                    # p <- p - lr * (something built from p.grad)
            total += loss.item() * len(idx)
        history.append(total / n)
    return np.array(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    epochs = 40 if args.quick else 400
    batch_size = 32
    print(f"[pytorch_regression] seed={args.seed} epochs={epochs} batch={batch_size} quick={args.quick}")
    t0 = time.time()

    x_train, y_train = make_data(rng, 256)
    x_test = torch.linspace(-3, 3, 1000).unsqueeze(1)
    y_test_clean = torch.sin(2 * x_test) + 0.3 * x_test

    # ------------------------------------------------------------------ (a)
    configs = {
        "SGD lr=0.05": lambda p: torch.optim.SGD(p, lr=0.05),
        "SGD+momentum lr=0.05, β=0.9": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        "Adam lr=1e-3": lambda p: torch.optim.Adam(p, lr=1e-3),
    }
    results = {}
    print("\n(a) MLP 1-64-64-1 (tanh), 256 training points")
    for name, make_opt in configs.items():
        torch.manual_seed(args.seed)                   # identical initial weights for every optimiser
        model = make_mlp()
        gen = torch.Generator().manual_seed(args.seed)  # identical minibatch order too
        hist = train(model, make_opt(model.parameters()), x_train, y_train, epochs, batch_size, gen)
        with torch.no_grad():                           # evaluation: no graph needed
            test_mse = ((model(x_test) - y_test_clean) ** 2).mean().item()
            pred = model(x_test).squeeze(1).numpy()
        results[name] = dict(hist=hist, test=test_mse, pred=pred)
        print(f"  {name:30s} final train MSE {hist[-1]:.4f}   test MSE vs clean function {test_mse:.4f}")
    print(f"  (noise variance is 0.01, so a train MSE near 0.01 means the noise floor is reached)")

    # ------------------------------------------------------------------ (b0) autograd check
    print("\n(b0) Autograd vs the chain rule by hand (Section 3.1): L(w) = 0.5 (tanh(2w) - 1)^2 at w = 0.5")
    w_ex = torch.tensor(0.5, requires_grad=True)
    loss_ex = 0.5 * (torch.tanh(w_ex * 2.0) - 1.0) ** 2
    loss_ex.backward()
    t1 = np.tanh(1.0)
    print(f"  autograd dL/dw = {w_ex.grad.item():.6f};  by hand (tanh(1)-1)(1-tanh(1)^2)*2 = "
          f"{(t1 - 1) * (1 - t1 ** 2) * 2:.6f}")

    # ------------------------------------------------------------------ (b1) shape bug
    print("\n(b1) Broadcasting bug: prediction (B,1) vs target (B,)")
    pred = torch.zeros(4, 1); target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bad = nn.functional.mse_loss(pred + target.unsqueeze(1), target)  # perfect preds, wrong shapes
        print(f"  perfect predictions, wrong shapes -> loss {bad.item():.3f}  (should be 0.000); "
              f"(pred - target) has shape {tuple((pred - target).shape)}")
        if w:
            print(f"  PyTorch warned: \"{str(w[0].message)[:90]}...\"")
    torch.manual_seed(args.seed)
    model_bug = make_mlp()
    gen = torch.Generator().manual_seed(args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist_bug = train(model_bug, torch.optim.Adam(model_bug.parameters(), lr=1e-3), x_train, y_train,
                         epochs, batch_size, gen, target_shape_bug=True)
    with torch.no_grad():
        pred_bug = model_bug(x_test).squeeze(1).numpy()
        test_bug = ((model_bug(x_test) - y_test_clean) ** 2).mean().item()
    print(f"  training WITH the bug (Adam): test MSE {test_bug:.4f}; prediction range "
          f"[{pred_bug.min():.3f}, {pred_bug.max():.3f}] vs mean(y_train) = {y_train.mean().item():.3f}")

    # ------------------------------------------------------------------ (b2) zero_grad
    print("\n(b2) Forgetting zero_grad: gradients accumulate")
    wpar = torch.tensor(1.0, requires_grad=True)
    for k in range(3):
        (3 * wpar).backward()               # d/dw (3w) = 3
        print(f"  after backward #{k + 1}: w.grad = {wpar.grad.item():.1f}")

    # ------------------------------------------------------------------ (b3) detach
    print("\n(b3) detach(): semi-gradient vs full gradient of a bootstrapped squared error")
    # value 'network' v(s) = w * x(s) with a scalar feature; one transition x=1 -> x'=0.5, reward 1
    gamma, s, s_next, r = 0.9, 1.0, 0.5, 1.0
    w1 = torch.tensor(0.5, requires_grad=True)
    target = r + gamma * w1 * s_next                     # depends on w1!
    loss_full = 0.5 * (target - w1 * s) ** 2
    g_full, = torch.autograd.grad(loss_full, w1)
    w2 = torch.tensor(0.5, requires_grad=True)
    target_d = (r + gamma * w2 * s_next).detach()        # treated as a constant label
    loss_semi = 0.5 * (target_d - w2 * s) ** 2
    g_semi, = torch.autograd.grad(loss_semi, w2)
    delta = (r + gamma * 0.5 * s_next - 0.5 * s)
    print(f"  TD error delta = {delta:.2f}; semi-gradient (detach) = {g_semi.item():.3f} = -delta*s; "
          f"full gradient = {g_full.item():.3f} = -delta*(s - gamma*s')")
    print(f"  -> different updates: without detach the target is also pulled toward the prediction "
          f"(Ch. 08 explains why TD uses the semi-gradient)")

    # ------------------------------------------------------------------ (b4) no_grad
    print("\n(b4) torch.no_grad() when acting / evaluating")
    model = make_mlp()
    out1 = model(x_test[:5])
    with torch.no_grad():
        out2 = model(x_test[:5])
    print(f"  outside no_grad: requires_grad={out1.requires_grad}, grad_fn={type(out1.grad_fn).__name__}; "
          f"inside: requires_grad={out2.requires_grad}, grad_fn={out2.grad_fn}")

    # ------------------------------------------------------------------ (b5) dtype
    print("\n(b5) float64 numpy array into a float32 network")
    obs = np.array([[0.1]])                               # numpy default dtype is float64
    try:
        model(torch.from_numpy(obs))
    except RuntimeError as e:
        print(f"  RuntimeError: {str(e).splitlines()[0][:80]}")
    ok = model(torch.as_tensor(obs, dtype=torch.float32))
    print(f"  fixed with torch.as_tensor(obs, dtype=torch.float32): output {tuple(ok.shape)}")

    # ------------------------------------------------------------------ (b6) gather
    print("\n(b6) Selecting Q(s, a) for the actions actually taken")
    q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (batch=2, actions=3)
    a = torch.tensor([2, 0])                               # (batch,)
    q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)          # (batch,)
    print(f"  q.shape={tuple(q.shape)}, a={a.tolist()} -> q.gather(1, a.unsqueeze(1)).squeeze(1) = {q_sa.tolist()}")
    print(f"  (the tempting q[:, a] gives shape {tuple(q[:, a].shape)}: every row indexed by every action)")

    # ------------------------------------------------------------------ (b7) bool arithmetic
    print("\n(b7) Arithmetic on a boolean `terminated` tensor")
    terminated = torch.tensor([True, False])               # what torch.as_tensor(np_bool_array) gives
    try:
        _ = 1.0 - terminated
    except Exception as e:                                 # PyTorch refuses '-' on bool tensors
        print(f"  1.0 - terminated  ->  {type(e).__name__}: {str(e).splitlines()[0][:70]}...")
    print(f"  fixed: 1.0 - terminated.float() = {(1.0 - terminated.float()).tolist()}")

    # ------------------------------------------------------------------ (b8) in-place modification
    print("\n(b8) In-place modification of a tensor that backward() needs")
    xin = torch.tensor([0.5, 1.0], requires_grad=True)
    yin = xin.exp()                                        # exp saves its OUTPUT for the backward pass
    yin += 1                                               # ...which this in-place add overwrites
    try:
        yin.sum().backward()
    except RuntimeError as e:
        print(f"  RuntimeError: {str(e).splitlines()[0][:95]}...")
    yok = xin.exp() + 1                                    # out-of-place: a new tensor, graph intact
    yok.sum().backward()
    print(f"  fixed (y = x.exp() + 1): grad = {xin.grad.tolist()} = exp(x)")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        ax = axes[0]
        ax.plot(x_train.squeeze(1), y_train.squeeze(1), "o", color=GREY, ms=3, alpha=0.6, label="training data")
        ax.plot(x_test.squeeze(1), y_test_clean.squeeze(1), color="#0b0b0b", lw=1.2, label="true function")
        ax.plot(x_test.squeeze(1), results["SGD lr=0.05"]["pred"], "-", color=C[0], lw=1.6, label="MLP, plain SGD")
        ax.plot(x_test.squeeze(1), results["Adam lr=1e-3"]["pred"], ":", color=C[2], lw=2.2, label="MLP, Adam")
        ax.plot(x_test.squeeze(1), pred_bug, "-.", color=C[7], label="MLP, Adam + (B,1)-vs-(B,) shape bug")
        ax.set_title("(a) Fits after training"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)

        ax = axes[1]
        styles = ["-", "--", ":"]
        for (name, res), c, ls in zip(results.items(), C, styles):
            ax.semilogy(np.arange(1, epochs + 1), res["hist"], ls, color=c, label=name)
        ax.semilogy(np.arange(1, epochs + 1), hist_bug, "-.", color=C[7], label="Adam with shape bug (its own, wrong, loss)")
        ax.axhline(0.01, color=GREY, lw=0.8, label="noise variance 0.01")
        ax.set_title("(b) Training loss per epoch"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "pytorch_regression.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    best = min(results, key=lambda k: results[k]["test"])
    print(f"\nSummary: best optimiser here: {best} (test MSE {results[best]['test']:.4f}); the shape bug "
          f"gives test MSE {test_bug:.3f}. runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
