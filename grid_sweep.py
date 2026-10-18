"""Bayesian search with a miscalibrated sensor: true detection prob vs the one the searcher assumes.
Grid of 400 cells, prior = discretised 2-D Gaussian around a last known point; one cell searched per step
(greedy: max posterior * assumed pD); search until found or 2000 steps. Monte Carlo over target draws."""
import numpy as np, sys
rng = np.random.default_rng(1937)
G = 20; xs, ys = np.meshgrid(np.arange(G), np.arange(G))
prior = np.exp(-((xs - 7) ** 2 + (ys - 9) ** 2) / (2 * 4.0 ** 2)); prior /= prior.sum(); prior = prior.ravel()

def run(p_true, p_assumed, trials=2000, cap=2000):
    steps = []
    for _ in range(trials):
        tgt = rng.choice(G * G, p=prior); post = prior.copy()
        for s in range(1, cap + 1):
            c = int(np.argmax(post * p_assumed))
            if c == tgt and rng.random() < p_true: steps.append(s); break
            post[c] *= (1 - p_assumed); post /= post.sum()
        else: steps.append(cap)
    a = np.array(steps); return np.median(a), np.mean(a), np.mean(a >= cap), np.percentile(a, 90)

print("p_true p_assumed  median  mean  p90  share_not_found_in_2000")
for pt in (0.9, 0.5, 0.3):
    for pa in (0.9, 0.5, 0.3):
        med, mean, fail, p90 = run(pt, pa)
        print(f"{pt:6.1f} {pa:9.1f} {med:7.0f} {mean:5.0f} {p90:5.0f} {fail:8.3f}")
