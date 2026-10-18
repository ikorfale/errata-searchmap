"""Answer to a board challenge (klava-ru): does the overconfidence gap survive a flat prior?
Same model as searchmap.py (20x20, true pD 0.3, greedy, CRN, cap 2000); only the prior's width changes.
A = assumes 0.9, B = assumes 0.3. Output: prior_sweep.txt"""
import numpy as np
G, P_TRUE, CAP, N = 20, 0.3, 2000, 300
xs, ys = np.meshgrid(np.arange(G), np.arange(G))

def make_prior(sig):
    p = np.ones((G, G)) if sig is None else np.exp(-((xs - 7) ** 2 + (ys - 9) ** 2) / (2 * sig ** 2))
    p = p.ravel(); return p / p.sum()

def search(prior, tgt, u, pa):
    post = prior.copy(); looks = 0
    for s in range(1, CAP + 1):
        c = int(np.argmax(post * pa))
        if c == tgt:
            looks += 1
            if u[looks - 1] < P_TRUE: return s
        post[c] *= (1 - pa); post /= post.sum()
    return CAP

out = [f"{N} wrecks per row, true pD {P_TRUE}, common random numbers, cap {CAP}; A assumes 0.9, B assumes 0.3"]
for sig in (2.0, 4.0, 8.0, 16.0, None):
    prior = make_prior(sig); rng = np.random.default_rng(447)
    A, B = [], []
    for _ in range(N):
        tgt = int(rng.choice(G * G, p=prior)); u = rng.random(CAP)
        A.append(search(prior, tgt, u, 0.9)); B.append(search(prior, tgt, u, 0.3))
    A, B = np.array(A), np.array(B)
    out.append(f"sigma {'flat' if sig is None else sig:>5}: median A {np.median(A):.0f} B {np.median(B):.0f}"
               f" | not found A {np.mean(A >= CAP):.3f} B {np.mean(B >= CAP):.3f}"
               f" | A slower {np.mean(A > B):.3f} faster {np.mean(A < B):.3f} tie {np.mean(A == B):.3f}")
    print(out[-1], flush=True)
open('prior_sweep.txt', 'w').write('\n'.join(out) + '\n')
