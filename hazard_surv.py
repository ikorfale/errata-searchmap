"""xboss-xoxomo #93983: hazard.py means are min(T, CAP) over every run (unfound = CAP, nothing dropped).
Same seeds as hazard.py; adds censored/n and the share found within each budget B. Output: hazard_surv.txt"""
import numpy as np, math
import prior_sweep as ps

def search_t(prior, tgt, u, pa):  # like ps.search, but returns None when unfound by CAP
    post = prior.copy(); looks = 0
    for s in range(1, ps.CAP + 1):
        c = int(np.argmax(post * pa))
        if c == tgt:
            looks += 1
            if u[looks - 1] < ps.P_TRUE: return s
        post[c] *= (1 - pa); post /= post.sum()
    return None

R = (1/3.4, 1/2, 1, 2, 3.4); B = (50, 200, 500, 1000, 2000)
out = [f"{ps.N} wrecks per row, CRN, cap {ps.CAP}. Per assumed h ratio: found by B={'/'.join(map(str, B))}, censored/n"]
for sig in (2.0, 4.0):
    prior = ps.make_prior(sig)
    for qt in (0.3, 0.5):
        h = -math.log(1 - qt); qa = [1 - math.exp(-h * r) for r in R]
        rng = np.random.default_rng(447); S = [[] for _ in R]
        for _ in range(ps.N):
            tgt = int(rng.choice(400, p=prior)); u = rng.random(ps.CAP)
            ps.P_TRUE = qt
            for j, q in enumerate(qa): S[j].append(search_t(prior, tgt, u, q))
        out.append(f"sigma {sig} true {qt}")
        for r, s in zip(R, S):
            t = np.array([x if x is not None else np.inf for x in s])
            fb = ' '.join(f"{np.mean(t <= b):.3f}" for b in B)
            cm = np.mean(np.minimum(t, ps.CAP))
            out.append(f"  x{r:.2f}: {fb}   censored {int(np.isinf(t).sum())}/{len(t)}   capped mean {cm:.0f}")
        print('\n'.join(out[-6:]), flush=True)
open('hazard_surv.txt', 'w').write('\n'.join(out) + '\n')
