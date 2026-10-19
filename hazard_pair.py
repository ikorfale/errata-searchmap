"""xboss-xoxomo #94663: marginals of hazard_surv.txt cannot show a paired difference.
Same seeds and searches as hazard_surv.py; for the low (h/3.4) and high (h*3.4) beliefs prints, at each budget B,
the paired table both / low only / high only / neither and the exact two-sided McNemar p. Output: hazard_pair.txt"""
import numpy as np, math
import prior_sweep as ps

def search_t(prior, tgt, u, pa):  # copied from hazard_surv.py (importing it would rerun it)
    post = prior.copy(); looks = 0
    for s in range(1, ps.CAP + 1):
        c = int(np.argmax(post * pa))
        if c == tgt:
            looks += 1
            if u[looks - 1] < ps.P_TRUE: return s
        post[c] *= (1 - pa); post /= post.sum()
    return None

R = (1/3.4, 1/2, 1, 2, 3.4); B = (50, 200, 500, 1000, 2000)

def mcnemar(b, c):
    m = b + c
    if m == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m)

out = [f"{ps.N} wrecks per row, CRN, cap {ps.CAP}. Paired low=x{R[0]:.2f} vs high=x{R[-1]:.2f}: both / low only / high only / neither, exact McNemar p"]
for sig in (2.0, 4.0):
    prior = ps.make_prior(sig)
    for qt in (0.3, 0.5):
        h = -math.log(1 - qt); qa = [1 - math.exp(-h * r) for r in R]
        rng = np.random.default_rng(447); S = [[] for _ in R]
        for _ in range(ps.N):
            tgt = int(rng.choice(400, p=prior)); u = rng.random(ps.CAP)
            ps.P_TRUE = qt
            for j in (0, len(R) - 1): S[j].append(search_t(prior, tgt, u, qa[j]))
        lo = np.array([x if x is not None else np.inf for x in S[0]]); hi = np.array([x if x is not None else np.inf for x in S[-1]])
        out.append(f"sigma {sig} true {qt}")
        for b in B:
            fl, fh = lo <= b, hi <= b
            t = [int((fl & fh).sum()), int((fl & ~fh).sum()), int((~fl & fh).sum()), int((~fl & ~fh).sum())]
            out.append(f"  B={b:4d}: {t[0]:3d} / {t[1]:3d} / {t[2]:3d} / {t[3]:3d}   p={mcnemar(t[1], t[2]):.4g}")
        print('\n'.join(out[-6:]), flush=True)
open('hazard_pair.txt', 'w').write('\n'.join(out) + '\n')
