"""zenith-claude #93778: is the loss from a wrong pD symmetric in log hazard h = -ln(1-q)?
Same model as prior_sweep.py (20x20 independent cells, greedy, CRN). For true q, assumed q at h ratios 1/3.4, 1/2, 1, 2, 3.4.
Output: hazard.txt"""
import numpy as np, math
import prior_sweep as ps
out = [f"{ps.N} wrecks per row, CRN, cap {ps.CAP}; cells: mean steps (median) for assumed q at h x ratio"]
R = (1/3.4, 1/2, 1, 2, 3.4)
for sig in (2.0, 4.0):
    prior = ps.make_prior(sig)
    for qt in (0.3, 0.5):
        h = -math.log(1 - qt); qa = [1 - math.exp(-h * r) for r in R]
        rng = np.random.default_rng(447); S = [[] for _ in R]
        for _ in range(ps.N):
            tgt = int(rng.choice(400, p=prior)); u = rng.random(ps.CAP)
            ps.P_TRUE = qt
            for j, q in enumerate(qa): S[j].append(ps.search(prior, tgt, u, q))
        cells = '  '.join(f"x{r:.2f} q{q:.2f}: {np.mean(s):.0f} ({np.median(s):.0f})" for r, q, s in zip(R, qa, S))
        out.append(f"sigma {sig} true {qt}: {cells}"); print(out[-1], flush=True)
open('hazard.txt', 'w').write('\n'.join(out) + '\n')
