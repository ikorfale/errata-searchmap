"""Other half of the discount question: true pD 0.9; a searcher that assumes 0.3 (underconfident) vs one that assumes 0.9.
Same model as prior_sweep.py. Output: underconf.txt"""
import numpy as np
import prior_sweep as ps
ps.P_TRUE = 0.9
out = [f"{ps.N} wrecks per row, true pD 0.9, CRN, cap {ps.CAP}; U assumes 0.3, T assumes 0.9"]
for sig in (2.0, 4.0, 8.0, None):
    prior = ps.make_prior(sig); rng = np.random.default_rng(447); U, T = [], []
    for _ in range(ps.N):
        tgt = int(rng.choice(400, p=prior)); u = rng.random(ps.CAP)
        U.append(ps.search(prior, tgt, u, 0.3)); T.append(ps.search(prior, tgt, u, 0.9))
    U, T = np.array(U), np.array(T)
    out.append(f"sigma {'flat' if sig is None else sig:>5}: median U {np.median(U):.0f} T {np.median(T):.0f} | mean U {U.mean():.0f} T {T.mean():.0f}"
               f" | U slower {np.mean(U > T):.3f} faster {np.mean(U < T):.3f}")
    print(out[-1], flush=True)
open('underconf.txt', 'w').write('\n'.join(out) + '\n')
