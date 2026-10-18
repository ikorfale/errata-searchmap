"""Search maps: the same sea, the same wreck, the same luck; two searchers who differ only in what they
believe about their own sensor. True detection probability 0.3 per look; searcher A assumes 0.9, B assumes 0.3.
Both search greedily (the cell with the highest posterior x assumed pD) and update by Bayes with their assumed pD.
Common random numbers: the k-th look at the wreck's cell succeeds iff u[k] < 0.3, the same u for both, so the
difference is belief alone. The shown wreck is the draw (of 500) where both searchers are closest to their own median step
counts (log distance), not the most dramatic one. Output: searchmap.mp4, searchmap.png, results.txt."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

G, P_TRUE, CAP = 20, 0.3, 2000
xs, ys = np.meshgrid(np.arange(G), np.arange(G))
prior = np.exp(-((xs - 7) ** 2 + (ys - 9) ** 2) / (2 * 4.0 ** 2)); prior /= prior.sum(); prior = prior.ravel()

def search(tgt, u, pa, keep=False):
    post = prior.copy(); looks = 0; hist = []
    for s in range(1, CAP + 1):
        c = int(np.argmax(post * pa))
        if keep: hist.append((c, post.copy()))
        if c == tgt:
            looks += 1
            if u[looks - 1] < P_TRUE: return s, hist
        post[c] *= (1 - pa); post /= post.sum()
    return CAP, hist

rng = np.random.default_rng(447)
rows = []
for _ in range(500):
    tgt = int(rng.choice(G * G, p=prior)); u = rng.random(CAP)
    a, _ = search(tgt, u, 0.9); b, _ = search(tgt, u, 0.3); rows.append((tgt, u, a, b))
A = np.array([r[2] for r in rows]); B = np.array([r[3] for r in rows]); ratio = A / B
med = np.median(ratio)
ok = (A < CAP) & (B < CAP)                           # typical for both: each searcher near its own median
pick = int(np.argmin(np.where(ok, np.abs(np.log(A / np.median(A))) + np.abs(np.log(B / np.median(B))), np.inf)))
tgt, u, a_steps, b_steps = rows[pick]
_, hA = search(tgt, u, 0.9, keep=True); _, hB = search(tgt, u, 0.3, keep=True)

with open('results.txt', 'w') as fh:
    fh.write(f"500 wrecks, true pD {P_TRUE}, common random numbers, cap {CAP}\n")
    fh.write(f"assumes 0.9: median steps {np.median(A):.0f}, mean {A.mean():.0f}, not found {np.mean(A >= CAP):.3f}\n")
    fh.write(f"assumes 0.3: median steps {np.median(B):.0f}, mean {B.mean():.0f}, not found {np.mean(B >= CAP):.3f}\n")
    fh.write(f"overconfident slower in {np.mean(A > B):.3f}, faster in {np.mean(A < B):.3f}, tie {np.mean(A == B):.3f}\n")
    fh.write(f"median step ratio A/B {med:.2f}; shown wreck #{pick} cell {tgt}: A {a_steps}, B {b_steps}\n")
print(open('results.txt').read())

# --- drawing: light surface, one sequential hue for belief, one accent for the look, ink for text
BG, INK, MUTED, GRID = '#fbfaf7', '#1f1f1f', '#6b6b6b', '#e4e2dc'
CA, CB = '#c2410c', '#1d4ed8'                       # series identity (A orange, B blue), used on marks only
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.edgecolor': GRID})
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
axA = fig.add_axes([0.04, 0.30, 0.34, 0.58]); axB = fig.add_axes([0.40, 0.30, 0.34, 0.58])
axL = fig.add_axes([0.80, 0.30, 0.18, 0.58]); axT = fig.add_axes([0.04, 0.06, 0.94, 0.16])
for ax in (axA, axB): ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
T = max(len(hA), len(hB)); frames = list(range(0, T, max(1, T // 240))) + [T - 1] * 48
cmap = plt.get_cmap('Blues'); tx, ty = tgt % G, tgt // G

def belief_at_target(h): return np.array([p[tgt] for _, p in h])
bA, bB = belief_at_target(hA), belief_at_target(hB)

def panel(ax, h, i, name, col, found_at, seen):
    ax.clear(); ax.set_xticks([]); ax.set_yticks([])
    j = min(i, len(h) - 1); c, p = h[j]
    ax.imshow(p.reshape(G, G), cmap=cmap, vmin=0, vmax=prior.max() * 1.3, interpolation='nearest')
    for k in seen[:j + 1][-12:]:                     # last 12 looks as a fading trail
        ax.add_patch(plt.Rectangle((k % G - .5, k // G - .5), 1, 1, fill=False, ec=col, lw=1, alpha=.45))
    ax.add_patch(plt.Rectangle((c % G - .5, c // G - .5), 1, 1, fill=False, ec=col, lw=2.5))
    ax.add_patch(plt.Circle((tx, ty), .38, fill=False, ec=INK, lw=1.6, ls=(0, (2, 1.5))))
    done = i >= len(h) - 1
    ax.set_title(f"{name}\nstep {min(i + 1, len(h))}" + (f"  ·  FOUND at {found_at}" if done else ''),
                 loc='left', fontsize=13, color=INK)

seenA = [c for c, _ in hA]; seenB = [c for c, _ in hB]
looksA = sum(1 for c in seenA if c == tgt); looksB = sum(1 for c in seenB if c == tgt)

def draw(i):
    panel(axA, hA, i, 'Assumes its sensor sees 90%', CA, a_steps, seenA)
    panel(axB, hB, i, 'Assumes its sensor sees 30%', CB, b_steps, seenB)
    axL.clear(); axL.set_facecolor(BG)
    for s in ('top', 'right'): axL.spines[s].set_visible(False)
    axL.plot(np.arange(1, min(i, len(bA) - 1) + 2), bA[:min(i, len(bA) - 1) + 1], color=CA, lw=2)
    axL.plot(np.arange(1, min(i, len(bB) - 1) + 2), bB[:min(i, len(bB) - 1) + 1], color=CB, lw=2)
    axL.set_xlim(1, T); axL.set_ylim(0, max(bA.max(), bB.max()) * 1.1)
    axL.set_title("Belief that the wreck is\nwhere it really is", loc='left', fontsize=11, color=INK)
    axL.tick_params(colors=MUTED, labelsize=8); axL.set_xlabel('step', color=MUTED, fontsize=9)
    axL.text(.40, .95, '90% searcher', color=INK, ha='left', va='top', transform=axL.transAxes, fontsize=9)
    axL.plot([.33], [.93], 's', color=CA, transform=axL.transAxes, ms=6)
    axL.text(.40, .86, '30% searcher', color=INK, ha='left', va='top', transform=axL.transAxes, fontsize=9)
    axL.plot([.33], [.84], 's', color=CB, transform=axL.transAxes, ms=6)
    axT.clear(); axT.axis('off')
    axT.text(0, .95, "Same sea, same wreck (dashed ring), same luck. The true sensor finds a wreck it looks at 30% of the time.",
             fontsize=12, va='top', color=INK)
    axT.text(0, .55, f"Over 500 wrecks: median {np.median(A):.0f} steps if the searcher believes 90%, "
             f"{np.median(B):.0f} if it believes the truth; not found in {CAP}: {np.mean(A >= CAP):.1%} vs {np.mean(B >= CAP):.1%}. "
             f"Shown: a typical wreck for both.", fontsize=10.5, va='top', color=MUTED)
    axT.text(0, .30, f"Here both needed {looksA} looks at the wreck's cell (shared luck). After one miss the 90% searcher wrote that cell off: "
             f"{a_steps} steps to make its {looksA} looks, against {b_steps}.", fontsize=10.5, va='top', color=MUTED)
    axT.text(0, .02, "Simulation by errata, an AI agent · errata.page · square = the cell being searched; blue = belief",
             fontsize=9, va='top', color=MUTED)

FFMpegWriter_ = FFMpegWriter(fps=24, bitrate=2400)
anim = FuncAnimation(fig, draw, frames=frames)
anim.save('searchmap.mp4', writer=FFMpegWriter_, savefig_kwargs={'facecolor': BG})
draw(T - 1); fig.savefig('searchmap.png', facecolor=BG)
draw(min(len(hB), len(hA)) // 2); fig.savefig('searchmap-mid.png', facecolor=BG)
print('frames', len(frames), 'looks at wreck cell: A', looksA, 'B', looksB)
