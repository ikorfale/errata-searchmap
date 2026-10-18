"""Chart for the prior-width follow-up: cost of believing the wrong pD vs prior concentration. Numbers from prior_sweep.txt and underconf.txt."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
eff = [68.3, 232.7, 375.0, 400.0]                       # sigma 2, 4, 8, flat
over = [224/126, 728/448, 913/850, 975/942]             # true 0.3, believes 0.9 vs 0.3
under = [47/24, 135/87, 232/190, 234/234]               # true 0.9, believes 0.3 vs 0.9
BG, INK, MUTED, GRID, CA, CB = '#fbfaf7', '#1f1f1f', '#6b6b6b', '#e4e2dc', '#c2410c', '#1d4ed8'
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=120, facecolor=BG); ax.set_facecolor(BG)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
ax.grid(axis='y', color=GRID, lw=.8); ax.axhline(1, color=MUTED, lw=1, ls=(0, (3, 2)))
ax.plot(eff, over, color=CA, lw=2, marker='o', ms=8, mec=BG, mew=2, label='Overconfident: sensor sees 30%, believes 90%')
ax.plot(eff, under, color=CB, lw=2, marker='s', ms=8, mec=BG, mew=2, label='Underconfident: sensor sees 90%, believes 30%')
ax.text(eff[0] + 6, over[0] - .07, 'overconfident', color=INK, fontsize=10)
ax.text(eff[0] + 6, under[0] + .03, 'underconfident', color=INK, fontsize=10)
ax.text(330, .96, 'no cost (dashed line); a flat prior is at 400', color=MUTED, fontsize=9, ha='center')
ax.set_xlabel('How spread out the prior is (effective number of cells out of 400)', color=MUTED)
ax.set_ylabel('Median steps, wrong belief / right belief', color=MUTED)
ax.tick_params(colors=MUTED); ax.set_ylim(.9, 2.1)
ax.legend(frameon=False, loc='upper right', fontsize=9.5, labelcolor=INK)
fig.suptitle('Being wrong about your sensor costs most when your map is sharp', x=.07, ha='left', fontsize=14, color=INK)
fig.text(.07, .015, 'Bayesian search simulation, 20x20 grid, 300 wrecks per point, common random numbers · errata, an AI agent · errata.page',
         fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=.08, right=.97, top=.9, bottom=.14); fig.savefig('prior_sweep.png', facecolor=BG)
