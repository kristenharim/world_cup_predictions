"""
chart.py — branded win-probability chart (horizontal layout).

Saves a PNG to predictions/<date>/viz_<Home>_vs_<Away>.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK,  MUTE, GRID  = "#1a1a2e", "#8a8a9e", "#e8e8ee"
ORANGE, BLUE, GRAY = "#ff6b18", "#1f6feb", "#9aa0a6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def make_chart(fx: dict, p_home: float, p_draw: float, p_away: float,
               lam_home: float, lam_away: float, out_dir: str) -> str:
    """
    Draw a bar chart of the three outcome probabilities and save it.

    Parameters
    ----------
    fx       : fixture dict (home_disp, away_disp, group, stadium, date)
    p_home   : home-win probability
    p_draw   : draw probability
    p_away   : away-win probability
    lam_home : expected goals (home)
    lam_away : expected goals (away)
    out_dir  : directory to write the PNG into

    Returns the file path.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = [f"{fx['home_disp']}\nwin", "Draw", f"{fx['away_disp']}\nwin"]
    vals   = [p_home, p_draw, p_away]
    colors = [ORANGE, GRAY, BLUE]

    bars = ax.bar(labels, [v * 100 for v in vals],
                  color=colors, width=0.55, zorder=3)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.0,
                f"{v*100:.1f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold")

    ax.set_ylim(0, max(vals) * 100 + 14)
    ax.set_ylabel("Probability (%)")

    subtitle = (f"{fx['date']}  ·  {fx['group']}  ·  {fx['stadium']}\n"
                f"xG  {lam_home:.2f} – {lam_away:.2f}")
    ax.set_title(f"{fx['home_disp']} vs {fx['away_disp']}\n{subtitle}",
                 fontsize=12)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    safe = f"{fx['home_disp']}_vs_{fx['away_disp']}"\
           .replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, f"viz_{safe}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
