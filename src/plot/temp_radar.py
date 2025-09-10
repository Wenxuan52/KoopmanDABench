import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import font_manager
# print([f.name for f in font_manager.fontManager.ttflist])

# Data
models = ["DMD", "DMD ROM", "Koopman ROM", "Linear ROM", "Weaklinear ROM", "MLP ROM"]
metrics = ["Acc.", "Eff.", "Imp.", "Den.", "Noi."]
data = np.array([
    [0.20, 0.70, 0.90, 0.2000, 0.9743],  # DMD
    [0.24, 0.20, 1.00, 0.9493, 1.0000],  # DMD ROM
    [1.00, 0.82, 0.20, 0.9258, 0.7554],  # Koopman ROM
    [0.82, 0.78, 0.88, 1.0000, 0.8352],  # Linear ROM
    [0.87, 0.27, 0.89, 0.9819, 0.7761],  # Weaklinear ROM
    [0.93, 1.00, 0.68, 0.9546, 0.2000]   # MLP ROM
])

colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#e6c229"]

# Radar chart setup
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

fig, axes = plt.subplots(1, 6, subplot_kw=dict(polar=True), figsize=(26, 5))

for i, ax in enumerate(axes):
    values = data[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors[i], linewidth=2)
    ax.fill(angles, values, color=colors[i], alpha=0.35)
    ax.set_title(models[i], size=28, pad=50, fontweight="bold")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=24, fontweight="bold", fontname="STIXGeneral")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)

# Unified legend/explanation below

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'

fig.text(0.5, -0.15, 
         r"$\mathbf{Acc.}$ = Prediction Accuracy | $\mathbf{Eff.}$ = Data Assimilation Efficiency | $\mathbf{Imp.}$ = Data Assimilation Improvement | $\mathbf{Den.}$ = Observation Density Robustness | $\mathbf{Noi.}$ = Observation Noise Robustness",
         ha="center", fontsize=24)

plt.rcParams["font.family"] = plt.rcParamsDefault["font.family"]
fig.suptitle("CFDBench Cylinder Benchmark Comparison", fontsize=36, fontweight="bold", y=1.15)

plt.tight_layout()
plt.savefig('temp_radar.png', dpi=100, bbox_inches='tight')
plt.close