import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

csv_path = "/crex/proj/uppmax2025-3-5/private/yaxj1/lora/mt_oli_en_no/experiments_npd/experiments/02_parameter_sensitivity_optuna/results.csv"
df = pd.read_csv(csv_path)

df['bleu'] = df['values_0']
df['chrf'] = -df['values_1']

print("Data check:")
print(f"BLEU range: [{df['bleu'].min():.4f}, {df['bleu'].max():.4f}]")
print(f"chrF range: [{df['chrf'].min():.4f}, {df['chrf'].max():.4f}]")

X = df[['params_alpha', 'params_r', 'params_dropout']]
y = df['bleu']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

plt.figure(figsize=(8, 5))
params = ['LoRA Alpha', 'LoRA Rank', 'Dropout']
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = plt.bar(params, importances, color=colors, edgecolor='black', linewidth=1.2)
plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
plt.title('Hyperparameter Importance from Optuna Study', fontsize=14, fontweight='bold')
plt.ylim(0, max(importances) * 1.15)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
output_dir = "/crex/proj/uppmax2025-3-5/private/yaxj1/lora/mt_oli_en_no/experiments_npd/experiments/02_parameter_sensitivity_optuna"
plt.savefig(f'{output_dir}/optuna_importance.png', dpi=300, bbox_inches='tight')
plt.close()

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

costs = np.column_stack([df['bleu'], df['chrf']])
pareto_mask = is_pareto_efficient(costs)

plt.figure(figsize=(10, 6))
plt.scatter(df[~pareto_mask]['bleu'], df[~pareto_mask]['chrf'], 
            c='lightgray', s=50, alpha=0.6, label='Other Trials', edgecolors='gray')
plt.scatter(df[pareto_mask]['bleu'], df[pareto_mask]['chrf'], 
            c='red', s=100, alpha=0.8, label='Pareto Front', edgecolors='darkred', linewidth=1.5)

best_idx = (df['bleu'] + df['chrf']).idxmax()
plt.scatter(df.loc[best_idx, 'bleu'], df.loc[best_idx, 'chrf'], 
            c='gold', s=300, marker='*', label='Selected Config', 
            edgecolors='black', linewidth=2, zorder=5)

plt.xlabel('BLEU Score', fontsize=12, fontweight='bold')
plt.ylabel('chrF Score', fontsize=12, fontweight='bold')
plt.title('Pareto Front for BLEU and chrF Scores', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f'{output_dir}/pareto_front.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved successfully!")
print(f"\nParameter Importances:")
for param, imp in zip(params, importances):
    print(f"  {param}: {imp:.4f}")
print(f"\nPareto Front: {pareto_mask.sum()} configurations")
print(f"Selected Config (r={df.loc[best_idx, 'params_r']}, "
      f"alpha={df.loc[best_idx, 'params_alpha']}, "
      f"dropout={df.loc[best_idx, 'params_dropout']}):")
print(f"  BLEU: {df.loc[best_idx, 'bleu']:.4f}")
print(f"  chrF: {df.loc[best_idx, 'chrf']:.4f}")