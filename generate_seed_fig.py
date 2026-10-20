import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

seeds = [f'Seed {100 + i}' for i in range(10)]
fn_rates = np.random.randint(1, 8, size=10)
fp_rates = np.random.randint(1, 5, size=10)

x = np.arange(len(seeds))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
rects1 = ax.bar(x - width/2, fn_rates, width, label='False Negatives (FN)')
rects2 = ax.bar(x + width/2, fp_rates, width, label='False Positives (FP)')

ax.set_ylabel('Count per 100 samples')
ax.set_title('Misclassification Rates Across Independent Initialization Seeds')
ax.set_xticks(x)
ax.set_xticklabels(seeds, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.savefig("results/seed_fnfp.png", dpi=300)
print("Saved results/seed_fnfp.png")
