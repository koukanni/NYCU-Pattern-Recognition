import pandas as pd
import matplotlib.pyplot as plt

bank = pd.read_csv("results/task2/explained_variance_bankruptcy.csv")
bean = pd.read_csv("results/task2/explained_variance_dry_bean.csv")

plt.figure(figsize=(8,5))
plt.plot(bank["k"], bank["cumulative_var_ratio"], label="Bankruptcy")
plt.plot(bean["k"], bean["cumulative_var_ratio"], label="Dry Bean")

plt.xlabel("Number of Principal Components $k$")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Explained Variance Curve (PCA)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
# plt.savefig("figs/task2_explained_variance.pdf")
plt.savefig("results/task2/explained_variance.png", dpi=300)