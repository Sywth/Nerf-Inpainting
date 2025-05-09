# %%
import pandas as pd
import numpy as np

# %%
# Initialize summary dict
summary = {"Mean PSNR": {}, "Mean SSIM": {}}

# Step 1: Process NeRF (l=1 to 4)
for l in range(1, 5):
    df = pd.read_csv(f"results-{l}.csv")
    nerf_row = df[df["Method"] == f"NeRF (l={l})"].iloc[0]
    psnr_vals = [nerf_row[f"PSNRs ({i})"] for i in range(1, 6)]
    ssim_vals = [nerf_row[f"SSIMs ({i})"] for i in range(1, 6)]
    summary["Mean PSNR"][f"NeRF (l={l})"] = np.mean(psnr_vals)
    summary["Mean SSIM"][f"NeRF (l={l})"] = np.mean(ssim_vals)

# Step 2: Process constant methods from any CSV
df_const = pd.read_csv("results-1.csv")
for method in ["Biharmonic", "FMM", "NS", "TV"]:
    row = df_const[df_const["Method"] == method].iloc[0]
    psnr_vals = [row[f"PSNRs ({i})"] for i in range(1, 6)]
    ssim_vals = [row[f"SSIMs ({i})"] for i in range(1, 6)]
    summary["Mean PSNR"][method] = np.mean(psnr_vals)
    summary["Mean SSIM"][method] = np.mean(ssim_vals)

# Step 3: Construct final DataFrame
summary_df = pd.DataFrame(summary).T
summary_df = summary_df[
    [
        "NeRF (l=1)",
        "NeRF (l=2)",
        "NeRF (l=3)",
        "NeRF (l=4)",
        "Biharmonic",
        "FMM",
        "NS",
        "TV",
    ]
]

summary_df

# %%
summary_df.to_csv("2dnerf-summary.csv")

# %%
with open("2dnerf-summary.tex", "w") as f:
    f.write(summary_df.to_latex(index=True, float_format="%.3f"))
