import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Years from the Decisions section of README.md (Person A sets these)
# PLACEHOLDERS: replace once A posts their years
PRICE_YEARS = (2016, 2017)   # (before, after) for price
PACKS_YEARS = (2016, 2017)   # (before, after) for pack sales

TAX = 2.0
CA = "California"

# TEMPORARY: change back to "data/clean.csv" once A's data is merged
df = pd.read_csv("data/temporary_clean.csv")
df = df[~df["state"].str.contains("United States|National", case=False, na=False)]


def value(state_df, year, col):
    return state_df.loc[state_df["year"] == year, col].iloc[0]


# 1. Control states: same state tax every year 2015-2019
window = df[df["year"].between(2015, 2019)]
tax_check = window.groupby("state")["state_tax"].agg(["nunique", "count"])
controls = tax_check[(tax_check["nunique"] == 1) & (tax_check["count"] == 5)].index
controls = [s for s in controls if s != CA]
ctrl = df[df["state"].isin(controls)]
ca = df[df["state"] == CA]
print(f"{len(controls)} control states: {', '.join(controls)}")

# 2. Pass-through
pb, pa = PRICE_YEARS
ca_dp = value(ca, pa, "price") - value(ca, pb, "price")
ctrl_price = ctrl.pivot(index="year", columns="state", values="price")
ctrl_dp = (ctrl_price.loc[pa] - ctrl_price.loc[pb]).mean()
passthrough_naive = ca_dp / TAX
passthrough = (ca_dp - ctrl_dp) / TAX

# 3. Elasticity
qb, qa = PACKS_YEARS
ca_dlnq = np.log(value(ca, qa, "packs_pc")) - np.log(value(ca, qb, "packs_pc"))
ca_dlnp = np.log(value(ca, pa, "price")) - np.log(value(ca, pb, "price"))
ctrl_packs = ctrl.pivot(index="year", columns="state", values="packs_pc")
ctrl_dlnq = (np.log(ctrl_packs.loc[qa]) - np.log(ctrl_packs.loc[qb])).mean()
ctrl_dlnp = (np.log(ctrl_price.loc[pa]) - np.log(ctrl_price.loc[pb])).mean()
elasticity_naive = ca_dlnq / ca_dlnp
elasticity = (ca_dlnq - ctrl_dlnq) / (ca_dlnp - ctrl_dlnp)

q0 = value(ca, qb, "packs_pc")
p0 = value(ca, pb, "price")

print(f"Pass-through: naive {passthrough_naive:.3f}, vs controls {passthrough:.3f}")
print(f"Elasticity:   naive {elasticity_naive:.3f}, diff-in-diff {elasticity:.3f}")
print(f"q0 = {q0:.2f} packs per person, p0 = ${p0:.2f}")

Path("results").mkdir(exist_ok=True)
with open("results/estimates.json", "w") as f:
    json.dump({"q0": float(q0), "p0": float(p0),
               "passthrough": float(passthrough), "elasticity": float(elasticity)},
              f, indent=2)

# 4. Chart: packs per person, CA vs control average, 2016 = 100
years = range(2005, 2020)
ca_series = ca.set_index("year")["packs_pc"].reindex(years)
ctrl_series = ctrl_packs.mean(axis=1).reindex(years)
ca_idx = 100 * ca_series / ca_series.loc[2016]
ctrl_idx = 100 * ctrl_series / ctrl_series.loc[2016]
gap = (ca_idx.loc[qa] - 100) - (ctrl_idx.loc[qa] - 100)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(ca_idx.index, ca_idx.values, marker="o", label="California")
ax.plot(ctrl_idx.index, ctrl_idx.values, marker="o",
        label=f"Control states (avg of {len(controls)})")
ax.axvline(2017, color="gray", linestyle="--")
ax.set_title(f"California pack sales fell {abs(gap):.0f} points more than control states after the 2017 tax")
ax.set_xlabel("Year")
ax.set_ylabel("Packs sold per person (2016 = 100)")
ax.legend()
fig.text(0.01, 0.01, "Source: CDC, The Tax Burden on Tobacco, 1970-2019 (Orzechowski and Walker)",
         fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
Path("figs").mkdir(exist_ok=True)
fig.savefig("figs/packs.png", dpi=150)
print("Saved results/estimates.json and figs/packs.png")