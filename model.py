"""Straight-line supply and demand model of California's $2 cigarette tax (Prop 56).

Calibrates linear demand and supply to B's estimates, solves for the after-tax
equilibrium, computes surplus changes and deadweight loss, and finds the
revenue-maximizing tax. Writes the results to results/model.json.

Curves (inverse form, price as a function of quantity per person):
    demand:  P_d(q) = d0 - d1 * q
    supply:  P_s(q) = s0 + s1 * q
"""

import json
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parent
ESTIMATES_PATH = ROOT / "results" / "estimates.json"  # B's file; never written here
MODEL_PATH = ROOT / "results" / "model.json"          # C's output

# Placeholder estimates, used until B's results/estimates.json is merged into main.
PLACEHOLDERS = {
    "q0": 20,            # pre-tax packs per person per year
    "p0": 6,             # pre-tax price per pack, dollars (includes old taxes)
    "elasticity": -0.5,  # price elasticity of demand at (q0, p0)
    "passthrough": 0.9,  # share of the tax that shows up in the buyers' price
}

TAX = 2.0  # Prop 56 increase, dollars per pack, on top of the old taxes


def load_estimates():
    """Read B's estimates if they exist, otherwise fall back to the placeholders."""
    if ESTIMATES_PATH.exists():
        with open(ESTIMATES_PATH) as f:
            est = json.load(f)
        print(f"Using estimates from {ESTIMATES_PATH.relative_to(ROOT)}")
    else:
        est = PLACEHOLDERS
        print("results/estimates.json not found: using PLACEHOLDER estimates")
    return {k: float(est[k]) for k in ("q0", "p0", "elasticity", "passthrough")}


def calibrate(q0, p0, elasticity, passthrough):
    """Pick slopes and intercepts so both curves pass through (q0, p0).

    q0, p0       -- the pre-tax equilibrium point
    elasticity   -- demand elasticity at (q0, p0); equals -(1/d1) * (p0/q0)
    passthrough  -- buyers' share of a per-unit tax; equals d1 / (d1 + s1)
    """
    d1 = -p0 / (q0 * elasticity)             # solve elasticity formula for d1
    s1 = d1 * (1 - passthrough) / passthrough  # solve passthrough formula for s1
    d0 = p0 + d1 * q0                         # demand intercept through (q0, p0)
    s0 = p0 - s1 * q0                         # supply intercept through (q0, p0)
    return d0, d1, s0, s1


def solve(t, d0, d1, s0, s1, q0, p0):
    """Solve the after-tax equilibrium for a per-unit tax t and compute welfare."""
    demand = lambda q: d0 - d1 * q  # buyers' willingness to pay at quantity q
    supply = lambda q: s0 + s1 * q  # sellers' marginal cost at quantity q

    # Quantity where the wedge between buyers' and sellers' price equals t.
    # Bracket [0, q0]: the wedge is d0 - s0 > t at q = 0 and 0 < t at q = q0.
    q1 = brentq(lambda q: demand(q) - supply(q) - t, 0, q0)
    pb, ps = demand(q1), supply(q1)  # buyers' price, sellers' price (incl. old taxes)

    # Surplus before (at q0, p0) and after (at q1, pb / ps); quad returns (value, error).
    cs0 = quad(lambda q: demand(q) - p0, 0, q0)[0]
    ps0 = quad(lambda q: p0 - supply(q), 0, q0)[0]
    cs1 = quad(lambda q: demand(q) - pb, 0, q1)[0]
    ps1 = quad(lambda q: ps - supply(q), 0, q1)[0]

    revenue = t * q1                                         # new tax revenue per person
    dwl = quad(lambda q: demand(q) - supply(q), q1, q0)[0]   # triangle between curves

    return {
        "q1": q1,
        "buyers_price": pb,
        "sellers_price": ps,
        "delta_cs": cs1 - cs0,
        "delta_ps": ps1 - ps0,
        "revenue": revenue,
        "dwl": dwl,
    }


def revenue_max(d0, d1, s0, s1):
    """Find the new tax t that maximizes t * q(t), and the analytic check (d0 - s0)/2."""
    q_of_t = lambda t: (d0 - s0 - t) / (d1 + s1)  # after-tax quantity for tax t
    # Minimize negative revenue over t in [0, d0 - s0] (beyond that, q would be negative).
    res = minimize_scalar(lambda t: -t * q_of_t(t), bounds=(0, d0 - s0), method="bounded")
    return res.x, -res.fun, (d0 - s0) / 2


def main():
    est = load_estimates()
    q0, p0 = est["q0"], est["p0"]

    # Step 1: calibrate the curves
    d0, d1, s0, s1 = calibrate(q0, p0, est["elasticity"], est["passthrough"])
    print(f"\nDemand: P = {d0:.3f} - {d1:.4f} q")
    print(f"Supply: P = {s0:.3f} + {s1:.4f} q")

    # Step 2: solve for t = $2
    r = solve(TAX, d0, d1, s0, s1, q0, p0)
    check = r["delta_cs"] + r["delta_ps"] + r["revenue"] + r["dwl"]

    print(f"\nTax of ${TAX:.2f} per pack (per person, per year)")
    print(f"  Quantity (packs)      {q0:8.3f} -> {r['q1']:8.3f}")
    print(f"  Buyers' price ($)     {p0:8.3f} -> {r['buyers_price']:8.3f}")
    print(f"  Sellers' price ($)    {p0:8.3f} -> {r['sellers_price']:8.3f}")
    print(f"  Change in CS ($)      {r['delta_cs']:8.3f}")
    print(f"  Change in PS ($)      {r['delta_ps']:8.3f}")
    print(f"  Revenue ($)           {r['revenue']:8.3f}")
    print(f"  Deadweight loss ($)   {r['dwl']:8.3f}")
    print(f"  Check dCS+dPS+R+DWL   {check:8.2e}")
    assert abs(check) < 1e-6, "Surplus accounting doesn't add up to zero"

    # Step 3: revenue-maximizing tax
    t_star, rev_star, t_analytic = revenue_max(d0, d1, s0, s1)
    print(f"\nRevenue-maximizing new tax: ${t_star:.3f} (analytic (d0 - s0)/2 = ${t_analytic:.3f})")
    print(f"  Revenue at that tax: ${rev_star:.3f} per person")

    out = {
        "inputs": est,
        "curves": {"d0": d0, "d1": d1, "s0": s0, "s1": s1},
        "tax": TAX,
        "q0": q0,
        "p0": p0,
        **r,
        "check_sum": check,
        "revenue_max_tax": t_star,
        "revenue_max_tax_analytic": t_analytic,
        "revenue_at_max": rev_star,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
