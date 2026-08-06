"""
generate_data.py

Simulates a tax-client population for a retention study.

Design notes (read this before you read the model code):
- We simulate a *confounded* intervention on purpose: clients who are more
  engaged are also more likely to receive the "proactive outreach" (call/email
  before filing season), AND more likely to retain anyway. This is what makes
  naive "compare outcomes between treated and untreated" analysis wrong, and
  is the whole reason a causal/uplift approach is needed instead of a plain
  classifier.
- True per-client treatment effects are also simulated heterogeneously:
  some segments genuinely respond to outreach, some don't. This lets us
  check, at the end, whether the uplift model recovers something close to
  the ground truth we built in -- a sanity check most public churn datasets
  can't offer, since we don't usually know the true causal effect in the
  real world.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_clients(n=20000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    filing_complexity = rng.choice(
        ["simple", "moderate", "complex"], size=n, p=[0.55, 0.30, 0.15]
    )
    complexity_score = pd.Series(filing_complexity).map(
        {"simple": 0, "moderate": 1, "complex": 2}
    ).to_numpy()

    tenure_years = rng.integers(1, 15, size=n)
    prior_refund = rng.gamma(shape=2.5, scale=900, size=n).round(2)
    channel = rng.choice(
        ["in_person", "online_diy", "online_assisted"], size=n, p=[0.35, 0.40, 0.25]
    )
    channel_score = pd.Series(channel).map(
        {"in_person": 2, "online_assisted": 1, "online_diy": 0}
    ).to_numpy()

    engagement = (
        0.15 * complexity_score
        + 0.25 * channel_score
        + 0.05 * np.minimum(tenure_years, 8)
        + rng.normal(0, 1, size=n)
    )
    engagement_pctile = pd.Series(engagement).rank(pct=True).to_numpy()

    prior_satisfaction = np.clip(
        rng.normal(loc=6.5 + 1.5 * engagement_pctile, scale=1.5, size=n), 0, 10
    ).round(1)

    price_sensitivity = np.clip(rng.normal(0.5, 0.2, size=n), 0, 1).round(2)

    propensity = 1 / (1 + np.exp(-(2.0 * engagement_pctile - 1.0)))
    treatment = rng.binomial(1, propensity)

    persuadable = (
        (complexity_score == 1).astype(float) * 0.10
        + (price_sensitivity > 0.5).astype(float) * 0.06
        + ((tenure_years >= 2) & (tenure_years <= 7)).astype(float) * 0.05
        - (prior_satisfaction < 4).astype(float) * 0.08
    )
    true_uplift = np.clip(persuadable + rng.normal(0, 0.02, size=n), -0.05, 0.25)

    base_logit = (
        -0.6
        + 0.35 * (prior_satisfaction - 5)
        + 0.20 * np.minimum(tenure_years, 6) / 6
        - 0.30 * complexity_score
        - 0.9 * price_sensitivity
        + 0.4 * channel_score
    )
    base_prob = 1 / (1 + np.exp(-base_logit))

    retention_prob = np.clip(base_prob + treatment * true_uplift, 0.02, 0.98)
    retained_next_year = rng.binomial(1, retention_prob)

    df = pd.DataFrame(
        {
            "client_id": np.arange(1, n + 1),
            "filing_complexity": filing_complexity,
            "tenure_years": tenure_years,
            "prior_refund": prior_refund,
            "channel": channel,
            "prior_satisfaction": prior_satisfaction,
            "price_sensitivity": price_sensitivity,
            "engagement_pctile": engagement_pctile.round(3),
            "received_outreach": treatment,
            "retained_next_year": retained_next_year,
            "_true_uplift": true_uplift.round(4),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_clients()
    df.to_csv("data/clients.csv", index=False)
    print(f"Generated {len(df):,} clients -> data/clients.csv")
    print(df.drop(columns=["_true_uplift"]).head())
