import os
import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------
OUT_DIR = "data_raw"
N_TXNS = 220_000               # "nagy adat" demo
START_DATE = "2024-01-01"
END_DATE   = "2026-01-01"
CURRENCY = "EUR"
SEED = 42

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Dimensions
# ----------------------------
# Cost Centers
n_cc = 45
cc_ids = [f"CC{100+i:03d}" for i in range(n_cc)]
departments = ["Sales", "R&D", "G&A", "Operations", "Marketing"]
regions = ["North", "South", "East", "West", "Central"]

dim_cc = pd.DataFrame({
    "cost_center_id": cc_ids,
    "cost_center_name": [f"CostCenter {i+1:02d}" for i in range(n_cc)],
    "department": np.random.choice(departments, size=n_cc),
    "region": np.random.choice(regions, size=n_cc),
    "manager": [f"Manager_{i%10:02d}" for i in range(n_cc)]
})

# Accounts (mini P&L)
accounts = []

def add_accounts(prefix, start_num, names, pl1, pl2):
    for i, nm in enumerate(names):
        # account_id like A4100, A4200...
        acc_id = f"A{prefix}{start_num+i:02d}00"
        accounts.append((acc_id, nm, pl1, pl2, nm))

add_accounts(prefix="41", start_num=1,
             names=["Product revenue", "Service revenue", "Subscription revenue"],
             pl1="Revenue", pl2="Revenue Streams")

add_accounts(prefix="51", start_num=1,
             names=["Materials", "Freight", "Manufacturing overhead"],
             pl1="COGS", pl2="Direct Costs")

add_accounts(prefix="61", start_num=1,
             names=["Salaries", "Rent", "IT & Cloud", "Travel", "Professional services", "Office & supplies", "Training"],
             pl1="OPEX", pl2="Operating Expenses")

dim_acc = pd.DataFrame(accounts, columns=[
    "account_id", "account_name", "pl_level_1", "pl_level_2", "pl_level_3"
])

sort_map = {"Revenue": 1, "COGS": 2, "OPEX": 3}
dim_acc["sort_pl1"] = dim_acc["pl_level_1"].map(sort_map)

# ----------------------------
# Generate FactGL
# ----------------------------
dates = pd.date_range(START_DATE, END_DATE, freq="D")
countries = ["AT", "DE", "PL", "HU", "CZ"]
vendors = [f"Vendor_{i:03d}" for i in range(1, 80)]
customers = [f"Customer_{i:03d}" for i in range(1, 60)]

# Realistic distribution
pl_weights = {"Revenue": 0.18, "COGS": 0.22, "OPEX": 0.60}
pl_choices = np.random.choice(
    ["Revenue", "COGS", "OPEX"],
    size=N_TXNS,
    p=[pl_weights["Revenue"], pl_weights["COGS"], pl_weights["OPEX"]]
)

acc_by_pl = {
    k: dim_acc.loc[dim_acc["pl_level_1"] == k, "account_id"].tolist()
    for k in ["Revenue", "COGS", "OPEX"]
}

account_ids = np.array([np.random.choice(acc_by_pl[p]) for p in pl_choices])
cc_pick = np.random.choice(cc_ids, size=N_TXNS)
date_pick = np.random.choice(dates, size=N_TXNS)
country_pick = np.random.choice(countries, size=N_TXNS)

amount = np.zeros(N_TXNS)

# Revenue positive, larger
rev_idx = (pl_choices == "Revenue")
amount[rev_idx] = np.random.lognormal(mean=8.1, sigma=0.55, size=rev_idx.sum())

# COGS negative
cogs_idx = (pl_choices == "COGS")
amount[cogs_idx] = -np.random.lognormal(mean=7.7, sigma=0.50, size=cogs_idx.sum())

# OPEX negative, many smaller entries
opex_idx = (pl_choices == "OPEX")
amount[opex_idx] = -np.random.lognormal(mean=6.8, sigma=0.65, size=opex_idx.sum())

amount = np.round(amount, 2)

vendor_customer = np.where(
    rev_idx,
    np.random.choice(customers, size=N_TXNS),
    np.random.choice(vendors, size=N_TXNS)
)

fact_gl = pd.DataFrame({
    "txn_id": [f"TX{i:06d}" for i in range(1, N_TXNS+1)],
    "txn_date": pd.to_datetime(date_pick).date,
    "amount": amount,
    "currency": CURRENCY,
    "account_id": account_ids,
    "cost_center_id": cc_pick,
    "country": country_pick,
    "vendor_customer": vendor_customer,
    "description": np.where(rev_idx, "Invoice", "Expense")
})

# ----------------------------
# Generate Budget (monthly)
# ----------------------------
fact_gl["month_start"] = pd.to_datetime(fact_gl["txn_date"]).values.astype("datetime64[M]")

monthly_actual = (
    fact_gl.groupby(["month_start", "account_id", "cost_center_id"], as_index=False)["amount"]
    .sum()
)

def budget_factor(amt):
    # revenue slightly optimistic, costs slightly "less negative"
    return 1.03 if amt >= 0 else 0.98

monthly_actual["budget_amount"] = np.round(monthly_actual["amount"] * monthly_actual["amount"].apply(budget_factor), 2)

fact_budget = monthly_actual[["month_start", "account_id", "cost_center_id", "budget_amount"]].copy()

# ----------------------------
# Save files
# ----------------------------
p_fact_gl = os.path.join(OUT_DIR, "fact_gl.csv")
p_budget  = os.path.join(OUT_DIR, "fact_budget_monthly.csv")
p_acc     = os.path.join(OUT_DIR, "dim_account.csv")
p_cc      = os.path.join(OUT_DIR, "dim_costcenter.csv")

fact_gl.to_csv(p_fact_gl, index=False)
fact_budget.to_csv(p_budget, index=False)
dim_acc.to_csv(p_acc, index=False)
dim_cc.to_csv(p_cc, index=False)

print("Generated files:")
print(" -", p_fact_gl, "rows:", len(fact_gl))
print(" -", p_budget, "rows:", len(fact_budget))
print(" -", p_acc, "rows:", len(dim_acc))
print(" -", p_cc, "rows:", len(dim_cc))
