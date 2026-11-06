import pandas as pd
import sqlite3, os

customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

# Look at the raw spellings first (useful for your audit and README)
print("raw city spellings:", customers["city"].nunique())
print("raw gender spellings:", customers["gender"].dropna().str.strip().replace("", pd.NA).dropna().nunique())
print(customers["gender"].value_counts(dropna=False))

# (a) City: remove leading/trailing spaces, then title case
customers["city"] = customers["city"].str.strip().str.title()

# (b) Gender: male / female / unknown (blank -> unknown)
g = customers["gender"].fillna("").astype(str).str.strip().str.lower()
customers["gender"] = "unknown"
customers.loc[g.str.startswith("m"), "gender"] = "male"
customers.loc[g.str.startswith("f"), "gender"] = "female"

# (c) Age: outside 10-100 becomes missing, then fill with the city median
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
out_of_range = (customers["age"] < 10) | (customers["age"] > 100)
print("ages outside 10-100 set to missing:", out_of_range.sum())
customers.loc[out_of_range, "age"] = float("nan")

print("missing ages before fill:", customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# (d) signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---------- Checks against the reference ----------
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("missing ages after fill:", customers["age"].isna().sum())
print("min age:", customers["age"].min(), "| max age:", customers["age"].max())
print(customers["signup_date"].dtype)

assert customers["city"].nunique() == 6
assert customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()
print("customers cleaning OK")

orders=pd.read_csv("data/raw/orders.csv")
products=pd.read_csv("data/raw/products.csv")
o = orders.copy()
# 1. datetime
o["order_ts"] = pd.to_datetime(o.order_ts)
# 2. exact duplicates
n0 = len(o); o = o.drop_duplicates(); print("dups removed", n0-len(o))
# 3. keep latest record per order_id
n0 = len(o); o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last"); print("removed", n0-len(o))
# 4. amount: numeric, then fill from price
o["order_amount"] = pd.to_numeric(o.order_amount, errors="coerce")
o = o.merge(products[["product_id","category","price_inr"]], on="product_id", how="left")
calc = (o.price_inr * o.quantity * (1 - o.discount_pct/100)).round(2)
o["order_amount"] = o.order_amount.fillna(calc)
# 5. delivery minutes
o.loc[o.delivery_minutes < 0, "delivery_minutes"] = None
med = o[o.status=="completed"].groupby("category").delivery_minutes.median()
mask = (o.status=="completed") & o.delivery_minutes.isna()
o.loc[mask, "delivery_minutes"] = o.loc[mask, "category"].map(med)

# ---------- Checks against the reference ----------
print("rows:", len(o))
print(o["status"].value_counts())
print("missing amounts:", o["order_amount"].isna().sum())
print("missing delivery (completed):", o.loc[o["status"] == "completed", "delivery_minutes"].isna().sum())
print("missing delivery (cancelled):", o.loc[o["status"] == "cancelled", "delivery_minutes"].isna().sum())
print("completed revenue:", round(o.loc[o["status"] == "completed", "order_amount"].sum(), 2))## Cleaning rules: orders

# ---------- Store: clean CSVs ----------
os.makedirs("data/clean", exist_ok=True)
customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
o.to_csv("data/clean/orders.csv", index=False)

# ---------- Store: SQLite database ----------
o_db = o.copy()
o_db["order_ts"] = o_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_db = customers.copy()
c_db["signup_date"] = c_db["signup_date"].dt.strftime("%Y-%m-%d")

con = sqlite3.connect("data/instamart.db")
c_db.to_sql("customers", con, if_exists="replace", index=False)
products.to_sql("products", con, if_exists="replace", index=False)
o_db.to_sql("orders", con, if_exists="replace", index=False)
con.close()
print("saved clean CSVs and data/instamart.db")