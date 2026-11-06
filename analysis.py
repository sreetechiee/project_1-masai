import sqlite3
import numpy as np
import pandas as pd

con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
products = pd.read_sql_query("SELECT * FROM products", con)
sql_total = pd.read_sql_query(
    "SELECT SUM(order_amount) AS t FROM orders WHERE status = 'completed'", con)["t"][0]
con.close()

# join the three tables, keep completed orders only
orders = orders.drop(columns=["category", "city"], errors="ignore")  # avoid column clashes
comp = (orders.merge(customers[["customer_id", "city"]], on="customer_id")
              .merge(products[["product_id", "category"]], on="product_id"))
comp = comp[comp["status"] == "completed"]

# pivot table with margins (grand total)
pv = pd.pivot_table(comp, values="order_amount", index="city",
                    columns="category", aggfunc="sum", margins=True,
                    margins_name="Total")
print(pv.round(2))
grand = pv.loc["Total", "Total"]
print("pivot grand total:", round(grand, 2), "| SQL total:", round(sql_total, 2))
assert round(grand, 2) == round(sql_total, 2) == 2998668.80
print("pivot total equals SQL total")

# per-city sum, mean and count (named aggregation)
city_stats = comp.groupby("city").agg(
    total_sum=("order_amount", "sum"),
    mean_amount=("order_amount", "mean"),
    order_count=("order_amount", "count"),
).round(2)
print(city_stats)

# ---------- Part (b): statistics and outliers ----------
s = comp["order_amount"]
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

print("\nmean:", round(s.mean(), 2), "| median:", round(s.median(), 2))
print("skewness:", round(s.skew(), 2))
print("Q1:", round(q1, 2), "| Q3:", round(q3, 2), "| IQR:", round(iqr, 2))
print("lower fence:", round(lower, 2), "| upper fence:", round(upper, 2))
print("outliers below:", (s < lower).sum(), "| outliers above:", (s > upper).sum())

# ---------- Correlation matrix ----------
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = comp[cols].corr()
print("\n", corr.round(3))

pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
best = pairs.abs().idxmax()
print("largest |correlation|:", best, round(pairs[best], 3))