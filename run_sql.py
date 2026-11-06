import re
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql").read()

# split at each header like "-- Q1", "--q1", "-- q2 ..." (any case, optional space)
pieces = re.split(r"(?im)^--\s*q(\d+)[^\n]*\n", text)
results = {}
for i in range(1, len(pieces), 2):
    results["Q" + pieces[i]] = pd.read_sql_query(pieces[i + 1], con)
con.close()
print("queries found:", list(results.keys()))

# ---- report each result ----
q = results["Q1"]; print("Q1 customers:", len(q), "(reference 30)")

q = results["Q2"]
print("Q2 rows:", len(q), "(30), total:", round(q.revenue.sum(), 2), "(2998668.80)")
print(q.sort_values("revenue", ascending=False).head(1))

q = results["Q3"]; print("Q3 products:", len(q), "(11)")
q = results["Q4"]; print("Q4 customers with no completed order:", len(q), "(25)")

q = results["Q5"]
print("Q5 customers:", len(q), "(550), average:", round(q.avg_total.iloc[0], 2), "(2552.06)")

q = results["Q6"]
print("Q6 top customer per city:")
print(q[q.rnk == 1].sort_values("city")[["city", "customer_id", "total_spend"]])
print("Q6 rows where RANK differs from ROW_NUMBER:", (q.rnk != q.rn).sum(), "(8)")

q = results["Q7"]
print("Q7 daily rows:", len(q), "(183), last cumulative:", q.cumulative_revenue.iloc[-1], "(2998668.80)")