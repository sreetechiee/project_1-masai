import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.decomposition import PCA

# ---------- Task 1: RFM table ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
snapshot = pd.Timestamp("2025-10-01 00:00:00")

done = orders[orders["status"] == "completed"]
rfm = done.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("customers:", len(rfm))
print("recency median/min/max:", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max())
print("frequency mean/max:", round(rfm["frequency"].mean(), 2), rfm["frequency"].max())
print("monetary mean/max:", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))

# ---------- Task 2: K-means for K = 2 to 8 ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))

table = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
print("\n", table.to_string(index=False))

# ---------- Task 3: profile, name, PCA, output ----------
FINAL_K = 4
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)

# name segments by their behaviour, not by their cluster number
premium_id = profile["mean_monetary"].idxmax()
rest = profile.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
atrisk_id = rest.drop(index=loyal_id)["mean_recency"].idxmax()

names = {i: "Occasional Buyers" for i in profile.index}
names[premium_id] = "Premium Big Spenders"
names[loyal_id] = "Loyal Regulars"
names[atrisk_id] = "At-Risk Lapsing"

profile["segment_name"] = profile.index.map(names)
print(profile.to_string())

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id

pca = PCA(n_components=2).fit(X)
print("explained variance ratio:", pca.explained_variance_ratio_.round(3),
      "total:", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id",
     "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("saved outputs/customer_segments.csv")