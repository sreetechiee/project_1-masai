import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import os
import joblib

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")

# ---------- Features: ONLY orders dated before the cutoff ----------
orders = orders.drop(columns=["category", "price_inr"], errors="ignore")  # avoid column clashes
before = orders[orders["order_ts"] < CUTOFF].merge(
    products[["product_id", "category"]], on="product_id", how="left")
done_before = before[before["status"] == "completed"]

feat = done_before.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
).reset_index()
feat["recency"] = (CUTOFF - feat["last_order"]).dt.days
feat = feat.drop(columns="last_order")

# cancellation rate = cancelled orders / all orders with a final status (before cutoff)
cancel = before.groupby("customer_id").agg(
    cancelled=("status", lambda s: (s == "cancelled").sum()),
    total=("status", "count"),
).reset_index()
cancel["cancel_rate"] = cancel["cancelled"] / cancel["total"]
feat = feat.merge(cancel[["customer_id", "cancel_rate"]], on="customer_id", how="left")

# most frequent product category among completed orders
fav = (done_before.groupby("customer_id")["category"]
       .agg(lambda s: s.value_counts().sort_index().idxmax())
       .rename("fav_category").reset_index())
feat = feat.merge(fav, on="customer_id", how="left")

# customer table: city, age, gender, tenure
feat = feat.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                  on="customer_id", how="left")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

# ---------- Label: no completed order from 2025-09-01 to 2025-09-30 ----------
september = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)
                   & (orders["status"] == "completed")]
active = set(september["customer_id"])
feat["lapsed"] = (~feat["customer_id"].isin(active)).astype(int)

print("cohort:", len(feat), "| lapsed:", feat["lapsed"].sum(),
      f"({feat['lapsed'].mean() * 100:.2f}%)")

# ---------- Split (no segment labels used as features) ----------
feature_cols = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
                "avg_delivery", "fav_category", "city", "age", "gender", "tenure_days"]
X = feat[feature_cols]
y = feat["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", len(X_train), "| test:", len(X_test))
print("missing values in features:", int(X.isna().sum().sum()))

# ---------- Baseline: always predict "not lapsed" ----------
pred = [0] * len(y_test)
print("baseline accuracy:", round(accuracy_score(y_test, pred) * 100, 1), "%")
print("baseline lapsed F1:", f1_score(y_test, pred, zero_division=0))

# ---------- Task 5: preprocessing, fitted on training rows only ----------
numeric_cols = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
                "avg_delivery", "age", "tenure_days"]
nominal_cols = ["city", "gender", "fav_category"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_cols),
])
preprocess.fit(X_train)                      # learns from TRAINING rows only
X_train_p = preprocess.transform(X_train)
X_test_p = preprocess.transform(X_test)      # only applies what was learned

print("processed train shape:", X_train_p.shape, "| test shape:", X_test_p.shape)
encoder = preprocess.named_transformers_["cat"]
for col, cats in zip(nominal_cols, encoder.categories_):
    print(col, "categories:", len(cats), "-> dummy columns:", len(cats) - 1)

scaler = preprocess.named_transformers_["num"]
freq_pos = numeric_cols.index("frequency")
print("scaler mean of frequency (training rows only):", round(scaler.mean_[freq_pos], 4))
print("full-cohort mean of frequency:", round(X["frequency"].mean(), 4))

print("any missing in raw features:", X.isna().any().any())
dense_train = X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
print("any missing after preprocessing:", bool(np.isnan(dense_train).any()))

# ---------- Task 6: train, tune, evaluate ----------
def make_pipe(model):
    return Pipeline([("prep", preprocess), ("model", model)])

# (b) Logistic regression: C in {0.01, 0.1, 1, 10, 100}, cv=5, scoring f1
lr_grid = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)),
                       {"model__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1")
lr_grid.fit(X_train, y_train)
print("LR best params:", lr_grid.best_params_, "| best CV F1:", round(lr_grid.best_score_, 4))

# (c) Random forest: max_depth x min_samples_leaf (45 fits)
rf_grid = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                       {"model__max_depth": [3, 5, 8],
                        "model__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
print("RF best params:", rf_grid.best_params_, "| best CV F1:", round(rf_grid.best_score_, 4))

# (d) Test-set evaluation (only now, after tuning is finished)
for name, grid in [("Logistic regression", lr_grid), ("Random forest", rf_grid)]:
    pred = grid.predict(X_test)
    print("\n" + name)
    print("confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, pred))
    print("lapsed precision:", round(precision_score(y_test, pred, zero_division=0), 3),
          "| recall:", round(recall_score(y_test, pred, zero_division=0), 3),
          "| F1:", round(f1_score(y_test, pred, zero_division=0), 3))

# ---------- Task 7: class-balance experiment (over-sample TRAINING rows only) ----------
print("\ntraining class counts before:", y_train.value_counts().to_dict())
lapsed_idx = y_train[y_train == 1].index
n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
extra_idx = pd.Series(lapsed_idx).sample(n=n_extra, replace=True, random_state=42).values

X_bal = pd.concat([X_train, X_train.loc[extra_idx]])
y_bal = pd.concat([y_train, y_train.loc[extra_idx]])
print("training class counts after: ", y_bal.value_counts().to_dict(), "| rows:", len(X_bal))

best_C = lr_grid.best_params_["model__C"]
lr_bal = make_pipe(LogisticRegression(max_iter=1000, C=best_C))
lr_bal.fit(X_bal, y_bal)

rows = []
for name, model in [("LR (original)", lr_grid.best_estimator_), ("LR (balanced)", lr_bal)]:
    pred = model.predict(X_test)
    rows.append((name,
                 round(precision_score(y_test, pred, zero_division=0), 3),
                 round(recall_score(y_test, pred, zero_division=0), 3),
                 round(f1_score(y_test, pred, zero_division=0), 3)))
print(pd.DataFrame(rows, columns=["model", "precision", "recall", "F1"]).to_string(index=False))
print("confusion matrix, balanced model [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, lr_bal.predict(X_test)))

# ---------- Task 8: interpret, persist, use ----------
# (a) top 5 features of the best random forest
rf_best = rf_grid.best_estimator_
names = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["model"].feature_importances_, index=names)
print("\nTop 5 random forest features:")
print(imp.sort_values(ascending=False).head(5).round(4))

# signs of three logistic regression coefficients
lr_best = lr_grid.best_estimator_
coef = pd.Series(lr_best.named_steps["model"].coef_[0],
                 index=lr_best.named_steps["prep"].get_feature_names_out())
print("\nLogistic regression coefficients (selected features):")
print(coef[["num__recency", "num__frequency", "num__cancel_rate"]].round(3))

# (b) save the selected model (random forest), reload it, compare predictions
selected = rf_best
os.makedirs("outputs", exist_ok=True)
joblib.dump(selected, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
print("\nreloaded model reproduces test predictions exactly:", same)

# (c) retention list: Premium customers in the TEST set, ranked by lapse probability
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
test_out = pd.DataFrame({
    "customer_id": feat.loc[X_test.index, "customer_id"].values,
    "lapse_probability": reloaded.predict_proba(X_test)[:, 1].round(4),
    "actually_lapsed": y_test.values,
})
prem = test_out[test_out["customer_id"].isin(premium_ids)]
retention = prem.sort_values("lapse_probability", ascending=False).head(10)
retention.to_csv("outputs/retention_list.csv", index=False)
print("\nPremium customers in the test set:", len(prem))
print(retention.to_string(index=False))
print("rows in list:", len(retention), "| truly lapsed among them:", int(retention["actually_lapsed"].sum()))