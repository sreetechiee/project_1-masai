import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")

for name, df in [("orders", orders), ("customers", customers), ("products", products)]:
    print(name, df.shape)
    print(df.dtypes)
    print(pd.DataFrame({"missing": df.isna().sum(),
                        "PCT": (df.isna().mean() * 100).round(2)}))

print("\n--- Specific audit figures ---")
amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
g = customers["gender"].dropna().astype(str).str.strip()

print("raw orders rows:", len(orders))                                         # 13084
print("exact duplicate rows:", orders.duplicated().sum())                      # 119
print("pending rows:", (orders["status"] == "pending").sum())                  # 656
print("distinct order_id:", orders["order_id"].nunique())                      # 12309
print("non-numeric order_amount:", amount_num.isna().sum())                    # 120
print("negative delivery_minutes:", (orders["delivery_minutes"] < 0).sum())    # 106
print("blank delivery_minutes:", orders["delivery_minutes"].isna().sum())      # 2541
print("blank age:", customers["age"].isna().sum())                             # 61
print("age outside 10-100:", ((customers["age"] < 10) | (customers["age"] > 100)).sum())  # 6
print("distinct raw city spellings:", customers["city"].nunique())             # 24
print("distinct non-blank gender spellings:", g[g != ""].nunique())            # 10
print("blank gender:", customers["gender"].isna().sum() + (g == "").sum())     # 30
