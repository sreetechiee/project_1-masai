import pandas as pd
orders=pd.read_csv("data/raw/orders.csv")
customers=pd.read_csv("data/raw/customers.csv")
products=pd.read_csv("data/raw/products.csv")
for name, df in [("orders", orders), ("customers", customers), ("products", products)]:
    print(name,df.shape)
    print(df.dtypes)
    print(pd.DataFrame({"missing": df.isna().sum(), "PCT": (df.isna().mean()*100).round(2)}))
    