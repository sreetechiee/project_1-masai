import sqlite3
import pandas as pd
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
done=orders[orders["status"]=="completed"]
cancelled=orders[orders["status"]=="cancelled"]

assert len(orders) == 12309, "order rows"
assert orders["order_id"].nunique()==12309, "one row per order_id"
assert len(done)==11110, "completed count"
assert len(cancelled)==1199, "cancelled count"
assert(orders["status"]=="pending").sum()==0, "no pending rows"
assert orders["order_amount"].isna().sum()==0, "no missing amounts"
assert done["delivery_minutes"].isna().sum()==0, "complete delivery filled"
assert cancelled["delivery_minutes"].isna().sum()==1199, "cancelled delivery left missing"
assert round(done["order_amount"].sum(),2)== 2998668.80, "completed revenue"
assert customers["city"].nunique()==6, "six cities"
assert customers["age"].between(18,60).all(), "ages 18-60"
assert customers["age"].isna().sum()==0, "no missing ages"
assert customers["gender"].value_counts().to_dict()=={"female":610,"male":560,"unknown":30}, "genders"

# the database must match too
con = sqlite3.connect("data/instamart.db")
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
assert {"customers", "products", "orders"} <= set(tables), "three tables"
n_db = pd.read_sql_query("SELECT COUNT(*) AS n FROM orders", con)["n"][0]
assert n_db == 12309, "orders rows in database"
con.close()

print("ALL CHECKPOINTS PASSED")