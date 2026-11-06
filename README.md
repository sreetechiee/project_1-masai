print(len(orders))                                   # 13084
print(orders.duplicated().sum())                     # 119
print((orders.status == "pending").sum())            # 656
print(orders.order_id.nunique())                     # 12309
print(pd.to_numeric(orders.order_amount, errors="coerce").isna().sum()
      - orders.order_amount.isna().sum())            # 120
print((orders.delivery_minutes < 0).sum())           # 106
print(orders.delivery_minutes.isna().sum())          # 2541
print(customers.age.isna().sum())                    # 61
print(((customers.age < 10) | (customers.age > 100)).sum())   # 6
print(customers.city.nunique())                      # 24
g = customers.gender.dropna()
print(g.nunique(), customers.gender.isna().sum())    # 10 and 30

## Cleaning rules: orders

| # | Rule | Why | Rows/values changed |
|---|------|-----|---------------------|
| 1 | Convert `order_ts` to datetime | Needed for sorting and date queries | all rows |
| 2 | Remove exact duplicate rows | Same record exported twice | 119 rows removed |
| 3 | Keep the latest `order_ts` per `order_id` | Earlier pending records are replaced by the final record | 656 rows removed |
| 4 | Convert `order_amount` to numeric (bad text becomes missing), then fill missing values with round(price_inr x quantity x (1 - discount_pct/100), 2) | Amount can be recomputed from the product price | ___ values filled |
| 5a | Set negative `delivery_minutes` to missing | Negative time is impossible | ___ values set to missing |
| 5b | Fill missing `delivery_minutes` for completed orders with the median of the product category (completed orders only) | Keeps typical delivery times without distorting the data | ___ values filled |
| 5c | Leave `delivery_minutes` missing for cancelled orders | A cancelled order was never delivered, so imputing a time would invent data | 1,199 left missing |

Result: 12,309 rows (11,110 completed, 1,199 cancelled, 0 pending), 0 missing amounts, completed revenue INR 2,998,668.80.