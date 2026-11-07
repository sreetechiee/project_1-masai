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
| 4 | Convert `order_amount` to numeric (bad text becomes missing), then fill missing values with round(price_inr x quantity x (1 - discount_pct/100), 2) | Amount can be recomputed from the product price | 112 values filled |
| 5a | Set negative `delivery_minutes` to missing | Negative time is impossible | 105 values set to missing |
| 5b | Fill missing `delivery_minutes` for completed orders with the median of the product category (completed orders only) | Keeps typical delivery times without distorting the data | 774 values filled |
| 5c | Leave `delivery_minutes` missing for cancelled orders | A cancelled order was never delivered, so imputing a time would invent data | 1,199 left missing |

Result: 12,309 rows (11,110 completed, 1,199 cancelled, 0 pending), 0 missing amounts, completed revenue INR 2,998,668.80.

## SQL analysis

### Q6: ranking window functions
Top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10), all amounts in INR.

Rows where RANK() differs from ROW_NUMBER(): 8.

RANK() gives customers with identical total spend the same rank and then skips the next numbers, while ROW_NUMBER() always gives every row a unique number, so any tie in total_spend within a city makes the two columns differ.

# Pandas Analysis
This is order data from a quick-commerce retailer (Instamart): 11,110 completed orders across six cities, with product category, quantity, discount and delivery time. Order values are strongly right-skewed: the mean (INR 269.91) is far above the median (INR 135.00), skewness is about 4.61, and the middle half of orders lie between 64.80 and 297.00. There are no low outliers, but 897 orders exceed the upper fence of 645.30; I treated these as genuine large orders (expensive products or higher quantities), not errors, since I have no sign they are wrong. All correlations between order_amount, discount_pct, quantity and delivery_minutes are very weak. The largest in absolute value is order_amount with discount_pct (about -0.095), just above order_amount with quantity (+0.087). A correlation of this size does not establish causation.