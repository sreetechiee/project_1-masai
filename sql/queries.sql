-- Q1 WHERE then HAVING
SELECT customer_id,sum(order_amount) AS total
FROM orders
WHERE status = 'completed' AND order_amount>500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;
-- Q2 three-table join
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2)AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;
-- Q3 scalar subquery
SELECT *
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);
-- Q4 left join 
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;
-- Q5 common table expression
 WITH spend AS (
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
GROUP BY customer_id),
avg_spend AS (
SELECT AVG(total_spend) AS avg_total FROM spend
)
SELECT s.customer_id, s.total_spend, a.avg_total
FROM spend s, avg_spend a
WHERE s.total_spend > a.avg_total;

-- Q6 ranking window functions
WITH spend AS (
SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS drnk,
ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rn
FROM spend;

-- Q7 running total
WITH daily AS (
SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
FROM orders
WHERE status = 'completed'
GROUP BY substr(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
  ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily;
