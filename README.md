# Instamart Customer Copilot

## Overview
This project turns raw quick-commerce exports into decisions. Part A cleans dirty order and customer data and analyses it with SQL and pandas. Part B groups customers into segments and predicts which customers will stop ordering. Part C triages support tickets safely around a mock language model. Part D builds a policy question-answering assistant (retrieval) and a refund agent that calls tools. All money amounts are in INR, and no real network call or API key is used anywhere.

| Part | Scripts | Main outputs |
|---|---|---|
| A | `generate_data.py`, `audit.py`, `clean.py`, `validate.py`, `run_sql.py`, `analysis.py` | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | `segments.py`, `lapse.py` | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | `triage.py`, `mock_llm.py` | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/retry_log.txt` |
| D | `rag.py`, `agent.py` | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

### Project structure
```
generate_data.py, mock_llm.py     supplied files (unmodified)
audit.py, clean.py, validate.py   Part A: audit, cleaning, database, checks
run_sql.py, analysis.py           Part A: SQL queries and pandas analysis
segments.py, lapse.py             Part B: segments and lapse model
triage.py                         Part C: ticket triage
rag.py, agent.py                  Part D: policy assistant and refund agent
sql/queries.sql                   the seven SQL queries
data/                             raw and clean data, the SQLite database
outputs/                          generated results (CSV, model, logs)
state/                            agent checkpoints
```

## Reproduce
1. Clone the repository: `git clone https://github.com/sreetechiee/project_1-masai.git` and open the folder.
2. Create a virtual environment: `python -m venv venv`
3. Activate it (Windows PowerShell): `venv\Scripts\Activate.ps1`
4. Install the libraries: `pip install -r requirements.txt`
5. Generate the raw data: `python generate_data.py`
6. Audit the raw data: `python audit.py`
7. Clean the data and build the database: `python clean.py`
8. Validate the clean data: `python validate.py` (prints `ALL CHECKPOINTS PASSED`)
9. Run the SQL queries: `python run_sql.py`
10. Run the pandas analysis: `python analysis.py`
11. Build the customer segments (Part B): `python segments.py`
12. Train the lapse model (Part B): `python lapse.py`
13. Triage the support tickets (Part C): `python triage.py`
14. Build and evaluate the policy assistant (Part D): `python rag.py`
15. Run the refund agent, its limits and the crash-and-resume demo (Part D): `python agent.py`

Run the steps in this order, because later scripts read the files made by earlier ones.

---


## Part A: Data Foundation (Clean, Store, Query)

The raw exports are deliberately dirty. In this part I audited them, cleaned them with documented rules, stored the clean tables in SQLite, and answered business questions in SQL and in pandas. All money amounts are in INR.

print(len(orders))                                   ## 13084
print(orders.duplicated().sum())                     ## 119
print((orders.status == "pending").sum())            ## 656
print(orders.order_id.nunique())                     ## 12309
print(pd.to_numeric(orders.order_amount, errors="coerce").isna().sum()
      - orders.order_amount.isna().sum())            ## 120
print((orders.delivery_minutes < 0).sum())           ## 106
print(orders.delivery_minutes.isna().sum())          ## 2541
print(customers.age.isna().sum())                    ## 61
print(((customers.age < 10) | (customers.age > 100)).sum())   ## 6
print(customers.city.nunique())                      ## 24
g = customers.gender.dropna()
print(g.nunique(), customers.gender.isna().sum())    ## 10 and 30

### Cleaning rules: orders

| ## | Rule | Why | Rows/values changed |
|---|------|-----|---------------------|
| 1 | Convert `order_ts` to datetime | Needed for sorting and date queries | all rows |
| 2 | Remove exact duplicate rows | Same record exported twice | 119 rows removed |
| 3 | Keep the latest `order_ts` per `order_id` | Earlier pending records are replaced by the final record | 656 rows removed |
| 4 | Convert `order_amount` to numeric (bad text becomes missing), then fill missing values with round(price_inr x quantity x (1 - discount_pct/100), 2) | Amount can be recomputed from the product price | 112 values filled |
| 5a | Set negative `delivery_minutes` to missing | Negative time is impossible | 105 values set to missing |
| 5b | Fill missing `delivery_minutes` for completed orders with the median of the product category (completed orders only) | Keeps typical delivery times without distorting the data | 774 values filled |
| 5c | Leave `delivery_minutes` missing for cancelled orders | A cancelled order was never delivered, so imputing a time would invent data | 1,199 left missing |

Result: 12,309 rows (11,110 completed, 1,199 cancelled, 0 pending), 0 missing amounts, completed revenue INR 2,998,668.80.

### Cleaning rules: customers

| ## | Rule | Rows/values changed |
|---|------|---------------------|
| a | Strip spaces and title-case `city` (24 raw spellings become 6 cities) | 24 spellings merged into 6 |
| b | Map every `gender` spelling to male, female or unknown (blank becomes unknown) | 10 spellings mapped, 30 blanks set to unknown |
| c | Treat ages outside 10 to 100 as missing, then fill missing ages with the city median | 6 set to missing, 67 filled in total |
| d | Convert `signup_date` to datetime | all rows |

### Audit of the raw data
| Table | Shape | Missing values |
|---|---|---|
| orders | 13,084 rows, 9 columns | order_amount: 53 (0.41%), delivery_minutes: 2,541 (19.42%), all other columns 0 (0.00%) |
| customers | 1,200 rows, 5 columns | age: 61 (5.08%), gender: 30 (2.50%), all other columns 0 (0.00%) |
| products | 40 rows, 4 columns | none (0.00% in every column) |

Data types: in orders, `quantity` and `discount_pct` are integers, `delivery_minutes` is a float, and the other columns are text (`order_amount` is text because some entries are not numbers). In customers, `age` is a float and the other columns are text. In products, `price_inr` is an integer and the other columns are text.

Specific audit figures:

| Check | Result |
|---|---|
| Raw orders rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id values | 12,309 |
| order_amount entries that cannot be converted to a number | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10 to 100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank genders | 30 |

## SQL analysis
The seven queries are saved in `sql/queries.sql` and run from Python with `pd.read_sql_query` (`python run_sql.py`). Only completed orders count as revenue or spend.

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers |
| Q2 | Three-table join (revenue by city and category) | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | Scalar subquery (price above the average) | 11 products |
| Q4 | LEFT JOIN (customers with no completed order) | 25 customers |
| Q5 | Common table expression (spend above the average) | 550 customers, average INR 2,552.06 |
| Q6 | RANK, DENSE_RANK and ROW_NUMBER per city | Top customers: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 | Running total of daily revenue | 183 daily rows, last cumulative value INR 2,998,668.80 |

For Q6: RANK() gives customers with identical total spend the same rank and then skips the next numbers, while ROW_NUMBER() always gives every row a unique number, so any tie in total spend within a city makes the two columns differ.

### pandas cross-check and exploratory analysis
I rebuilt Q2 with `pd.pivot_table(..., margins=True)`. Its grand total is INR 2,998,668.80, which equals the SQL total.

Completed orders per city (`groupby` with named aggregation):

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Statistics of completed `order_amount`:

| Statistic | Value |
|---|---|
| Mean | 269.91 |
| Median | 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| 1.5 x IQR fences | lower fence below zero, upper fence 645.30 |
| Outliers below / above | 0 / 897 |

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**What the data shows.** This is order data from a quick-commerce retailer: 11,110 completed orders across six cities, with product category, quantity, discount and delivery time. Order values are strongly right-skewed: the mean (INR 269.91) is well above the median (INR 135.00) and skewness is about 4.61. Half of all orders fall between INR 64.80 and INR 297.00. There are no low outliers, but 897 orders lie above the upper fence of INR 645.30. I treat them as genuine large purchases, not errors. Correlations between order_amount, discount_pct, quantity and delivery_minutes are all very weak. The largest in absolute value is order_amount with discount_pct (about -0.095), just above order_amount with quantity (+0.087). A correlation of this size does not establish causation.


## Part B: Customer Segments and Lapse Prediction

### What I did in Part B
The retention team wanted two things. First, groups of similar customers (segments) so they can plan campaigns. Second, a model that predicts which customers are likely to stop ordering (lapse). I used only **completed orders** from `data/clean/orders.csv` and the clean customers file from Part A.

**How to run:** `python segments.py` for the segments, then `python lapse.py` for the lapse model. The results are saved in the `outputs` folder.

---

### Task 1: RFM table
RFM describes each customer with three numbers. I measured them on the snapshot date **1 October 2025**.
- **Recency:** days since the customer's last completed order (smaller is better).
- **Frequency:** how many different completed orders they made.
- **Monetary:** how much they spent in total, in INR.

| Measure | Result |
|---|---|
| Customers with at least one completed order | 1,175 |
| Recency (median / min / max) | 19 / 0 / 180 days |
| Frequency (mean / max) | 9.46 / 33 |
| Monetary (mean / max) | INR 2,552.06 / INR 14,766.55 |

### Task 2: K-means clustering
K-means groups similar customers together. The three RFM columns are in different units (days, counts, rupees), so I first standardised them with `StandardScaler`. Then I tried K = 2 to 8 groups and recorded two scores: inertia (lower means tighter groups) and silhouette (higher means better separated groups).

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**I chose K = 4** for three reasons:
- On the elbow chart, inertia falls fast until K = 4 and then falls slowly.
- K = 4 has the best silhouette score (0.526).
- Four groups are easy for the business to understand and act on.

**Why the lowest inertia cannot choose K:** inertia always goes down when I add more clusters, so K = 8 will always look better than K = 2. If every customer had their own cluster, inertia would be zero, which is useless. So I also used the silhouette score and common sense.

### Task 3: Segment profiles
The cluster numbers (0 to 3) are only labels and have no order. I named each segment by looking at its averages.

| Segment | Customers | Avg recency (days) | Avg frequency | Avg spend (INR) | Action |
|---|---|---|---|---|---|
| **Premium Big Spenders** | 138 | 24.1 | 6.8 | 7,129.7 | Give VIP perks and early access to keep our best spenders |
| **Loyal Regulars** | 419 | 8.8 | 19.4 | 3,410.0 | Start a loyalty rewards program so they keep ordering |
| **Occasional Buyers** | 436 | 28.9 | 3.8 | 1,057.6 | Send offers to make them order more often |
| **At-Risk Lapsing** | 182 | 109.0 | 2.2 | 686.0 | Send a win-back discount before we lose them |

The segment with the highest average spend (Premium Big Spenders) is marked `is_premium = True`.

**PCA:** I used PCA to squeeze the 3 RFM columns into 2. The two components explain 63.4% and 21.8% of the variation, 85.2% in total, so a 2-D picture keeps most of the information.

**Output file:** `outputs/customer_segments.csv` has 1,175 rows with `customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name` and `is_premium`.

---

### Task 4: Defining the lapse problem
- **Cutoff date:** 1 September 2025.
- **Who is included:** every customer with at least one completed order before the cutoff (1,156 customers).
- **The label:** `lapsed = 1` if the customer made no completed order between 1 and 30 September 2025, otherwise `0`. 443 customers lapsed (38.32%).
- **Features** (made only from orders before the cutoff plus the customer table): recency, frequency, monetary, average discount, cancellation rate (cancelled orders divided by all orders), average delivery time, favourite product category, city, age, gender and tenure in days (cutoff minus signup date).
- **Split:** 80% training and 20% test with `train_test_split(test_size=0.2, stratify=y, random_state=42)`. That gives 924 training rows and 232 test rows.
- **Baseline:** if I always predict "not lapsed", I am right 61.6% of the time, but I never find a lapsed customer, so the F1 score for the lapsed class is 0. Any real model must beat this.

### Task 5: Preprocessing without leakage
- Numeric columns were standardised with `StandardScaler`.
- Text columns (city, gender, favourite category) were one-hot encoded with `drop="first"`, so each one gets (number of categories minus 1) columns. In total there are 19 columns.
- Both steps are inside one pipeline that was **fitted on the 924 training rows only**. The 232 test rows were only transformed using what was learned from training.
- The scaler's learned mean of frequency is **8.11**, while the mean of frequency over the full cohort is **8.03**. They differ because the full cohort includes the test rows, and the scaler never saw them.
- I checked that no feature column has missing values.

### Task 6: Training, tuning and evaluation
I tuned two models with `GridSearchCV` (cv=5, scoring = F1). The preprocessing sits inside the pipeline, so it is refitted on the training part of every fold.

| Model | Best settings | Best CV F1 |
|---|---|---|
| Logistic regression (C = 0.01, 0.1, 1, 10, 100) | C = 100 | 0.651 |
| Random forest (max_depth 3, 5, 8 and min_samples_leaf 5, 10, 20; 45 fits) | depth 8, leaf 10 | 0.681 |

Results on the untouched test set (lapsed class):

| Model | Precision | Recall | F1 | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|
| Baseline (always "not lapsed") | 0 | 0 | 0 | none |
| Logistic regression | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Random forest | 0.659 | 0.652 | 0.655 | [[113, 30], [31, 58]] |

**I selected the random forest.** It found the same share of lapsing customers as logistic regression (recall 0.65) with fewer false alarms (precision 0.66), and it had the best F1 (0.655). Missing a customer who is about to leave means losing all their future orders, while wrongly contacting a loyal customer only costs a small offer or message. So catching lapsing customers matters most, and the random forest gives the best balance. Both models are far above the required test F1 of 0.55.

### Task 7: Class-balance experiment
Only 354 of my 924 training customers lapsed, so the lapsed class is the smaller one. I balanced the **training data only**: I kept all 924 rows and added 216 randomly chosen copies of lapsed training rows (with replacement, `random_state=42`), so both classes have 570 rows. Then I refitted the best logistic regression on this data and tested it on the original test set.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic regression, original | 0.644 | 0.652 | 0.648 |
| Logistic regression, balanced | 0.589 | 0.854 | 0.697 |

**Recall changed the most**, from 0.65 to 0.85, while precision stayed around 0.6 (it did not go up). For the retention team this means the balanced model finds many more customers who are about to leave (76 out of 89 instead of 58), but it also flags more loyal customers by mistake (53 instead of 32). If an offer is cheap, the balanced model is the better choice.

### Task 8: Interpretation, saved model and retention list
**Top 5 features of the random forest:**
1. Frequency (0.280): customers who order often rarely leave.
2. Monetary (0.271): high spenders are more engaged.
3. Recency (0.115): a long gap since the last order raises the risk.
4. Average delivery time (0.061): slow deliveries may push customers away.
5. Cancellation rate (0.052): many cancellations point to a bad experience.

**Logistic regression coefficients:**
- Frequency is **negative** (-1.70): more orders means a lower chance of lapsing. This is the strongest effect.
- Recency is **positive** (+0.04): a longer gap means a higher chance of lapsing, but the effect is small.
- Cancellation rate is slightly **negative** (-0.12): this is very small and surprising, so I do not read much into it.

**Saved model:** I saved the random forest with `joblib.dump` to `outputs/lapse_model.joblib`. When I loaded it back with `joblib.load`, it gave exactly the same predictions on the test set.

**Retention list:** 40 customers in the test set are Premium. I ranked them by predicted chance of lapsing and saved the top 10 in `outputs/retention_list.csv` (columns `customer_id`, `lapse_probability`, `actually_lapsed`). Only 1 of these 10 really lapsed. The Premium flag comes from the whole-window RFM, which includes September (the month that defines the lapse label), so **the retention list is illustrative and not a leakage-free evaluation**.

---

### Checks against data leakage
- No encoder, scaler or resampler saw any test rows. Everything was fitted on the training rows only.
- No data from 1 September 2025 onwards was used in the features. September orders were only used to create the label.
- The segment labels from Task 3 were not used as features.

### Limitations and Next Steps
- The Premium flag is a whole-window label made after the fact, so the retention list is only illustrative and not a leakage-free evaluation.
- The test set has only 232 customers, so the scores can change a lot with a different split.
- Copying lapsed rows to balance the classes can make the model overconfident about those rows.
- Next steps: use a time-based split, choose the probability threshold using the real cost of an offer compared to losing a customer, and add more behaviour features.


## Part C: Support Ticket Triage with Guardrails

**How to run:** `python triage.py` (needs `mock_llm.py`, `data/raw/tickets.csv` and `outputs/customer_segments.csv`). Outputs: `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` and `outputs/retry_log.txt`.

### Input guardrails
Before any model call, a ticket whose text is empty or only whitespace gets `skipped_empty`, and a ticket with more than 400 estimated tokens gets `flagged_long`. Neither is sent to the model. There are 4 empty tickets and 4 long tickets (each about 734 tokens).

### Redaction
Before a ticket goes into a prompt, three regular expressions replace emails with `[EMAIL]`, phone numbers with `[PHONE]` and order ids with `[ORDER_ID]`, in that order. Across all 164 tickets: 55 emails, 91 phones, 132 order ids, **278 in total**, and re-applying the patterns finds 0 left. The user message is exactly the redacted text, and an assertion in the code checks that no prompt contains an email, phone or order id.

### Prompts
The zero-shot prompt contains the word CLASSIFY, the four categories and the JSON-only reply format. The few-shot prompt adds an `EXAMPLES:` section with four labelled examples (billing, delivery, technical and a fraud example).

### Retries
Every model call is wrapped in a retry function: at most 3 retries, with a wait of 0.05 x 2^n x a random factor between 0.9 and 1.15 seconds. Each retry writes one log line (ticket id, retry number, wait). A full run has **19 transient errors, all recovered**, and every wait is between 0.045 and 0.0575 seconds.

### Output validation
A reply that is valid JSON with an allowed category is `classified_json`. A reply that is not JSON is split into lowercase words with `re.findall`, and if exactly one allowed category word is present it is `classified_keyword`. Everything else, including JSON with the category "shipping", is `flagged_invalid_output` and goes to human review. We never guess. One full run gives **135 classified_json, 12 classified_keyword and 9 flagged_invalid_output**.

### Zero-shot vs few-shot
| | Zero-shot | Few-shot |
|---|---|---|
| Tickets classified | 147 | 147 |
| Accuracy | 85.7% (126 of 147) | 100% (147 of 147) |
| Total tokens | 12,052 | 24,220 |
| Tokens per classified ticket | 82.0 | 164.8 |

The zero-shot errors are all 11 fraud tickets predicted as billing and 10 technical tickets predicted as billing. (Add your own 3 to 4 sentences, for example: the few-shot prompt costs about 83 extra tokens per ticket, roughly double, but it removes all 21 mistakes. It is worth paying when a wrong label is expensive, such as a fraud ticket sitting in the billing queue instead of reaching the security team, and not worth it for low-stakes tickets or a tight token budget.)

### Token budget and routing
**Budget demo:** with a zero-shot run and TOKEN_BUDGET = 6000, the run stopped calling the model once the running total reached the budget, and **78 tickets ended as `budget_exhausted`** (see `outputs/triage_budget_demo.csv`).

**Final run:** a few-shot run with TOKEN_BUDGET = 100000 produced `outputs/triage_results.csv` (164 rows). Routing rules: skipped, long, invalid or budget-exhausted tickets go to `human_review` (NORMAL); fraud goes to `security_team` (HIGH); Premium customers (from Part B) go to `premium_desk` (HIGH); all others go to `support_queue` (NORMAL). A customer with no row in the segment file is treated as not premium. Result: support_queue 102, security_team 34, human_review 17, premium_desk 11.

### What the rules validated and what went to humans
147 tickets were classified and checked by rules (135 valid JSON, 12 by keyword extraction). 17 tickets went to humans: 9 invalid model outputs, 4 empty tickets and 4 tickets that were too long.

## Part D: Policy RAG Assistant and Refund Agent

**How to run:** `python rag.py` (chunking, retrieval and evaluation) and `python agent.py` (tools, agent runs, limits and crash-and-resume).

### Chunking and retrieval
Each of the five policy files is split into chunks of 40 words with an overlap of 10 words (a new chunk starts every 30 words). There are **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2. A `TfidfVectorizer` is fitted on the chunk texts, a query is transformed with the same vectorizer, and chunks are ranked by cosine similarity. For the query "refund policy late delivery" the top three chunks all come from `refund_policy.txt`.

**TF and IDF:** term frequency (TF) measures how often a word appears in a chunk, and inverse document frequency (IDF) measures how rare the word is across all chunks, so rare words count more. **Limitation:** TF-IDF matches words, not meaning, so two phrasings with the same meaning but different words (for example "money back" and "refund") may not match.

### Evaluation
| Run | Answerable: expected_doc in top 3 | Answerable: answer contains expected_fact | Unanswerable: fallback answer |
|---|---|---|---|
| Overlap 10 | 12/12 | 12/12 | 2/2 |
| Overlap 0 | 12/12 | 10/12 | 2/2 |

With zero overlap, two questions fail: "What delivery time makes a completed order eligible for a full refund?" (the fact "more than 45 minutes" is split across a chunk boundary) and "How soon is money returned when a payment fails but is deducted?" (the fact "48 hours" is split the same way). The overlap keeps these sentences together.

### Tools
The agent has five tools in `TOOLS_SPEC` and `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (uses the same retrieval), `calculate_refund` (100% of the order amount if delivery took more than 45 minutes, otherwise 0), and `issue_refund` and `escalate_to_human` (each appends one JSON line to `outputs/refund_ledger.jsonl`). A refund above INR 1000 is escalated, not issued.

### Agent results (ReAct loop)
| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,132 |
| S2 | REFUND | 135.00 | 5 | 1,132 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,119 |
| S4 | DENIED | 0.00 | 4 | 793 |
| S5 | DENIED | 0.00 | 3 | 478 |
| S6 | NOT_FOUND | 0.00 | 2 | 166 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,282 |

(Copy the numbers from your own `python agent.py` output if they differ.)

### Stopping safely
- Max iterations: S7 loops on `search_policy` and stops after 6 model calls with `FALLBACK_MAX_ITERATIONS`.
- Token limit: S1 with `token_limit=200` ended with **FALLBACK_TOKEN_LIMIT** on the third model call (495 tokens), before the requested tool ran.
- Timeout: S1 with `time_limit_seconds=0` ended with **FALLBACK_TIMEOUT** before the first model call.
- Every fallback returns the steps completed so far and a short human-handoff message.

### Crash and resume
After every tool result a checkpoint is saved in `state/`. I crashed S1 right after its second tool and called `resume_agent`. The tools executed across the crash and the resume, in order, were: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated). The final status was REFUND with INR 128.25.

### ReAct versus plan-and-execute
 My ReAct agent asks the model after every tool result. It used 5 model calls for a refund (S1, 1,132 tokens), 4 for a denied order (S4), 3 for a cancelled order (S5), 2 for a missing order (S6, 166 tokens) and 6 for the looping S7 (3,282 tokens). A plan-and-execute design asks the model once for the whole list of steps and once for the final answer, so a normal refund would need about 2 model calls and fewer tokens, because the growing message history is not sent again after every tool call. The price is rigidity: if the order is not found (S6) or is cancelled (S5), a fixed plan keeps running steps that make no sense until the code notices the error and asks for a new plan, which costs extra calls. ReAct reads each observation and stops at once: S6 ended after 2 calls and S5 after 3. For this refund workflow I would choose ReAct, because money is involved and each decision depends on the previous result (order status, delivery time, amount above INR 1000), so a wrong automatic refund costs more than a few extra model calls. A hybrid that plans once and replans on any error would be a good way to cut cost later.

## Limitations and Next Steps
- **Part A:** the cleaning rules are fixed rules for this dataset. A different export with new kinds of errors would need the audit and the rules to be reviewed again.
- **Part B:** the Premium flag comes from the whole-window RFM of Task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label and the retention list is illustrative and not a leakage-free evaluation. The test set has only 232 customers, so scores can change a lot with a different split, and over-sampling by duplication can make the model over-confident about repeated rows.
- **Part C:** the language model is a mock that classifies by keywords. Redaction covers only emails, phone numbers and order ids, so names and addresses in free text are not removed. Token counts are estimates.
- **Part D:** TF-IDF matches words and not meaning, so a question phrased with different words can miss the right chunk. The five policy documents are tiny (15 chunks), and the mock model answers with one matching sentence. The refund rules are simplified and all requests are treated as inside the 7-day window.
- **Next steps:** a time-based split and a probability threshold chosen from real offer costs (Part B), a real model with a held-out ticket set (Part C), embedding-based retrieval and a plan-and-replan agent (Part D).
