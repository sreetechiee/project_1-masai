import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

# ---------- Settings ----------
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]

EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE = r"\b[6-9]\d{9}\b"
ORDER = r"ORD-\d{6}"
PATTERNS = [(EMAIL, "[EMAIL]", "email"), (PHONE, "[PHONE]", "phone"), (ORDER, "[ORDER_ID]", "order_id")]

ZERO_SHOT = (
    "You are a support ticket router. CLASSIFY the customer ticket into exactly one "
    "category: billing, delivery, technical or fraud. "
    'Reply with JSON only, in the form {"category": "<category>", "reason": "<short reason>"}. '
    "Do not add any other text."
)
FEW_SHOT = ZERO_SHOT + (
    "\nEXAMPLES:\n"
    'Ticket: Someone made an unauthorized payment from my account -> {"category": "fraud", "reason": "unauthorized payment"}\n'
    'Ticket: I was charged twice on my invoice -> {"category": "billing", "reason": "double charge"}\n'
    'Ticket: The app crashes when I try to login -> {"category": "technical", "reason": "app crash"}\n'
    'Ticket: The rider never arrived with my order -> {"category": "delivery", "reason": "order not delivered"}'
)


# ---------- Redaction ----------
def redact(text):
    counts = {}
    for pattern, placeholder, name in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        counts[name] = n
    return text, counts


# ---------- Retry with exponential backoff and jitter ----------
def call_with_retry(messages, ticket_id, log):
    retries, errors = 0, 0
    while True:
        try:
            return mock_llm.chat(messages), errors
        except TransientLLMError:
            errors += 1
            if retries >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retries * random.uniform(0.9, 1.15)
            line = f"RETRY ticket={ticket_id} retry_number={retries} wait={wait:.4f}s"
            log.append(line)
            print(line)
            time.sleep(wait)
            retries += 1


# ---------- Output validation ----------
def validate(text):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        found = {w for w in words if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""
    if isinstance(obj, dict) and obj.get("category") in ALLOWED:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""


# ---------- One full run over the tickets ----------
def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()                       # immediately before each full run
    log, rows = [], []
    running_tokens, transient_errors = 0, 0
    for _, t in tickets.iterrows():
        text = t["text"]
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"],
               "true_category": t["true_category"], "category": "", "tokens_used": 0}
        if text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and running_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            redacted, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redacted}]
            assert messages[1]["content"] == redacted
            for m in messages:             # guardrail check: no PII in any prompt
                assert not any(re.search(p, m["content"]) for p, _, _ in PATTERNS)
            response, errs = call_with_retry(messages, t["ticket_id"], log)
            transient_errors += errs
            used = response.usage.total_tokens
            running_tokens += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(response.choices[0].message.content)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["log"] = log
    result.attrs["transient_errors"] = transient_errors
    result.attrs["total_tokens"] = running_tokens
    return result


def summarise(result, label):
    classified = result[result["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    total_tokens = result.attrs["total_tokens"]
    print(f"\n=== {label} ===")
    print(result["status"].value_counts().to_string())
    print("transient errors:", result.attrs["transient_errors"], "| retry lines:", len(result.attrs["log"]))
    print(f"classified: {len(classified)} | accuracy: {correct}/{len(classified)} = "
          f"{correct / len(classified) * 100:.1f}%")
    print(f"total tokens: {total_tokens} | tokens per classified ticket: {total_tokens / len(classified):.1f}")
    return classified


if __name__ == "__main__":
    random.seed(42)
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("tickets:", len(tickets))

    # ---------- Task 1 and 2: guardrails and redaction (counts over all 164 tickets) ----------
    empty = (tickets["text"].str.strip() == "").sum()
    long_ = (tickets["text"].map(lambda s: est_tokens(s) > MAX_INPUT_TOKENS)).sum()
    print("empty tickets:", empty, "| long tickets:", long_,
          "| long estimate:", sorted({est_tokens(s) for s in tickets["text"] if est_tokens(s) > MAX_INPUT_TOKENS}))
    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted_all = []
    for s in tickets["text"]:
        r, c = redact(s)
        redacted_all.append(r)
        for k in totals:
            totals[k] += c[k]
    print("substitutions:", totals, "| total:", sum(totals.values()))
    left = sum(len(re.findall(p, r)) for r in redacted_all for p, _, _ in PATTERNS)
    print("PII left after redaction:", left)
    assert left == 0

    # ---------- Task 6: zero-shot and few-shot runs (no budget) ----------
    zero = run_pipeline(tickets, ZERO_SHOT, None, "zero-shot")
    zc = summarise(zero, "ZERO-SHOT")
    print("\nconfusion table (zero-shot, classified tickets):")
    print(pd.crosstab(zc["true_category"], zc["category"]))

    few = run_pipeline(tickets, FEW_SHOT, None, "few-shot")
    summarise(few, "FEW-SHOT")

    # ---------- Task 7a: budget demo ----------
    BUDGET_DEMO = 6000
    demo = run_pipeline(tickets, ZERO_SHOT, BUDGET_DEMO, "budget demo")
    summarise(demo, f"BUDGET DEMO (TOKEN_BUDGET={BUDGET_DEMO})")
    print("budget_exhausted:", int((demo["status"] == "budget_exhausted").sum()))

    # ---------- Task 7b: routing (reads outputs/customer_segments.csv from Part B) ----------
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")

    def add_routing(result):
        HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}
        names, priorities, routes = [], [], []
        for _, r in result.iterrows():
            in_seg = r["customer_id"] in seg.index
            names.append(seg.loc[r["customer_id"], "segment_name"] if in_seg else "")
            premium = bool(seg.loc[r["customer_id"], "is_premium"]) if in_seg else False
            if r["status"] in HUMAN:
                routes.append("human_review"); priorities.append("NORMAL")
            elif r["category"] == "fraud":
                routes.append("security_team"); priorities.append("HIGH")
            elif premium:
                routes.append("premium_desk"); priorities.append("HIGH")
            else:
                routes.append("support_queue"); priorities.append("NORMAL")
        out = result.copy()
        out["segment_name"], out["priority"], out["route"] = names, priorities, routes
        return out[["ticket_id", "customer_id", "status", "category", "segment_name",
                    "priority", "route", "tokens_used"]]

    os.makedirs("outputs", exist_ok=True)
    add_routing(demo).to_csv("outputs/triage_budget_demo.csv", index=False)

    final = run_pipeline(tickets, FEW_SHOT, 100000, "final")
    summarise(final, "FINAL (few-shot, TOKEN_BUDGET=100000)")
    final_out = add_routing(final)
    final_out.to_csv("outputs/triage_results.csv", index=False)
    print("\nrows in triage_results.csv:", len(final_out))
    print(final_out["route"].value_counts().to_string())
    print(pd.crosstab(final_out["route"], final_out["priority"]))

    waits = [float(re.search(r"wait=([\d.]+)s", l).group(1)) for l in final.attrs["log"]]
    print("retry waits min/max:", min(waits), max(waits), "| all in 0.045-0.0575:",
          all(0.045 <= w <= 0.0575 for w in waits))
    with open("outputs/retry_log.txt", "w") as f:
        f.write("\n".join(final.attrs["log"]))