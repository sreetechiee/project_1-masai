import glob
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime

import pandas as pd

import mock_llm
from rag import retrieve

DB_PATH = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
RUNS_LOG = "outputs/agent_runs.jsonl"
STATE_DIR = "state"

AGENT_SYSTEM = ("You are a refund AGENT for a quick-commerce company. Use the tools to look up the "
                "order, check the policy, calculate the refund and then issue it or escalate it to a "
                "human. Finish with a short answer that starts with REFUND, ESCALATE, DENIED or NOT_FOUND.")


# ---------- Task 5: tools ----------
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                      "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    results = retrieve(query)
    return {"results": [{"source": r["source"], "score": round(r["score"], 4), "text": r["text"]}
                        for r in results]}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}      # 100% of the order amount
    return {"refund_inr": 0.0}                                    # delivered within 45 minutes


def _ledger(tool_name, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                            "tool": tool_name, "order_id": order_id, "refund_inr": refund_inr}) + "\n")


def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the refund in INR from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund (INR 1000 or less) and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                      "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                      "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


# ---------- Helpers: checkpoint, logging, fallback ----------
class SimulatedCrash(Exception):
    pass


def save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w") as f:
        json.dump(state, f, indent=2)


def log_run(result, demo=None):
    os.makedirs("outputs", exist_ok=True)
    line = {"scenario_id": result["scenario_id"], "status": result["status"],
            "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"], "steps": result["steps"]}
    if demo:
        line["demo"] = demo
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps(line) + "\n")


def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]}


def _fallback(state, status, reason):
    tools = ", ".join(s["tool"] for s in state["steps"]) or "none"
    summary = (f"HUMAN HANDOFF ({status}): {reason}. Completed {len(state['steps'])} tool step(s): {tools}. "
               f"Please review the request manually.")
    return _result(state, status, summary)


# ---------- Task 6 and 7: the ReAct loop with safety limits ----------
def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:                       # (c) timeout
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached")
        if state["llm_calls"] >= max_iterations:                                  # (a) max iterations
            return _fallback(state, "FALLBACK_MAX_ITERATIONS",
                             f"no final answer after {max_iterations} model calls")
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        if state["tokens_used"] > token_limit:                                    # (b) token limit
            return _fallback(state, "FALLBACK_TOKEN_LIMIT",
                             f"{state['tokens_used']} tokens used, limit {token_limit}")
        msg = response.choices[0].message
        if not msg.tool_calls:                                                    # final answer
            answer = msg.content
            return _result(state, answer.split(":")[0].strip(), answer)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        observation = TOOL_MAP[tc.name](**tc.arguments)                           # look up and call
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(observation)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        state["tools_executed"].append(tc.name)
        state["step"] = len(state["steps"])
        save_checkpoint(state)                                                    # Task 8
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool {crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, crash_after_tools=None, demo=None):
    state = {"run_id": f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "tools_executed": [], "steps": [], "llm_calls": 0}
    state["_limits"] = {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds, "demo": demo}
    result = _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)
    log_run(result, demo)
    return result


def resume_agent(run_id, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json")) as f:
        state = json.load(f)
    lim = state["_limits"]
    result = _loop(state, lim["max_iterations"], lim["token_limit"], time_limit_seconds)
    log_run(result, "crash_and_resume")
    return result


def refund_amount(result):
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0


if __name__ == "__main__":
    for p in [LEDGER, RUNS_LOG] + glob.glob(os.path.join(STATE_DIR, "*.json")):
        if os.path.exists(p):
            os.remove(p)                                   # start clean so the run is reproducible

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    rows = []
    for _, s in scenarios.iterrows():
        res = run_agent(s["message"], s["scenario_id"])
        amt = refund_amount(res)
        ok = (res["status"].startswith(s["expected_outcome"])
              and abs(amt - s["expected_refund_inr"]) <= 0.01)
        rows.append({"scenario": s["scenario_id"], "status": res["status"], "refund_inr": amt,
                     "expected": s["expected_refund_inr"], "model_calls": res["llm_calls"],
                     "tokens": res["tokens_used"], "ok": ok})
    print(pd.DataFrame(rows).to_string(index=False))

    s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]
    r = run_agent(s1["message"], "S1", token_limit=200, demo="token_limit_200")
    print("\nS1 with token_limit=200 ->", r["status"], "| llm_calls:", r["llm_calls"],
          "| tokens:", r["tokens_used"], "| steps:", len(r["steps"]))
    print("  ", r["final_answer"])
    r = run_agent(s1["message"], "S1", time_limit_seconds=0, demo="time_limit_0")
    print("S1 with time_limit_seconds=0 ->", r["status"], "| llm_calls:", r["llm_calls"])
    print("  ", r["final_answer"])

    # crash after the second tool execution, then resume
    crashed_run_id = None
    try:
        run_agent(s1["message"], "S1", crash_after_tools=2, demo="crash_and_resume")
    except SimulatedCrash as e:
        crashed_run_id = sorted(glob.glob(os.path.join(STATE_DIR, "S1-*.json")),
                               key=os.path.getmtime)[-1].split(os.sep)[-1][:-5]
        print("\nCRASH:", e, "| run id:", crashed_run_id)
    r = resume_agent(crashed_run_id)
    print("RESUMED ->", r["status"], "| refund:", refund_amount(r))
    with open(os.path.join(STATE_DIR, f"{crashed_run_id}.json")) as f:
        print("tools executed across crash and resume:", json.load(f)["tools_executed"])