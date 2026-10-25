"""
eval_rag.py — faithfulness check for Document Q&A on a held-out policy
document (eval_data/heldout/policy_doc.txt).

For answerable questions: the answer must contain the expected fact AND
cite at least one passage, and the cited fact must actually appear in a
returned source passage (so the answer is grounded, not recalled).
For unanswerable questions: the answer must say the information is not
in the document instead of inventing one.

Also reports RETRIEVAL RANK (separately from correctness): how often the
passage that contains the answer is the FIRST one retrieved. The app
retrieves the top 5 passages, so on a short document every passage is
returned anyway; rank is what shows whether retrieval is actually picking
the right part of the document. The test document deliberately contains
near-miss figures (domestic vs international meal allowance, metro vs
tier-2 hotel caps) so that ranking is not trivial.

Text is compared after Unicode normalisation, so a model writing
"8\u202fhours" (narrow no-break space) or "₹" variants is not marked
wrong for formatting.

Usage: python eval_rag.py   (needs GROQ_API_KEY; indexes locally)
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
HELDOUT = os.path.join(HERE, "eval_data", "heldout")
def norm(text: str) -> str:
    """Unicode-normalise and collapse every kind of space, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s\u00a0\u202f\u2009\u2007]+", " ", text)
    return text.lower().strip()


NOT_FOUND = re.compile(r"not\s+(found|mentioned|covered|stated|included|available)|does\s+not\s+(say|mention|contain|specify)", re.I)


def main() -> int:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 2
    from eval_harness import wilson_interval
    from rag_engine import rag_answer_with_sources
    from rag_service import index_document

    with open(os.path.join(HELDOUT, "policy_doc.txt"), "rb") as f:
        meta = index_document(f.read(), "policy_doc.txt", session_id="eval-rag")
    if not meta.get("success"):
        print(f"indexing failed: {meta.get('error')}", file=sys.stderr)
        return 1
    with open(os.path.join(HELDOUT, "policy_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for q in questions:
        out = rag_answer_with_sources(meta["doc_id"], q["question"], "policy_doc.txt")
        ans, sources = out["answer"], out["sources"]
        n_ans = norm(ans)
        source_text = norm(" ".join(s["text"] for s in sources))
        top1 = None
        if q["answerable"]:
            expected = [norm(e) for e in q["expect_all"]]
            has_fact = all(e in n_ans for e in expected)
            grounded = all(e in source_text for e in expected)
            cites = bool(re.search(r"passage\s*\d", n_ans))
            top1 = bool(sources) and all(e in norm(sources[0]["text"]) for e in expected)
            ok = has_fact and grounded and cites
            why = "" if ok else f"fact={has_fact} grounded={grounded} cites={cites}"
        else:
            ok = bool(NOT_FOUND.search(n_ans))
            why = "" if ok else "answered an unanswerable question"
        rows.append({"id": q["id"], "correct": ok, "why": why, "retrievedFirst": top1, "answer": ans})
        print(f"[{'OK  ' if ok else 'FAIL'}] {q['id']} {q['question'][:55]} {why}", flush=True)

    k, n = sum(r["correct"] for r in rows), len(rows)
    lo, hi = wilson_interval(k, n)
    ranked = [r["retrievedFirst"] for r in rows if r["retrievedFirst"] is not None]
    top1_rate = round(sum(ranked) / len(ranked), 3) if ranked else None
    summary = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "faithfulness": round(k / n, 3), "accuracy95": [round(lo, 3), round(hi, 3)],
               "retrievalTop1": top1_rate, "results": rows}
    out_dir = os.path.join(HERE, "eval_results", "rag")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRAG faithfulness {k}/{n} (95% interval {lo * 100:.0f}–{hi * 100:.0f}%)")
    if ranked:
        print(f"Retrieval: answer passage ranked first in {sum(ranked)}/{len(ranked)} answerable questions")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
