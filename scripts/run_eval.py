#!/usr/bin/env python3
"""
Simple evaluation harness: run test questions, collect answers and citations, write report.
Usage: from repo root, OPENAI_API_KEY set:
  python scripts/run_eval.py
  python scripts/run_eval.py scripts/eval_questions.json
Produces: artifacts/eval_report.json (and optionally artifacts/eval_report.md).
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from app.config import ARTIFACTS_DIR, SAMPLE_DOCS_DIR
from app.pipeline import ingest_directory, query
from app.retrieval import VectorStore


def load_questions(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("questions", [])


def run_eval(questions_path: Path) -> list:
    store = VectorStore(reset=False)
    if store.collection.count() == 0 and SAMPLE_DOCS_DIR.exists():
        ingest_directory(store, SAMPLE_DOCS_DIR)

    questions = load_questions(questions_path)
    results = []
    for q in questions:
        qid = q.get("id", "")
        question = q.get("question", "")
        expect_refusal = q.get("expect_refusal", False)
        expect_citations = q.get("expect_citations", True)
        answer, citations = query(store, question)
        got_refusal = (
            "couldn't find" in answer.lower() or "cannot find" in answer.lower() or "not in the uploaded" in answer.lower()
        )
        got_citations = bool(citations and len(citations) > 0)
        if expect_refusal:
            passed = got_refusal and not got_citations  # refuse with no fake citations
        else:
            passed = not got_refusal and (not expect_citations or got_citations)
        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "citations": citations,
            "expected_behavior": q.get("expect", ""),
            "expect_refusal": expect_refusal,
            "expect_citations": expect_citations,
            "got_refusal": got_refusal,
            "got_citations": got_citations,
            "pass": passed,
        })
    return results


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run evaluation.", file=sys.stderr)
        sys.exit(1)

    questions_path = REPO_ROOT / "scripts" / "eval_questions.json"
    if len(sys.argv) > 1:
        questions_path = Path(sys.argv[1])
    if not questions_path.exists():
        print(f"Questions file not found: {questions_path}", file=sys.stderr)
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_eval(questions_path)

    report = {"questions": results, "summary": {"total": len(results), "passed": sum(1 for r in results if r.get("pass"))}}
    out_json = ARTIFACTS_DIR / "eval_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}")

    out_md = ARTIFACTS_DIR / "eval_report.md"
    lines = ["# Evaluation Report\n", f"Total: {report['summary']['total']}, Passed: {report['summary']['passed']}\n"]
    for r in results:
        lines.append(f"## {r['id']} {'PASS' if r.get('pass') else 'FAIL'}\n")
        lines.append(f"**Q:** {r['question']}\n")
        lines.append(f"**A:** {r['answer'][:500]}{'...' if len(r['answer']) > 500 else ''}\n")
        lines.append(f"**Citations:** {len(r['citations'])}\n")
        if r["citations"]:
            for c in r["citations"][:3]:
                lines.append(f"- {c.get('source')} ({c.get('locator')})\n")
        lines.append("")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
