.PHONY: sanity run eval

sanity:
	@mkdir -p artifacts
	@python scripts/run_sanity.py

run:
	@uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

eval:
	@mkdir -p artifacts
	@python scripts/run_eval.py