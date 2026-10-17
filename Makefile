.PHONY: help lint format docs drills

help:
	@echo "make lint     check formatting, imports and style"
	@echo "make format   auto-format with isort and black"
	@echo "make docs     build the standards PDFs into docs/standards/pdf"
	@echo "make drills   run all debugging drills"

lint:
	isort --check-only --diff .
	black --check --diff .
	flake8 .
	mypy drills

format:
	isort .
	black .

docs:
	mkdir -p docs/standards/pdf
	cd docs/standards && for f in *.tex; do \
		latexmk -pdf -interaction=nonstopmode -halt-on-error \
			-outdir=pdf "$$f" || exit 1; \
	done
	cd docs/standards && latexmk -c -outdir=pdf

drills:
	python drills/pytorch_debug_drills.py
