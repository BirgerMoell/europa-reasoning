.PHONY: install test check validate packs suite site

install:
	python3 -m pip install -e '.[dev]'

test:
	python3 -m unittest discover -s tests -v

check:
	python3 -m ruff check src tests
	python3 -m unittest discover -s tests -v

validate:
	python3 -m europa_reasoning validate

packs:
	python3 scripts/audit_language_packs.py

suite:
	python3 scripts/build_core_suite.py

site:
	python3 -m europa_reasoning build-site
