# Common tasks: make install | test | lint | app | experiments | quick | assets
.PHONY: install test lint app experiments quick assets

install:
	pip install -r requirements.txt pytest ruff

test:
	python -m pytest -q

lint:
	ruff check .

app:
	streamlit run app/Home.py

experiments:
	python scripts/run_experiments.py

quick:
	python scripts/run_experiments.py --quick

assets:
	python scripts/make_assets.py
