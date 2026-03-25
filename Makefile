.PHONY: install test lint format docker-build run

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/

lint:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	python -m black .
	python -m isort .

docker-build:
	docker build -t placeholders-backend .

run:
	uvicorn app.main:app --reload
