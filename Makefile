.DEFAULT_GOAL := build
.PHONY: build lint test docs package deploy
PROJ_SLUG = xyzcad
SHELL = bash

build:
	poetry version $(shell git describe --tags --abbrev=0)
	poetry build

lint:
	poetry run black --check .
	poetry run isort --check .

test: clean
	mkdir -p build
	poetry run pytest tests/

docs:
	poetry run sphinx-apidoc -f -o ./docs/source/ ./$(PROJ_SLUG)
	cd docs && poetry run make html

package: clean docs
	poetry build

publish:
	poetry publish

clean:
	poetry version 0.0.0+devel
	rm -rf .pytest_cache \
	rm -rf dist \
	rm -rf build \
	rm -rf __pycache__
	#rm -rf docs/build \
	rm -rf *.egg-info \
	rm -rf docs/source/modules \
	rm -rf htmlcov \
	rm -rf meta\
	rm -rf output

reformat:
	poetry run isort .
	poetry run black .
