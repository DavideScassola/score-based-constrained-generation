.ONESHELL:
SHELL := /bin/bash
EXPERIMENTS := artifacts
CONFIGS_FOLDER := config

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  Install:"
	@echo "  - venv                      : set up the virtual environment for development"
	@echo "  - install                   : installs requirements"

# Installation

venv:
	python3.10 -m venv .venv

.PHONY: install
install:
	pip install --upgrade pip
	python -m pip install --upgrade torch && \
	python -m pip install --upgrade -r requirements.txt


.PHONY: format
format:
	black .
	isort .


.PHONY: test
test:
	sh scripts/mode_selection_test.sh


.PHONY: clear_experiments
clear_experiments:
	rm -r $(EXPERIMENTS)


.PHONY: clean_workspace
clean_workspace:
	mkdir -p .closet
	mv --backup=t *.png .closet/
