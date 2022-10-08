
.PHONY: style test

check_dirs := lekin examples tests

# run checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
	flake8

# run tests for the library

test:
	python -m unittest

# release

