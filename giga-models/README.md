# Giga Models

## Introduction

[Giga Models](https://codeup.aliyun.com/645a08b9d983fb47ec1d5df2/algorithm/giga_codes/giga-models)

## Installation

```shell
git clone git@codeup.aliyun.com/645a08b9d983fb47ec1d5df2/algorithm/giga_codes/giga-models.git
cd giga-models
pip3 install -r requirements.txt
```

## Quickstart

You can learn more by [Wiki](https://gigaai0118.feishu.cn/wiki/Hz0bwfE6oidZ3wkIusTc0aEjnpd)

## Code style

We use the following tools for linting and formatting:

- [isort](https://github.com/PyCQA/isort): A Python utility to sort imports.
- [black](https://github.com/psf/black): A formatter for Python files.
- [flake8](https://github.com/PyCQA/flake8): A wrapper around some linter tools.
- [codespell](https://github.com/codespell-project/codespell): A Python utility to fix common misspellings in text files.
- [mdformat](https://github.com/executablebooks/mdformat): Mdformat is an opinionated Markdown formatter that can be used to enforce a consistent style in Markdown files.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of isort, black, flake8 and codespell can be found in [setup.cfg](./setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip3 install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```

Before you create a PR, make sure that your code lints and is formatted by black.

```shell
pre-commit run --all-files
```

## License

Giga Inference is released under the [Apache 2.0 license](LICENSE).
