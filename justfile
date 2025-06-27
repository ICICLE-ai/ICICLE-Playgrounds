#!/usr/bin/env just --justfile

# Used to export PyPI token for publishing the package.
set dotenv-load := true

export PATH := join(justfile_directory(), ".venv", "bin") + ":" + env_var('PATH')

upgrade:
  uv lock --upgrade

publish:
    rm ./dist/* && uv build && uv publish

