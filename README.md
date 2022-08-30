# My Book

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mkihara.github.io/mybook/)

## Install requirements

```sh
pip install --upgrade pip
pip install --upgrade -r requirements.txt
julia packages.jl
```

## Build

```sh
jb build docs
```

## Publish

```sh
ghp-import -n -p -f docs/_build/html
```
