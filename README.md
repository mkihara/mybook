# My Book

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

数分後に[ここ](https://mkihara.github.io/mybook/)で公開されたサイトを確認できる。
