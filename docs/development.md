# Development Guildline
## Setup
```
python3 -m venv env
source  env/bin/activate
pip install colored polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com
python3 -m pip install -e . --extra-index-url https://pypi.ngc.nvidia.com
```

## Before commit
```
black .
isort test src
autoflake --in-place --remove-unused-variables --recursive src test
```

## Test
run
`pytest`