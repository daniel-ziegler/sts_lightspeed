#!/bin/bash
cd "$(dirname "$0")"
rsync . ubuntu@lambda:sts -av --exclude='*.pt*' --exclude='*.parquet' --exclude='.git' --exclude-from=.gitignore --exclude=CMakeFiles --exclude=playground --exclude=__pycache__
