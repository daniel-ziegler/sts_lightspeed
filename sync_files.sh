#!/bin/bash
cd "$(dirname "$0")"
rsync . ubuntu@lambda:sts2 -av --exclude='*.pt*' --exclude='*.parquet' --exclude='.git' --exclude-from=.gitignore --exclude=CMakeFiles --exclude=playground
