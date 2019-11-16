#!/bin/sh
set -e
nvcc -Xcudafe "--diag_suppress=1427" "$@"

# Change host compiler with --ccbin
# E.g.: -ccbin clang-3.8 -lstdc++
# GCC is highly recommended. When using clang, some source code modifications
# are required. See example/api-example/example.cu.
