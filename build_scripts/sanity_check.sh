#!/bin/sh
set -e

build_scripts/build_api_example.sh -a dynasoar
echo "Built API example."

bin/api_example
echo "Ran API example. Done."
