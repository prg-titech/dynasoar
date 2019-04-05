#!/bin/sh
set -e

mkdir -p bin/

OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Helper variables
args=""
optimizations="-O3 -DNDEBUG"
allocator="dynasoar"

while getopts "h?da:s:" opt; do
    case "$opt" in
    h|\?)
        echo "Optional arguments:"
        echo "  -a ALLOC    Choose allocator. Possible values:"
        echo "              bitmap, cuda, dynasoar (default), halloc, mallocmc"
        echo "  -d          Debug mode"
        echo "  -s          Heap size"
        echo "              dynasoar: #objects of the smallest type"
        echo "              other allocators: heap size in bytes"
        echo ""
        echo "Example: ${0} -a mallocmc"
        exit 0
        ;;
    s)  args="${args} -DPARAM_HEAP_SIZE=${OPTARG}"
        ;;
    d)  optimizations="-g -O3"
        ;;
    a)  allocator="${OPTARG}"
        ;;
    esac
done

shift $((OPTIND-1))
[ "${1:-}" = "--" ] && shift


args="${args} ${optimizations} -std=c++11 -lineinfo --expt-extended-lambda"
args="${args} -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61"
args="${args} -maxrregcount=64 -Iexample/configuration/${allocator} -I. -Ilib/cub"

if [ "$allocator" = "mallocmc" ]; then
  args="${args} -Iexample/configuration/mallocmc/mallocMC"
fi;

if [ "$allocator" = "halloc" ]; then
  args="${args} -Iexample/configuration/halloc/halloc/src"
fi;


build_scripts/nvcc.sh ${args} example/api-example/example.cu -o bin/api_example
