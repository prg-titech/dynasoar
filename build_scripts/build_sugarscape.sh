#!/bin/sh
set -e

mkdir -p bin/

OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Helper variables
args=""
optimizations="-O3 -DNDEBUG"
allocator="dynasoar"

while getopts "h?x:y:rda:s:m:" opt; do
    case "$opt" in
    h|\?)
        echo "Optional arguments:"
        echo "  -a ALLOC    Choose allocator. Possible values:"
        echo "              bitmap, cuda, dynasoar (default), halloc, mallocmc"
        echo "  -d          Debug mode"
        echo "  -m NUM      Max. #objects (of the smallest type)"
        echo "  -r          Render visualization"
        echo "  -s BYTES    Heap size (for allocators different from dynasoar)"
        echo "  -x SIZE     Size X/Y (#pixels)"
        echo ""
        echo "Example: ${0} -x 400"
        exit 0
        ;;
    x)  args="${args} -DPARAM_SIZE=${OPTARG}"
        ;;
    r)  args="${args} -DOPTION_RENDER -lSDL2 example/sugarscape/rendering.cu"
        ;;
    s)  args="${args} -DPARAM_HEAP_SIZE=${OPTARG}"
        ;;
    m)  args="${args} -DPARAM_MAX_OBJ=${OPTARG}"
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


build_scripts/nvcc.sh ${args} example/sugarscape/dynasoar_no_cell/sugarscape.cu -o bin/sugarscape_dynasoar_no_cell
build_scripts/nvcc.sh ${args} example/sugarscape/dynasoar/sugarscape.cu -o bin/sugarscape_dynasoar
build_scripts/nvcc.sh ${args} example/sugarscape/baseline_aos/sugarscape.cu -o bin/sugarscape_baseline_aos
build_scripts/nvcc.sh ${args} example/sugarscape/baseline_soa/sugarscape.cu -o bin/sugarscape_baseline_soa
