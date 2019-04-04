#!/bin/sh

OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Helper variables
args=""
render=0
optimizations="-O3 -DNDEBUG"

while getopts "h?x:y:rd" opt; do
    case "$opt" in
    h|\?)
        echo "Optional arguments:"
        echo "  -d          Debug mode"
        echo "  -r          Render visualization"
        echo "  -x SIZE_X   Size X (#pixels)"
        echo "  -y SIZE_Y   Size Y (#pixels)"
        echo ""
        echo "Example: ${0} -x 512 -y 512"
        exit 0
        ;;
    x)  args="${args} -DPARAM_SIZE_X=${OPTARG}"
        ;;
    y)  args="${args} -DPARAM_SIZE_Y=${OPTARG}"
        ;;
    r)  args="${args} -DOPTION_RENDER -lSDL2 example/wa-tor/rendering.cu"
        render=1
        ;;
    d)  optimizations="-g -O3"
        ;;
    esac
done

shift $((OPTIND-1))
[ "${1:-}" = "--" ] && shift


args="${args} ${optimizations} -std=c++11 -lineinfo --expt-extended-lambda"
args="${args} -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61"
args="${args} -maxrregcount=64 -Iexample/configuration/dynasoar -I. -Ilib/cub"

build_scripts/nvcc.sh ${args} example/wa-tor/dynasoar_no_cell/wator.cu -o bin/wator_dynasoar_no_cell
build_scripts/nvcc.sh ${args} example/wa-tor/dynasoar/wator.cu -o bin/wator_dynasoar
build_scripts/nvcc.sh ${args} example/wa-tor/baseline_aos/wator.cu -o bin/wator_baseline_aos
build_scripts/nvcc.sh ${args} example/wa-tor/baseline_soa/wator.cu -o bin/wator_baseline_soa
