#!/bin/bash
for (( i=1; i<=32; i++))
do
  build_scripts/build_structure.sh -f ${i}
  bin/structure_dynasoar >> structure_defrag_result.csv
  cat structure_defrag_result.csv
done

