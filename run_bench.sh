#!/bin/bash
cmake -DCMAKE_BUILD_TYPE=Release -DALLOCATOR=SoaAlloc .

for (( i=1; i<=256; i++))
do
  sizey=$(($i*16))
  echo "static const int kSizeY = ${sizey};" > example/wa-tor/soa/extra_config.h

  make wator_soa_no_cell

  res=`bin/wator_soa_no_cell`
  echo $res >> results/soaalloc.csv
  cat results/soaalloc.csv
done

