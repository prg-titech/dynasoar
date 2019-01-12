#!/bin/bash
cmake -DCMAKE_BUILD_TYPE=Release -DALLOCATOR=mallocMC .

for (( i=1; i<=512; i++))
do
  sizey=$(($i*32))
  echo "static const int kSizeY = ${sizey};" > example/wa-tor/soa/extra_config.h

  make wator_soa_no_cell

  res=`bin/wator_soa_no_cell`
  echo $res >> results/mallocmc.csv
  cat results/mallocmc.csv
done

