#!/bin/bash
cmake -DCMAKE_BUILD_TYPE=Release -DALLOCATOR=halloc .

for (( i=1; i<=2048; i++))
do
  sizey=$(($i*512))
  echo "static const int kSizeY = ${sizey};" > example/wa-tor/soa/extra_config.h

  make wator_soa_no_cell

  res=`bin/wator_soa_no_cell`
  echo $res >> results/halloc.csv
  cat results/halloc.csv
done

