#!/usr/bin/env python3

import subprocess

NUM_RUNS = 5

benchmarks = [
  ("collision_baseline_aos", "", "collision", "baseline_aos"),
  ("collision_baseline_soa", "", "collision", "baseline_soa"),
  ("collision_soa", "", "collision", "soaalloc"),
  ("gol_baseline_aos", "~/Downloads/utm.pgm", "gol", "baseline_aos"),
  ("gol_baseline_soa", "~/Downloads/utm.pgm", "gol", "baseline_soa"),
  ("gol_soa_no_cell", "~/Downloads/utm.pgm", "gol", "soaalloc"),
  ("nbody_baseline_aos", "", "nbody", "baseline_aos"),
  ("nbody_baseline_soa", "", "nbody", "baseline_soa"),
  ("nbody_soa", "", "nbody", "soaalloc"),
  ("structure_baseline_aos", "", "structure", "baseline_aos"),
  ("structure_baseline_soa", "", "structure", "baseline_soa"),
  ("structure_soa", "", "structure", "soaalloc"),
  ("sugarscape_baseline_aos", "", "sugarscape", "baseline_aos"),
  ("sugarscape_baseline_soa", "", "sugarscape", "baseline_soa"),
  ("sugarscape_no_cell", "", "sugarscape", "soaalloc"),
  ("traffic_baseline_aos", "", "traffic", "baseline_aos"),
  ("traffic_baseline_soa", "", "traffic", "baseline_soa"),
  ("traffic_soa", "", "traffic", "soaalloc")
]


with open('benchmark_results.csv', 'a') as file:
  for binary, arg, bench_name, bench_type in benchmarks:
    time = []
    for r in range(NUM_RUNS):
      output = subprocess.check_output(["bin/" + binary, arg])
      time.append(int(output))
      print(int(output))

    time.sort()

    # Take the median
    line = bench_name + "," + bench_type + "," + str(time[int(NUM_RUNS / 2)]) + "\n"
    file.write(line)
    
    print(line)
