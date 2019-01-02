#!/usr/bin/env python3

import subprocess

NUM_RUNS = 5

benchmarks = [
  ("collision_baseline_aos", ""),
  ("collision_baseline_soa", ""),
  ("collision_soa", ""),
  ("gol_baseline_aos", "~/Downloads/utm.pgm"),
  ("gol_baseline_soa", "~/Downloads/utm.pgm"),
  ("gol_soa_no_cell", "~/Downloads/utm.pgm"),
  ("nbody_baseline_aos", ""),
  ("nbody_baseline_soa", ""),
  ("nbody_soa", ""),
  ("structure_baseline_aos", ""),
  ("structure_baseline_soa", ""),
  ("structure_soa", ""),
  ("sugarscape_baseline_aos", ""),
  ("sugarscape_baseline_soa", ""),
  ("sugarscape_no_cell", ""),
  ("traffic_baseline_aos", ""),
  ("traffic_baseline_soa", ""),
  ("traffic_soa", "")
]


with open('benchmark_results.csv', 'a') as file:
  for binary, arg in benchmarks:
    time = []
    for r in xrange(NUM_RUNS):
      output = subprocess.check_output([binary, arg])
      time.append(int(output))

    time.sort()

    file.write(binary)
    file.write(",")
    file.write(str(time[NUM_RUNS / 2]))
    file.write("\n")

