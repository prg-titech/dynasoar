#!/usr/bin/env python3

import subprocess

NUM_RUNS = 5

def run_benchmarks(benchmarks, allocator):
  with open('benchmark_results_' + allocator + '.csv', 'a') as file:
    for binary, arg, bench_name, bench_type in benchmarks:
      time = []
      for r in range(NUM_RUNS):
        try:
          output = subprocess.check_output(["bin/" + binary, arg])
          data = output.strip().decode("ascii")
          time.append(data)
          print(data)
        except subprocess.CalledProcessError:
          print("Process error")

      # Take the median
      line = bench_name + "," + bench_type + "," + ",".join(map(str, time)) + "\n"
      file.write(line)
      
      print(line)


# Baseline benchmark.
subprocess.call(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DALLOCATOR=SoaAlloc", "."])
subprocess.call(["make"])
benchmarks = [
  ("collision_baseline_aos", "", "collision", "baseline_aos"),
  ("collision_baseline_soa", "", "collision", "baseline_soa"),
  ("gol_baseline_aos", "/home/matthias/Downloads/utm.pgm", "gol", "baseline_aos"),
  ("gol_baseline_soa", "/home/matthias/Downloads/utm.pgm", "gol", "baseline_soa"),
  ("nbody_baseline_aos", "", "nbody", "baseline_aos"),
  ("nbody_baseline_soa", "", "nbody", "baseline_soa"),
  ("structure_baseline_aos", "", "structure", "baseline_aos"),
  ("structure_baseline_soa", "", "structure", "baseline_soa"),
  ("sugarscape_baseline_aos", "", "sugarscape", "baseline_aos"),
  ("sugarscape_baseline_soa", "", "sugarscape", "baseline_soa"),
  ("traffic_baseline_aos", "", "traffic", "baseline_aos"),
  ("traffic_baseline_soa", "", "traffic", "baseline_soa"),
]
run_benchmarks(benchmarks, "baseline")


# SoaAlloc benchmark.
subprocess.call(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DALLOCATOR=SoaAlloc", "."])
subprocess.call(["make"])
benchmarks = [
  ("collision_soa", "", "collision", "allocator"),
  ("gol_soa_no_cell", "/home/matthias/Downloads/utm.pgm", "gol", "allocator"),
  ("linux_scalability_soa", "", "linux", "allocator"),
  ("nbody_soa", "", "nbody", "allocator"),
  ("structure_soa", "", "structure", "allocator"),
  ("sugarscape_no_cell", "", "sugarscape", "allocator"),
  ("traffic_soa", "", "traffic", "allocator"),
  ("wator_soa_no_cell", "", "wator", "allocator")
]
run_benchmarks(benchmarks, "soaalloc")


# mallocMC benchmark.
subprocess.call(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DALLOCATOR=mallocMC", "."])
subprocess.call(["make"])
benchmarks = [
  ("collision_soa", "", "collision", "allocator"),
  ("gol_soa_no_cell", "/home/matthias/Downloads/utm.pgm", "gol", "allocator"),
  ("linux_scalability_soa", "", "linux", "allocator"),
  ("nbody_soa", "", "nbody", "allocator"),
  ("structure_soa", "", "structure", "allocator"),
  ("sugarscape_no_cell", "", "sugarscape", "allocator"),
  ("traffic_soa", "", "traffic", "allocator"),
  ("wator_soa_no_cell", "", "wator", "allocator")
]
run_benchmarks(benchmarks, "mallocmc")


# halloc benchmark.
subprocess.call(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DALLOCATOR=halloc", "."])
subprocess.call(["make"])
benchmarks = [
  ("collision_soa", "", "collision", "allocator"),
  ("gol_soa_no_cell", "/home/matthias/Downloads/utm.pgm", "gol", "allocator"),
  ("linux_scalability_soa", "", "linux", "allocator"),
  ("nbody_soa", "", "nbody", "allocator"),
  ("structure_soa", "", "structure", "allocator"),
  ("sugarscape_no_cell", "", "sugarscape", "allocator"),
  ("traffic_soa", "", "traffic", "allocator"),
  ("wator_soa_no_cell", "", "wator", "allocator")
]
run_benchmarks(benchmarks, "halloc")


# bitmap benchmark.
subprocess.call(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DALLOCATOR=bitmap", "."])
subprocess.call(["make"])
benchmarks = [
  ("collision_soa", "", "collision", "allocator"),
  ("gol_soa_no_cell", "/home/matthias/Downloads/utm.pgm", "gol", "allocator"),
  ("linux_scalability_soa", "", "linux", "allocator"),
  ("nbody_soa", "", "nbody", "allocator"),
  ("structure_soa", "", "structure", "allocator"),
  ("sugarscape_no_cell", "", "sugarscape", "allocator"),
  ("traffic_soa", "", "traffic", "allocator"),
  ("wator_soa_no_cell", "", "wator", "allocator")
]
run_benchmarks(benchmarks, "bitmap")


# cuda benchmark.
subprocess.call(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DALLOCATOR=cuda", "."])
subprocess.call(["make"])
benchmarks = [
  ("collision_soa", "", "collision", "allocator"),
  ("gol_soa_no_cell", "/home/matthias/Downloads/utm.pgm", "gol", "allocator"),
  ("linux_scalability_soa", "", "linux", "allocator"),
  ("nbody_soa", "", "nbody", "allocator"),
  ("structure_soa", "", "structure", "allocator"),
  ("sugarscape_no_cell", "", "sugarscape", "allocator"),
  ("traffic_soa", "", "traffic", "allocator"),
  ("wator_soa_no_cell", "", "wator", "allocator")
]
run_benchmarks(benchmarks, "cuda")
