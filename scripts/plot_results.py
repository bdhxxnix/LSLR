#!/usr/bin/env python3
"""Plot experiment results for LSLR-PLA comparison."""

import csv
import sys
import os
from collections import defaultdict
import math

def load_results(filename):
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ('method', 'dataset', 'workload'):
                    continue
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    row[key] = 0.0
            results.append(row)
    return results

def print_table_1(results):
    """Update performance table."""
    print("\n" + "=" * 120)
    print("TABLE 1: Update Performance")
    print("=" * 120)

    # Group by dataset, epsilon, workload
    groups = defaultdict(list)
    for r in results:
        key = (r['dataset'], int(r['epsilon']), r['workload'], int(r['n_initial']))
        groups[key].append(r)

    for (dataset, eps, workload, n), group in sorted(groups.items()):
        if len(group) <= 1:
            continue
        print(f"\n--- {dataset}, n={n}, eps={eps}, {workload} ---")
        print(f"{'Method':<20} {'Total(s)':>10} {'Avg(us)':>10} {'p50(us)':>10} {'p95(us)':>10} {'p99(us)':>10} {'Speedup':>10}")
        print("-" * 80)

        # Find FullRebuildEach time for speedup baseline
        fre_time = None
        for r in group:
            if r['method'] == 'FullRebuildEach':
                fre_time = r['total_update_us']
                break

        for r in sorted(group, key=lambda x: x['total_update_us']):
            speedup = fre_time / r['total_update_us'] if fre_time and r['total_update_us'] > 0 else 0
            print(f"{r['method']:<20} {r['total_update_us']/1e6:>10.2f} {r['avg_insert_us']:>10.1f} {r['p50_insert_us']:>10.1f} {r['p95_insert_us']:>10.1f} {r['p99_insert_us']:>10.1f} {speedup:>9.1f}x")

def print_table_2(results):
    """Model quality table."""
    print("\n" + "=" * 120)
    print("TABLE 2: Model Quality (Segment Count and Error)")
    print("=" * 120)

    groups = defaultdict(list)
    for r in results:
        key = (r['dataset'], int(r['epsilon']), r['workload'], int(r['n_initial']))
        groups[key].append(r)

    for (dataset, eps, workload, n), group in sorted(groups.items()):
        print(f"\n--- {dataset}, n={n}, eps={eps}, {workload} ---")
        print(f"{'Method':<20} {'Segments':>10} {'OptFinal':>10} {'Overhead':>10} {'MaxErr':>10} {'Viols':>8}")
        print("-" * 68)
        for r in sorted(group, key=lambda x: x['segment_overhead']):
            print(f"{r['method']:<20} {int(r['final_segments']):>10} {int(r['optimal_final_segments']):>10} {r['segment_overhead']:>10.3f} {r['max_error']:>10.1f} {int(r['violations']):>8}")

def print_table_3(results):
    """Local repair behavior table."""
    print("\n" + "=" * 120)
    print("TABLE 3: Local Repair Behavior (LSLR methods only)")
    print("=" * 120)

    lslr_methods = {'LSLR-c0', 'LSLR-c1', 'LSLR-c2', 'LSLR-adaptive'}
    groups = defaultdict(list)
    for r in results:
        if r['method'] not in lslr_methods:
            continue
        key = (r['dataset'], int(r['epsilon']), r['workload'], int(r['n_initial']))
        groups[key].append(r)

    for (dataset, eps, workload, n), group in sorted(groups.items()):
        print(f"\n--- {dataset}, n={n}, eps={eps}, {workload} ---")
        print(f"{'Method':<20} {'AvgKeysRep':>12} {'AvgSegsRep':>12} {'ExpRate':>10} {'WindowExp':>10}")
        print("-" * 64)
        for r in sorted(group, key=lambda x: x['method']):
            print(f"{r['method']:<20} {r['avg_keys_repaired']:>12.1f} {r['avg_segments_repaired']:>12.2f} {r['expansion_rate']:>10.3f} {int(r['window_expansions']):>10}")

def print_summary_table(results):
    """Print summary statistics across all experiments."""
    print("\n" + "=" * 120)
    print("AGGREGATE SUMMARY")
    print("=" * 120)

    methods = defaultdict(list)
    for r in results:
        methods[r['method']].append(r)

    print(f"\n{'Method':<20} {'Avg Speedup':>12} {'Avg Overhead':>13} {'Max Viols':>10}")
    print("-" * 55)

    fre_times = {}
    for r in results:
        if r['method'] == 'FullRebuildEach':
            key = (r['dataset'], int(r['epsilon']), r['workload'], int(r['n_initial']))
            fre_times[key] = r['total_update_us']

    for method in sorted(methods.keys()):
        group = methods[method]
        if len(group) == 0:
            continue

        speedups = []
        overheads = []
        max_viols = 0
        for r in group:
            key = (r['dataset'], int(r['epsilon']), r['workload'], int(r['n_initial']))
            if key in fre_times and r['total_update_us'] > 0:
                speedups.append(fre_times[key] / r['total_update_us'])
            if r['optimal_final_segments'] > 0:
                overheads.append(r['segment_overhead'])
            max_viols = max(max_viols, int(r['violations']))

        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        avg_overhead = sum(overheads) / len(overheads) if overheads else 0
        print(f"{method:<20} {avg_speedup:>11.1f}x {avg_overhead:>13.3f} {max_viols:>10}")

def main():
    if len(sys.argv) < 2:
        filename = 'results/experiment_results.csv'
    else:
        filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        sys.exit(1)

    results = load_results(filename)
    print(f"Loaded {len(results)} result rows from {filename}")

    print_table_1(results)
    print_table_2(results)
    print_table_3(results)
    print_summary_table(results)

if __name__ == '__main__':
    main()
