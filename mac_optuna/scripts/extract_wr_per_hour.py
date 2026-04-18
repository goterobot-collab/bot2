#!/usr/bin/env python3
"""
EXTRACTOR LIGERO de wr_per_hour desde forensic V2 JSON
========================================================
El JSON completo puede ser 7+ GB por los trades detallados.
Este script extrae SOLO strategy/symbol/timeframe/metrics (sin trades/equity/drawdown).

Reduce 7.6 GB → ~50 MB.
"""
import json
import sys
import os
from datetime import datetime

def extract_metrics(input_path, output_path):
    """Extract just the metrics from a forensic V2 JSON (streaming-friendly)."""
    print(f"Loading {input_path}...")
    print(f"Size: {os.path.getsize(input_path) / 1024 / 1024 / 1024:.2f} GB")

    data = json.load(open(input_path))

    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        print("ERROR: Unknown format")
        sys.exit(1)

    print(f"Results: {len(results)}")

    # Extract only what we need (no trades, equity, drawdown)
    light_results = []
    has_wph = 0
    for r in results:
        entry = {
            'strategy': r.get('strategy', ''),
            'symbol': r.get('symbol', ''),
            'timeframe': r.get('timeframe', ''),
            'optuna_wr': r.get('optuna_wr', 0),
            'gate_approved': r.get('gate_approved', False),
            'metrics': r.get('metrics', {}),
        }
        # Remove heavy sub-fields from metrics if any
        if 'trades' in entry['metrics']:
            del entry['metrics']['trades']

        if entry['metrics'].get('wr_per_hour'):
            has_wph += 1

        light_results.append(entry)

    output = {
        'timestamp': data.get('timestamp', datetime.now().isoformat()),
        'total_tested': data.get('total_tested', len(results)),
        'total_approved': data.get('total_approved', sum(1 for r in results if r.get('gate_approved'))),
        'total_blocked': data.get('total_blocked', sum(1 for r in results if not r.get('gate_approved'))),
        'results': light_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=1)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nExtracted: {len(light_results)} results ({has_wph} with wr_per_hour)")
    print(f"Output: {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 extract_wr_per_hour.py <input.json> [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.json', '_LIGHT.json')
    extract_metrics(input_path, output_path)
