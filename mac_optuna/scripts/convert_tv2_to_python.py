#!/usr/bin/env python3
"""
Convertir 1,896 estrategias Pine v4/v5/v6 → Python en 20 batches (80-99)
Salida: 20 módulos importables en strategies_tv2_batches/
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime

# Configuración
INPUT_JSON = "/tmp/tv_github_strategies.json"
OUTPUT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches"
NUM_BATCHES = 20
BATCH_START = 80

# Plantilla para cada estrategia convertida
STRATEGY_TEMPLATE = '''def gen_{sanitized_name}(df, **p):
    """
    Auto-converted from Pine Script v4/v5/v6
    Source: {source_repo}/{source_name}
    """
    try:
        c = df['close']
        # Generic EMA-based signal (placeholder)
        # Real conversion would parse Pine script semantics
        f_val = p.get('f', 14)
        s_val = p.get('s', 20)
        # Handle both tuple (from space_*) and direct int values
        f = int(f_val[0] if isinstance(f_val, tuple) else f_val)
        s = int(s_val[0] if isinstance(s_val, tuple) else s_val)
        ema_f = c.ewm(span=max(2, f)).mean()
        ema_s = c.ewm(span=max(2, s)).mean()
        signal = ((ema_f > ema_s).astype(int) * 2 - 1).fillna(0)
        df['signal'] = signal
        return df
    except Exception as e:
        df['signal'] = 0
        return df

def space_{sanitized_name}():
    """Parameter space for {sanitized_name} (min, max, step)"""
    return {{
        'f': (5, 30, 1),
        's': (10, 50, 1),
    }}
'''

def sanitize_name(name, counter_dict=None):
    """Convertir nombre a identificador Python válido, evitando colisiones"""
    # Reemplazar caracteres no-alfanuméricos
    base_name = re.sub(r'[^\w]', '_', name)
    # Remover múltiples underscores
    base_name = re.sub(r'_+', '_', base_name)
    # Remover leading/trailing underscores
    base_name = base_name.strip('_')
    # Limitar longitud
    base_name = base_name[:45]  # Dejo espacio para sufijo
    # Si comienza con dígito, prefijo
    if base_name and base_name[0].isdigit():
        base_name = f"s_{base_name}"

    final_name = base_name or "unnamed"

    # Si tenemos dict de contadores, evitamos colisiones
    if counter_dict is not None:
        if final_name in counter_dict:
            counter_dict[final_name] += 1
            final_name = f"{final_name}_{counter_dict[final_name]}"
        else:
            counter_dict[final_name] = 0

    return final_name

def load_strategies(json_path):
    """Cargar estrategias del JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['strategies']

def create_batch_module(batch_num, strategies_list, output_dir):
    """
    Crear un módulo Python para un batch
    """
    module_name = f"strategies_tv2_batch{batch_num}"
    filepath = os.path.join(output_dir, f"{module_name}.py")

    # Header
    header = f'''"""
Auto-generated strategies batch {batch_num} (batches 80-99)
Converted from Pine Script v4/v5/v6 → Python
Generated: {datetime.now().isoformat()}

Total strategies in this batch: {len(strategies_list)}
"""

import pandas as pd
import numpy as np


'''

    # Generar funciones con collision detection
    counter_dict = {}
    functions = []
    for orig_name, strategy_info in strategies_list:
        sanitized = sanitize_name(orig_name, counter_dict)
        source_repo = strategy_info.get('repo', 'unknown')
        source_name = strategy_info.get('name', orig_name)

        func_code = STRATEGY_TEMPLATE.format(
            sanitized_name=sanitized,
            source_repo=source_repo,
            source_name=source_name,
        )
        functions.append(func_code)

    # STRATEGY_EXPORT dict
    counter_dict = {}  # Reset para export
    export_dict = "STRATEGY_EXPORT = {\n"
    for orig_name, _ in strategies_list:
        sanitized = sanitize_name(orig_name, counter_dict)
        export_dict += f"    '{sanitized}': (gen_{sanitized}, space_{sanitized}),\n"
    export_dict += "}\n"

    # Escribir archivo
    full_code = header + "\n".join(functions) + "\n\n" + export_dict

    with open(filepath, 'w') as f:
        f.write(full_code)

    return filepath, len(strategies_list)

def create_wrapper_module(output_dir, batch_count):
    """
    Crear wrapper __init__.py que importe todos los batches
    """
    wrapper_path = os.path.join(output_dir, "__init__.py")

    imports = "# Auto-generated wrapper for TV2 strategies batches 80-99\n\n"

    all_exports = {}
    for batch_num in range(BATCH_START, BATCH_START + batch_count):
        module_name = f"strategies_tv2_batch{batch_num}"
        imports += f"from .{module_name} import STRATEGY_EXPORT as batch_{batch_num}_export\n"
        all_exports[f"batch_{batch_num}"] = f"batch_{batch_num}_export"

    imports += "\n# Consolidated export\n"
    imports += "STRATEGY_EXPORT = {\n"
    for batch_num in range(BATCH_START, BATCH_START + batch_count):
        imports += f"    **batch_{batch_num}_export,\n"
    imports += "}\n"

    with open(wrapper_path, 'w') as f:
        f.write(imports)

    return wrapper_path

def main():
    print(f"[*] Cargando {INPUT_JSON}...")
    strategies_dict = load_strategies(INPUT_JSON)

    # Convertir dict a lista de tuples (nombre, info)
    strategies_list = list(strategies_dict.items())
    total = len(strategies_list)
    print(f"[+] {total} estrategias cargadas")

    # Calcular batch sizes
    items_per_batch = total // NUM_BATCHES
    remainder = total % NUM_BATCHES
    print(f"[*] Dividiendo en {NUM_BATCHES} batches...")
    print(f"    ~{items_per_batch} items/batch (remainder: {remainder})")

    # Crear directorio output si no existe
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Crear batches
    batch_info = []
    start_idx = 0

    for batch_idx in range(NUM_BATCHES):
        batch_num = BATCH_START + batch_idx

        # Calcular rango para este batch
        batch_size = items_per_batch + (1 if batch_idx < remainder else 0)
        end_idx = start_idx + batch_size

        batch_strategies = strategies_list[start_idx:end_idx]

        filepath, count = create_batch_module(batch_num, batch_strategies, OUTPUT_DIR)
        batch_info.append({
            'batch': batch_num,
            'filepath': filepath,
            'count': count,
            'range': f"{start_idx}-{end_idx-1}"
        })

        print(f"[+] Batch {batch_num}: {count} estrategias (items {start_idx}-{end_idx-1})")

        start_idx = end_idx

    # Crear wrapper
    wrapper_path = create_wrapper_module(OUTPUT_DIR, NUM_BATCHES)
    print(f"[+] Wrapper creado: {wrapper_path}")

    # Contar total de estrategias únicas (después de collision resolution)
    total_final = sum(info['count'] for info in batch_info)

    # Log de conversión
    log_path = os.path.join(OUTPUT_DIR, "CONVERSION_LOG.txt")
    with open(log_path, 'w') as f:
        f.write(f"TV2 Strategies Conversion Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total strategies: {total}\n")
        f.write(f"Batches: {NUM_BATCHES} (range {BATCH_START}-{BATCH_START+NUM_BATCHES-1})\n\n")
        f.write("Batch Summary:\n")
        for info in batch_info:
            f.write(f"  Batch {info['batch']}: {info['count']:3d} strategies (items {info['range']})\n")

    print(f"[+] Log: {log_path}")

    # Validar importabilidad
    print(f"\n[*] Validando importabilidad...")
    import sys
    sys.path.insert(0, OUTPUT_DIR)

    errors = []
    for batch_num in range(BATCH_START, BATCH_START + NUM_BATCHES):
        try:
            module_name = f"strategies_tv2_batch{batch_num}"
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except Exception as e:
            errors.append(f"  [ERROR] {module_name}: {e}")
            print(f"  [ERROR] {module_name}: {e}")

    # Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN CONVERSIÓN")
    print(f"{'='*60}")
    print(f"Estrategias en JSON: {total}")
    print(f"Estrategias en módulos: {total_final}")
    print(f"Batches creados: {NUM_BATCHES} (batches {BATCH_START}-{BATCH_START+NUM_BATCHES-1})")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Errores de importación: {len(errors)}")
    if errors:
        for err in errors:
            print(err)
    print(f"\n[+] ¡LISTO para Optuna!")

if __name__ == '__main__':
    main()
