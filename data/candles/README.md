# Candles — gzipped CSVs para sandbox (sin acceso Binance API)

**Fuente**: `/Users/sabrina/CLAUDE CODE/data/activos_binance.db` (Mac, 27GB)
**Export date**: 2026-04-17
**Formato**: CSV gzipped. Header: `ts,open,high,low,close,volume`. `ts` = Unix ms.

## Contenido (42 archivos, 109 MB)

21 símbolos × 2 TFs nativos (5m + 1h). 15m/4h/1d se resamplean desde 5m/1h.

**Símbolos**: SFP, AGT, SWARMS, INJ, DYDX, GMX, OP, ARB, LINK, AVAX, NEAR, APT, SUI, TIA, SEI, JUP, PYTH, JTO, ONDO, WLD, PENDLE

**NO incluidos** (ausentes en DB Mac): MATIC, FTM — skip de lista de testing.

## Uso desde sandbox

```python
import pandas as pd
df = pd.read_csv("data/candles/SFP_5m.csv.gz", compression="gzip")
# columns: ts (ms), open, high, low, close, volume
```

Para 15m/4h/1d: resamplear desde 5m/1h con pandas `resample()`.
