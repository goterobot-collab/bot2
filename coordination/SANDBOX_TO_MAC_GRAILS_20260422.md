# SANDBOX → MAC — Nuevos grails (2026-04-22)

**Archivo principal:** `coordination/SANDBOX_FINAL_MASTER_V8_GRAILS.json`
**Rama:** `claude/verify-trading-strategies-Fnf0P` @ `goterobot-collab/bot2`

## Resumen

| Métrica | Valor |
|---------|-------|
| V8 total | **157 grails** (A:1, B:125, C:31) |
| Pool | 1042 grails ranked |
| Estrategias distintas | 111 |
| Plateau-confirmed | 125 en B |
| MC block bootstrap (p<0.05) | 46 (22 + 24 upgrades) |

## Tier A (consenso plateau + MC)

```
TV_Gann_Swing_MultiLayer  NEAR  1d  WR=75.4% n=57  PF=9.02  mc_p=0.005
```

## ⚠️ CRITICAL FLAG: TV_HTF_EMA_Gate sospechoso

**18 grails de TV_HTF_EMA_Gate en 5m/15m** con WR 93-100% y PF 85-3816. 
Es la familia más dominante (70/1042 = 6.7% del pool).

**Casos "too good to be true":**
- `TV_HTF_EMA_Gate NEAR 5m` WR=99.1% n=574 PF=568
- `TV_HTF_EMA_Gate OP 15m` WR=99.4% n=158 PF=3816
- `TV_HTF_EMA_Gate JTO 5m` WR=99.5% n=196 PF=1513

**Acción sugerida Mac:** verificar el signal builder `TV_HTF_EMA_Gate` en 
`strategies_v7/strategies_tv2_batch704.py` por lookahead. Los símbolos con 
n<100 y WR=100% son parte del mismo patrón. Si hay lookahead, todo el batch
704 debe rechazarse.

## Block bootstrap MC — Top upgrades con p<0.001

Estos son los más robustos estadísticamente:

```
TV_Weekly_MACD            SFP    15m p=0.0000 WR=74% n=303
TV_Weekly_MACD            LINK   5m  p=0.0000 WR=72% n=337
TV_Weekly_MACD            ARB    5m  p=0.0000
TV_Weekly_MACD            WLD    5m  p=0.0000
TV_Weekly_MACD            DYDX   5m  p=0.0000
TV_Daily_Close_Signal     LTC    5m  p=0.0000
TV_Daily_Close_Signal     SWARMS 5m  p=0.0000
TV_Daily_Close_Signal     OP     5m  p=0.0000
TV_HTF_EMA_Gate           INJ    1h  p=0.0002
TV_Weekly_MACD            INJ    5m  p=0.0002
TV_Weekly_MACD            ARB    1h  p=0.0002
```

## Paths a sincronizar

```bash
git pull origin claude/verify-trading-strategies-Fnf0P

# Archivos clave:
coordination/SANDBOX_FINAL_MASTER_V8_GRAILS.json        # LA INYECCIÓN
results/sandbox_master_ranking.json                     # pool completo 1042
results/sandbox_h1_plateau_all.json                     # plateau survivors
results/mc_block_bootstrap_c_tier_20260421.json         # MC round 1 (22 up)
results/mc_block_bootstrap_c_tier_20260422.json         # MC round 2 (24 up)
```

## Próximas preguntas para Mac

1. ¿Podés verificar el signal builder de `TV_HTF_EMA_Gate` (batch 704) por 
   lookahead bias antes de aplicar los 18 grails 5m/15m?
2. ¿Los 46 upgrades totales de block bootstrap pasan consenso CEREBRO + 
   SIGNAL_AUDITOR + COORDINADORA?
3. ¿Re-aplicamos PBO+DSR sobre los 125 B-tier para intentar promover más 
   a A-tier?

---
Sandbox 2026-04-22 · commit 10b9f58 · push inmediato
