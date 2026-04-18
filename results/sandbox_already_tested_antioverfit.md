# Re-validation of 38 already_tested grails
- Total: 38
- Status breakdown: {'MATCH': 3, 'IMPL_MISSING': 23, 'DATA_MISSING: MASK_4h': 1, 'DATA_MISSING: BEL_1d': 1, 'DATA_MISSING: D_1h': 1, 'DATA_MISSING: USTC_1h': 1, 'DATA_MISSING: XMR_1d': 1, 'DATA_MISSING: XVG_4h': 1, 'DATA_MISSING: KERNEL_1h': 1, 'DATA_MISSING: Q_1h': 1, 'DATA_MISSING: QNT_4h': 1, 'DATA_MISSING: ALGO_5m': 1, 'DATA_MISSING: XMR_1h': 1, 'DATA_MISSING: RLC_4h': 1}
- Missing impl: 23 (['B2_MeanRev_EMA21', 'B6_Rubber_Band_5', 'B6_Rubber_Band_3', 'B5_MA_Envelope_5', 'B6_Rubber_Band_5']...)
- Missing data: 12
| Strategy | Symbol | TF | Expected WR | Observed WR | Gap | Status |
|----------|--------|----|-------------|-------------|-----|--------|
| `B5_MA_Envelope_3` | SFP/USDT:USDT | 1d | 85.2% | 85.2% | 0.0pp | MATCH |
| `B2_MeanRev_EMA21` | SFP/USDT:USDT | 1d | 84.5% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_5` | SFP/USDT:USDT | 1d | 84.2% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_3` | BEL/USDT:USDT | 1d | 75.6% | N/A% | Nonepp | IMPL_MISSING |
| `B5_MA_Envelope_5` | Q/USDT:USDT | 1h | 70.6% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_5` | BEL/USDT:USDT | 1d | 76.7% | N/A% | Nonepp | IMPL_MISSING |
| `B5_MA_Envelope_5` | D/USDT:USDT | 1h | 72.8% | N/A% | Nonepp | IMPL_MISSING |
| `VWAP_Double` | SFP/USDT:USDT | 4h | 79.7% | 79.7% | 0.0pp | MATCH |
| `VWAP_Double` | MASK/USDT:USDT | 4h | 81.9% | N/A% | Nonepp | DATA_MISSING: MASK_4h |
| `B5_MA_Envelope_3` | BEL/USDT:USDT | 1d | 74.3% | N/A% | Nonepp | DATA_MISSING: BEL_1d |
| `B5_MA_Envelope_3` | D/USDT:USDT | 1h | 76.4% | N/A% | Nonepp | DATA_MISSING: D_1h |
| `B5_MA_Envelope_3` | USTC/USDT:USDT | 1h | 78.4% | N/A% | Nonepp | DATA_MISSING: USTC_1h |
| `B5_MA_Envelope_5` | ALGO/USDT:USDT | 1h | 77.9% | N/A% | Nonepp | IMPL_MISSING |
| `B5_MA_Envelope_5` | AGT/USDT:USDT | 5m | 72.4% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_3` | NAORIS/USDT:USDT | 1h | 83.5% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_5` | LPT/USDT:USDT | 1d | 80.5% | N/A% | Nonepp | IMPL_MISSING |
| `B5_MA_Envelope_3` | XMR/USDT:USDT | 1d | 78.8% | N/A% | Nonepp | DATA_MISSING: XMR_1d |
| `B6_Rubber_Band_3` | C98/USDT:USDT | 1d | 79.6% | N/A% | Nonepp | IMPL_MISSING |
| `B2_MeanRev_EMA21` | TST/USDT:USDT | 1h | 74.2% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_5` | HANA/USDT:USDT | 1h | 76.4% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_5` | TST/USDT:USDT | 1h | 79.6% | N/A% | Nonepp | IMPL_MISSING |
| `VWAP_Double` | XVG/USDT:USDT | 4h | 84.0% | N/A% | Nonepp | DATA_MISSING: XVG_4h |
| `B6_Rubber_Band_7` | AGT/USDT:USDT | 1h | 76.0% | N/A% | Nonepp | IMPL_MISSING |
| `B6_Rubber_Band_3` | TST/USDT:USDT | 1h | 77.5% | N/A% | Nonepp | IMPL_MISSING |
| `VWAP_Double` | AGT/USDT:USDT | 1h | 82.2% | 82.2% | 0.0pp | MATCH |
| `VWAP_Double` | KERNEL/USDT:USDT | 1h | 80.3% | N/A% | Nonepp | DATA_MISSING: KERNEL_1h |
| `B5_MA_Envelope_3` | Q/USDT:USDT | 1h | 76.1% | N/A% | Nonepp | DATA_MISSING: Q_1h |
| `VWAP_Double` | QNT/USDT:USDT | 4h | 83.6% | N/A% | Nonepp | DATA_MISSING: QNT_4h |
| `B5_MA_Envelope_5` | SOL/USDT:USDT | 5m | 73.9% | N/A% | Nonepp | IMPL_MISSING |
| `B2_BB_1Std` | D/USDT:USDT | 1h | 72.1% | N/A% | Nonepp | IMPL_MISSING |
| `B5_MA_Envelope_3` | ALGO/USDT:USDT | 5m | 81.8% | N/A% | Nonepp | DATA_MISSING: ALGO_5m |
| `B2_MeanRev_EMA21` | AIO/USDT:USDT | 1h | 73.1% | N/A% | Nonepp | IMPL_MISSING |
| `B5_MA_Envelope_3` | XMR/USDT:USDT | 1h | 67.5% | N/A% | Nonepp | DATA_MISSING: XMR_1h |
| `B6_Rubber_Band_7` | MLN/USDT:USDT | 1h | 81.1% | N/A% | Nonepp | IMPL_MISSING |
| `B2_MeanRev_EMA21` | LTC/USDT:USDT | 5m | 70.9% | N/A% | Nonepp | IMPL_MISSING |
| `VWAP_Double` | RLC/USDT:USDT | 4h | 76.5% | N/A% | Nonepp | DATA_MISSING: RLC_4h |
| `B5_MA_Envelope_5` | XRP/USDT:USDT | 5m | 77.4% | N/A% | Nonepp | IMPL_MISSING |
| `B2_MeanRev_EMA21` | XRP/USDT:USDT | 5m | 72.5% | N/A% | Nonepp | IMPL_MISSING |
