# AI predictive maintenance: remaining useful life of engines

**Interactive dashboard:** [English](https://alvaromcib.github.io/ia-mantenimiento-predictivo/en/) · [Español](https://alvaromcib.github.io/ia-mantenimiento-predictivo/)

![Dashboard](docs/portfolio/cover.jpg)

A model that estimates how many cycles an engine has left before it fails, based on its sensors, and turns that prediction into **maintenance alerts** and an **estimated saving** compared with reactive maintenance.

Demo project built on public data (NASA C-MAPSS FD001). No client data is used.

## Results (test set, 100 engines)

| Metric | Value |
|---|---|
| RMSE | 13.8 cycles (67% better than predicting the mean) |
| MAE | 9.9 cycles |
| Failures caught in time (≤ 30 cycles) | 21 of 25 (84%) |
| False alarms | 0 of 75 healthy engines |
| Saving in the cost simulation* | 67% versus waiting for the failure |

\*Illustrative assumptions: unplanned failure €50,000, planned intervention €10,000. In a real project they are replaced with the plant's own costs.

## Approach

1. **Data:** 21 sensors and 3 operating settings per cycle; sensors with no variation are dropped.
2. **Features:** rolling means and standard deviations (5, 15 and 30 cycles) and the short-versus-long-term trend of each sensor.
3. **Model:** LightGBM with the RUL target capped at 125 cycles (a healthy engine carries no degradation information). Validation splits whole engines, never rows, to avoid leakage.
4. **Decision:** alert threshold, hit matrix and cost simulation, so the discussion is about euros and downtime, not just error.

## How to run it

```bash
pip install -r requirements.txt
python src/train.py              # trains, evaluates and writes results/results.json
python shared/build.py .         # rebuilds docs/index.html (ES) and docs/en/index.html (EN)
```

## How it would apply in a plant

- Data from SCADA/PLC or sensor history of pumps, compressors, motors or production lines.
- First, an assessment of the available data and of the business metric (downtime hours, cost of a breakdown).
- A closed pilot on one asset type, with alerts integrated into the tools the maintenance team already uses.

---

**Álvaro Martín Coll** · Physicist · AI engineer for industry and energy · [Malt profile](https://www.malt.com/profile/alvaromartincoll)

Data: A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08. NASA Prognostics Data Repository.
