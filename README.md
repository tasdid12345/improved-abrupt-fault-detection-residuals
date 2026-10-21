# Residual-Window LightGBM vs. ARX Residual Threshold for Abrupt Fault Detection

Model-based fault detection, isolation and identification (FDI) for abrupt faults, as studied in the
motivating paper "Email:{eric. j. manders, gautam. biswas}@vanderbilt.edu" (2004), relies on residuals
between a nominal model and the measured system. A fixed residual threshold is simple but can miss
small faults or raise false alarms when noise and excitation vary. This experiment asks whether a
class-balanced gradient-boosted classifier trained on sliding-window residual features does better.

**Question:** Can a class-balanced LightGBM classifier on residual windows detect abrupt parameter faults
in a simulated dynamic system with fewer missed faults and false alarms than a standard model-based
residual-threshold detector?

**Dataset:** Synthetic. A discrete-time second-order mass-spring-damper is simulated in numpy with seeded
random excitation and measurement noise. Faulty episodes switch stiffness or damping abruptly at a random
sample. This is not the paper's own dataset.

**Baseline:** a nominal ARX(2,2) one-step-ahead predictor fit by least squares on fault-free training
episodes, with a causally smoothed absolute residual compared to a fault-free threshold.

**Improved model:** a class-balanced LightGBM classifier on sliding-window features (residual, |residual|,
output and input statistics). Its hyper-parameters are chosen by grouped cross-validation on the training
episodes only.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json`. Charts are saved to `figures/`.

## Reference

This project is motivated by the following paper:

> EJ Manders (2004). *Email:{eric. j. manders, gautam. biswas}@ vanderbilt. edu*. Fault Detection, Supervision and Safety of. https://books.google.com/books?hl=en&lr=&id=QAU9N34xOSsC&oi=fnd&pg=PA339&dq=gautam.biswas%40vanderbilt.edu&ots=ubXJYLS2zb&sig=szoxeJlk3l3i8bNkge_fuUKlPqw

Paper author: Prof. Gautam Biswas, Vanderbilt University (gautam.biswas@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
