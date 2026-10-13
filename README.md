# Predicting Airline Satisfaction

[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10) asks for the probability that a passenger is satisfied. Submissions are scored by ROC AUC. The deadline is 31 October 2026.

Local files are in `dataset/playground-series-s6e10/`:

| File | Rows |
|---|---|
| `train.csv` | 699,635 |
| `test.csv` | 299,844 |
| `sample_submission.csv` | 299,844 |

The only missing value is `Arrival Delay in Minutes` (292 training rows). A row with that value missing is written as `missing`.

## Models

This project compares two open decision models on a Kaggle T4. It does not train LightGBM or XGBoost.

| Notebook | Model | Output |
|---|---|---|
| `notebooks/laya.ipynb` | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | `submission_laya.csv` |
| `notebooks/kev-0.8b.ipynb` | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) `@v1.0` | `submission_kev.csv` |

Kev-4B and OpenJev-4B are out of scope. They need about 14 GB of bf16 memory and are documented for an L40S, H100, or RTX 5090, not a 16 GB T4.

Both notebooks use the same protocol:

- Seed 42.
- 50,000 stratified fine-tune rows and 5,000 disjoint validation rows.
- One `choice` question: `satisfied` versus `not_satisfied`.
- The score is the probability of `satisfied`.

Each notebook prints `val_roc_auc`, `val_brier`, `val_log_loss`, and `sec_per_row` for the zero-shot run and after each fine-tune epoch. The test set is scored only by the better of those runs inside that notebook.

## Run on Kaggle

1. Create a notebook from `notebooks/laya.ipynb` or `notebooks/kev-0.8b.ipynb`.
2. Turn Internet on.
3. Set the accelerator to GPU T4 x2. Each notebook uses one T4.
4. Attach the competition data.
5. Run every cell except the last one.

Do not install Kev with `pip install kev`. That name is a different package. The Kev notebook installs it from GitHub.

Kev training turns off none-of-the-above augmentation so the two answer options stay intact. Serving turns off fused kernels and CUDA graphs because those paths target newer GPUs. If bf16 fails, the notebook retries in fp32.

Laya trains in the notebook on one T4. The second T4 is unused because dual-GPU training would require a separate Python file.

## Compare and submit

On Kaggle, metrics are written to `/kaggle/working/mlflow.db` and to:

- `mlflow_run_summary_laya.json`
- `mlflow_run_summary_kev-0.8b.json`

Download each summary into this repo, open the matching notebook, and run the last cell. That imports the run into `mlflow.db` at the repo root.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The experiment name is `s6e10-satisfaction`. Upload the submission whose notebook printed the higher validation ROC AUC. If the two AUC values are within 0.005, upload the one with the lower Brier score.

`mlflow.db` and the generated submission files are gitignored.
