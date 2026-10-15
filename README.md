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

The Kaggle notebooks are separate from the local ones in `notebooks/`. Do not upload the local notebooks.

1. Edit `id` in `kaggle/laya/kernel-metadata.json` or `kaggle/kev-0.8b/kernel-metadata.json`. Replace `YOUR_KAGGLE_USERNAME` with your Kaggle username.
2. Accept the competition rules, then push:

```bash
kaggle kernels push -p kaggle/laya
kaggle kernels push -p kaggle/kev-0.8b
```

Or upload `kaggle/laya/laya.ipynb` or `kaggle/kev-0.8b/kev-0.8b.ipynb` in the Kaggle UI. Turn Internet on, set the accelerator to GPU T4 x2, and attach the competition data. Each notebook uses one T4.

3. Run every cell.
4. Download `submission_laya.csv` or `submission_kev.csv` and the matching `mlflow_run_summary_*.json` from `/kaggle/working`.

Do not install Kev with `pip install kev`. That name is a different package. The Kev notebook installs it from GitHub.

Kev training turns off none-of-the-above augmentation so the two answer options stay intact. Serving turns off fused kernels and CUDA graphs because those paths target newer GPUs. If bf16 fails, the notebook retries in fp32.

Laya trains in the notebook on one T4. The second T4 is unused because dual-GPU training would require a separate Python file.

Checkpoints are written to `/tmp` and are not saved with the Kaggle version. Download the submission before the session ends.

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
