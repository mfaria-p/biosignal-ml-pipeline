# PulseLM Biosignal Pipeline

A big-data pipeline for clinical prediction from physiological signals. It streams the [`Manhph2211/PulseLM`](https://huggingface.co/datasets/Manhph2211/PulseLM) dataset from Hugging Face, extracts features from raw biosignals, and trains machine learning models with Apache Spark and PyTorch. The data is multi-label by nature: each signal carries labels across several clinical categories at once. The dataset defines 12 such categories; this project models 10 of them (heart rate, blood pressure, stress, sleep-disordered breathing, HRV SDNN/RMSSD/pNN50, atrial fibrillation, arrhythmia, and respiratory rate). Two modeling approaches are explored: a binary-relevance setup with one Spark classifier per task, and a single multi-task PyTorch neural network that predicts all tasks at once (a shared trunk with per-task output heads).

The project ships in two flavours:

- **`egd_code_colab_final.ipynb`**: runs end-to-end in Google Colab, with optional Google Drive for persistence. Includes the multi-task PyTorch model and feature-importance analysis.
- **`egd_code_dataproc_final.ipynb`**: runs on a Google Cloud Dataproc Spark cluster, with GCS storage and a cross-cluster scaling benchmark.

## Pipeline overview

1. **Feature extraction**: each raw signal is reduced to a compact set of summary features instead of keeping every sample.
2. **QA label parsing**: labels are stored in a question/answer field with inconsistent shapes; a parser normalises them into consistent task labels and validates them against the allowed label set.
3. **Streaming ingestion**: splits are streamed from Hugging Face, features and labels are extracted, and everything is written to Parquet in batches to keep memory manageable.
4. **Distributed processing**: Parquet is read back with Spark for cleaning (schema check, null/NaN filtering, SQI signal-quality gating), label-coverage analysis, and class-distribution checks.
5. **Machine learning**: the 10 clinical tasks form a multi-label target (a label matrix with a mask marking which labels are present per sample), attacked two ways:
   - **Binary relevance on Spark** (one classifier per task): Random Forest with subset training (labeled rows only), Random Forest with whole-dataset training and class weights, and a per-task Spark MLP, plus a majority-class baseline.
   - **Multi-task PyTorch network** *(Colab notebook)*: a single `MultiTaskMLP` with a shared trunk and one output head per task, predicting all labels at once. Trained with masked binary-cross-entropy so missing labels are ignored, plus inverse-frequency class weights for imbalance.
6. **Evaluation & demo**: per-task accuracy and F1 tables, side-by-side model comparison, and a worked clinical Q&A demo that runs predictions on real test signals.
7. **Runtime benchmark** *(Dataproc only)*: times EDA, preprocessing, and training phases independently and saves results per cluster size to compare scaling across 2-worker and 4-worker clusters.

## Classification tasks

Each task starts from the dataset's original multi-class answer set and is collapsed into a binary target (`0` = normal, `1` = abnormal) for modeling. A reference class (usually `normal`) is treated as negative and everything else as positive; for atrial fibrillation, `af` is the positive class. Labels are sparse, so each sample only contributes to the tasks it actually has answers for, and a mask tracks which targets are present.

| Task | Original classes | Abnormal (positive) when |
|------|------------------|--------------------------|
| Heart rate (HR) | bradycardia, normal, tachycardia | not `normal` |
| Blood pressure (BP) | normal, elevated, hypertension_stage1/2, hypertensive_crisis | not `normal` |
| Stress | baseline, stress, amusement, meditation | not `baseline` |
| Sleep-disordered breathing (SDB) | normal (AHI<5), mild, moderate, severe | not `normal*` |
| HRV SDNN | low, normal, high | not `normal` |
| HRV RMSSD | low, normal, high | not `normal` |
| HRV pNN50 | low, normal, high | not `normal` |
| Atrial fibrillation (AF) | af, non_af | equals `af` |
| Arrhythmia | sinus_rhythm, pvc, pac, vt, svt, af | not `sinus_rhythm` |
| Respiratory rate (RR) | bradypnea, normal, tachypnea | not `normal` |

This gives a 10-column multi-label target per sample. The Spark models learn one binary classifier per column (binary relevance), while the PyTorch model learns all 10 columns jointly from a shared representation.

## Tech stack

Python, Apache Spark (PySpark: SQL, MLlib), PyTorch, scikit-learn, Hugging Face `datasets`, PyArrow/Parquet, pandas, NumPy, matplotlib, seaborn, and Google Cloud Dataproc + GCS for distributed runs.

## Getting started

### Colab
Open `egd_code_colab_final.ipynb` in Google Colab and run the cells in order. Dependencies install in the first section:

```bash
pip install -q huggingface_hub datasets pyarrow gcsfs scikit-learn matplotlib seaborn tqdm
```

Optionally mount Google Drive in section 0 so outputs survive after the session ends.

### Dataproc
Open `egd_code_dataproc_final.ipynb` on a Dataproc cluster and note the setup cells:

- **`OUT_DIR`** (section 2): point this at your GCS bucket (replace `YOUR_BUCKET_NAME`) before running ingestion.
- **Spark session** (section 4): auto-detects Dataproc and attaches to the existing YARN session.
- **`CLUSTER_LABEL`** (section 5.7): set this env var to tag each run (e.g. `dataproc_2w`, `dataproc_4w`); the benchmark cell uses it to label the saved JSON.

**Scaling benchmark workflow:** run on a 2-worker cluster, save JSON, resize to 4 workers, run again, then plot the cross-cluster comparison.

## Repository contents

| File | Description |
|------|-------------|
| `egd_code_colab_final.ipynb` | Colab version of the full pipeline |
| `egd_code_dataproc_final.ipynb` | Dataproc / Spark-cluster version with scaling benchmark |
| `Big_Data.pdf` | Project report |

## Dataset

[`Manhph2211/PulseLM`](https://huggingface.co/datasets/Manhph2211/PulseLM) is a large-scale, standardized PPG (photoplethysmography) dataset for multimodal physiological signal understanding, accessed via the Hugging Face `datasets` library. Each sample contains:

- A **PPG signal**: 10 seconds at 125 Hz, cleaned, processed, and normalized.
- A **text description**: metadata and recording context such as demographics, sensor and recording conditions, activities, and clinical measurements.
- **Question-answer pairs** spanning 12 clinical categories: heart rate, blood pressure, signal quality, stress, sleep-disordered breathing, HRV (SDNN, RMSSD, pNN50), atrial fibrillation, arrhythmia, SpO2, and respiratory rate.

The data is aggregated from 16 underlying source datasets. This project loads it via streaming, extracts summary features from the signals, and parses the QA fields into consistent labels for the 10 modeled tasks listed above.

## Authors

- Mariana Pereira
- Carolina Dias
- Elif Ozturk​
- Simão Bernardo
- Leonor Couto
- Sofia Fernandes
