# Adaptive Demand Forecasting and Inventory Optimization

An AI/ML project for retail demand forecasting and inventory planning. The project brings together time-series forecasting models, baseline comparisons, a human-in-the-loop (HITL) feedback component, evaluation outputs, and a web application.

## Project Overview

Retail demand can vary across products and time. This project explores forecasting approaches that estimate future item-level demand and support inventory replenishment decisions.

The repository is organized around the following components:

- **Demand forecasting:** HeavyLSTM and baseline forecasting experiments.
- **Model evaluation:** Experiment outputs and metrics used to compare model performance.
- **Human-in-the-loop feedback:** A Bayesian feedback-correction experiment for adapting forecasts using feedback.
- **Dataset validation:** Experiments using retail datasets, including Favorita-adapted data and a secondary Indian retail dataset where available.
- **Web application:** The `webapp/` directory contains the project's application interface and related code.

> Dataset availability, model checkpoints, and generated outputs may depend on files that are not stored in this repository. See [Data and Artifacts](#data-and-artifacts).

## Repository Structure

```text
Adaptive-Demand-forecasting/
├── data/
│   ├── raw/                  # Source datasets (normally kept out of Git)
│   └── sequences/            # Prepared model sequences (may be generated locally)
├── experiments/
│   ├── 01_baselines/         # Baseline and model-comparison experiments, if present
│   ├── 02_bayesian/          # HITL/Bayesian feedback experiments, if present
│   └── 03_indian/            # Secondary-dataset experiments, if present
├── logs/                     # Local training and execution logs
├── models/                   # Model code, checkpoints, and related assets
├── results/                  # Evaluation metrics, figures, and predictions
├── webapp/                   # Web application
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

The folder names shown under `experiments/` describe the intended organization. The exact scripts available may differ by checkout; inspect each experiment folder for its entry-point script.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Karthikeyap-dev/Adaptive-Demand-forecasting.git
cd Adaptive-Demand-forecasting
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some packages may require platform-specific setup, especially if GPU acceleration is used. Follow the installation requirements for the relevant framework and hardware.

### 4. Prepare the data

Place the required datasets in the expected local paths used by the relevant experiment scripts. The previous project workspace used paths such as:

```text
data/raw/favorita/
data/raw/indian/
data/sequences/favorita/
data/sequences/indian/
```

These directories may not be included in Git because datasets and generated sequences can be large or subject to redistribution restrictions. Check the experiment scripts and configuration files for the exact filenames and paths expected by your local copy.

### 5. Run an experiment

Open the relevant folder under `experiments/` and identify its entry-point script. For example, if these scripts are present in your checkout:

```bash
python experiments/01_baselines/run_baselines.py
python experiments/02_bayesian/run_bayesian.py
```

Run only the command corresponding to a script that exists in your local repository. Experiments may require datasets, preprocessing, or model checkpoints to be prepared first.

## Data and Artifacts

Large datasets, generated sequences, training logs, and model checkpoints may be excluded from version control to keep the repository manageable.

- Obtain datasets from their original source and comply with their licenses and terms.
- Do not commit private, restricted, or unlicensed data.
- Checkpoint files may need to be generated locally or distributed separately.
- Selected, useful evaluation tables and figures can be committed under `results/`.
- Avoid committing temporary outputs, large prediction dumps, or logs unless they are necessary for reproducibility.

Add a `data/README.md` if you want to document dataset sources, expected filenames, preprocessing steps, and local directory setup.

## Results and Evaluation

Experiment metrics, figures, and prediction outputs are stored under `results/` when generated. Use the corresponding experiment scripts and configuration files to understand how each result was produced.

When reporting results, specify the dataset, evaluation split, metric definitions, and model configuration so that comparisons are reproducible. Results from different datasets or evaluation setups should not be compared without explaining the differences.

## Web Application

The application's code is in `webapp/`. Its startup command and any frontend/backend setup depend on the files present in that directory. Check for a project-specific README or package/dependency file there before launching it.

Do not commit environment files, credentials, API keys, or production secrets. Use environment variables and provide a sanitized `.env.example` when configuration documentation is needed.

## Reproducibility

For repeatable experiments:

1. Use the dependency versions recorded in `requirements.txt`.
2. Record dataset source, preprocessing steps, and data splits.
3. Keep experiment configurations and random seeds with the code where applicable.
4. Save evaluation metrics and figures with enough context to identify the experiment.
5. Document any hardware-specific settings, including GPU configuration.

## Contributing

For changes to this project, keep experiment scripts and configuration files clearly named, avoid committing generated or sensitive files, and update this README when setup steps or project structure change.

## License

No license is specified in this repository yet. Unless a license is added, do not assume that others have permission to reuse, modify, or redistribute the project. Add a license only after deciding how the project should be shared.
