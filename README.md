# MLOps_IMDb

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Film review sentiment analysis.

## Setup Instructions

Follow these steps to set up the project environment and run the complete pipeline:

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [DVC](https://dvc.org/) for data versioning and pipeline management

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOps_IMDb
   ```

2. **Create and activate virtual environment**
   ```bash
   make create_environment
   source ./.venv/bin/activate  # On Unix/macOS
   # or
   .\.venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   make requirements
   # or directly with uv
   uv sync
   ```

4. **Run the complete DVC pipeline**
   ```bash
   dvc pull -T data/raw/imdb_train.csv data/raw/imdb_test.csv
   dvc repro
   ```

This will execute all pipeline stages in order:
- **prepare**: Clean and preprocess the raw IMDB data
- **features**: Build TF-IDF features from the cleaned text
- **train**: Train the sentiment analysis model
- **eval**: Evaluate the model and generate metrics

### Alternative: Run individual stages

You can also run individual pipeline stages:

```bash
# Data preparation
dvc repro prepare

# Feature engineering
dvc repro features

# Model training
dvc repro train

# Model evaluation
dvc repro eval
```

### Verify Setup

After running the pipeline, you should see:
- Processed data in `data/processed/`
- Trained models in `models/`
- Evaluation metrics in `reports/metrics.json`
- Visualization plots in `reports/figures/`

### Development Commands

```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Clean compiled files
make clean
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_imdb and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_imdb   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_imdb a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

