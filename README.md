# Shark Tank India Funding Modeling and Prediction

## Background

Build an end-to-end ML system that analyzes the Shark Tank India dataset (750+ pitches, 80 columns, Seasons 1–5) to:

1. **Funding amount received** for a pitching team (numerical prediction)
2. **Whether a team receives funding at all** (binary prediction)
3. **Whether each shark is included in the funding** (binary prediction per shark)

This repository uses the Shark Tank India dataset from Kaggle and is organized as a modular ML pipeline with separate scripts for loading, preprocessing, splitting, training, testing, and visualization.

---

## Dataset

Source: Kaggle — **Shark Tank India** dataset  
Link: https://www.kaggle.com/datasets/thirumani/shark-tank-india/data

The dataset contains pitch-level information from Shark Tank India episodes, including business details, ask amount, valuation, equity offered, revenue figures, pitch outcomes, and shark participation information.

---

## Project Goals

### 1) Funding Prediction

Predict the amount of funding a startup receives.

- **Target type:** Regression
- **Output:** Predicted funding amount
- **Zero funding case:** If no deal is made, the model can output `0` or the classification stage can be used to decide whether funding happens first.

### 2) Deal / No-Deal Prediction

Predict whether a pitch results in funding.

- **Target type:** Binary classification
- **Output:** `1` if funded, `0` if not funded

### 3) Shark Participation Prediction

Predict which sharks take part in the deal.

- **Target type:** Multi-label / binary classification
- **Output:** One binary label per shark
- **Sharks considered:** Main sharks only, unless the team decides to expand to guest sharks later

---

## Repository Structure

```text
Shark-tank-India-Decision-Modeling/
├── EDA/
│   ├── eda_shreyash.ipynb
│   ├── eda_soham.ipynb
│   ├── eda_varnan.ipynb
│   └── eda_yadnik.ipynb
├── scripts/
│   ├── dataloader.py
│   ├── preprocessing.py
│   ├── split.py
│   ├── train.py
│   ├── test.py
│   ├── main.py
│   └── graph.py
├── README.md
└── LICENSE
```

---

## What Each Script Is For

### `scripts/dataloader.py`

Handles dataset loading and basic validation.

- Read the Kaggle CSV
- Check columns, shape, and missing values
- Return a clean dataframe for the next stage

### `scripts/preprocessing.py`

Cleans the data and prepares it for modeling.

- Handle missing values
- Encode categorical features
- Create derived features
- Prepare targets for:
  - funding amount prediction
  - funding/no-funding classification
  - shark-wise binary labels

### `scripts/split.py`

Creates train/test splits.

- Split the dataset reproducibly
- Use stratification for classification tasks where needed
- Keep the split consistent across experiments

### `scripts/train.py`

Trains the machine learning models.

- Fit models on the training data
- Save trained artifacts
- Compare results across models

### `scripts/test.py`

Evaluates model performance.

- Run predictions on the test set
- Calculate metrics
- Compare actual vs predicted values

### `scripts/main.py`

Acts as the main entry point for the project.

- Run the pipeline in order
- Connect loading, preprocessing, splitting, training, testing, and plotting

### `scripts/graph.py`

Generates visualizations.

- Plot distributions
- Show model comparison charts
- Visualize prediction outcomes and feature patterns

---

## Responsibility Division

The project is divided into **data pipeline development** and **modeling**, with each member possessing understanding giving inputs in all domains, while having their main area of working as follows:

### [Shreyash](https://github.com/cse-shreyashjaiswal), [Soham](https://github.com/sohamnakhate), and [Yadnik](https://github.com/yadnikbangale)

Responsible for building the core data foundation of the project:

- **Data Loading (`dataloader.py`)**
  - Dataset ingestion and validation
  - Schema checks and consistency

- **Preprocessing (`preprocessing.py`)**
  - Data cleaning and handling missing values
  - Feature engineering and encoding
  - Preparing targets for:
    - funding amount prediction
    - deal/no-deal prediction
    - shark participation prediction

- **Exploratory Data Analysis (EDA)**
  - Understanding dataset structure and distributions
  - Analyzing funding patterns, industries, and shark behavior
  - Generating insights to guide feature engineering and modeling

- **Iterative Preprocessing Updation**
  - Studying model performance analysis created by [Varnan](https://github.com/varnan6)
  - Interpreting the influence of preprocessing on model performmance
  - Updating and iterating through viable preprocessing techniques to improve model performance

> As there are ~80 attributes, these responsibilities are handled **jointly** with all three members extensively collaborating across loading, preprocessing, and EDA.

---

### [Varnan](https://github.com/varnan6)

Responsible for building and validating the predictive system:

- **Preprocessing Supervision**:
  - Oversees dataloading and preprocessing done by [Shreyash](https://github.com/cse-shreyashjaiswal), [Soham](https://github.com/sohamnakhate), and [Yadnik](https://github.com/yadnikbangale)
  - Verify processed data integrity and compatibility with respect to model expected input
  - Implementing suggestions from [Shreyash](https://github.com/cse-shreyashjaiswal), [Soham](https://github.com/sohamnakhate), and [Yadnik](https://github.com/yadnikbangale) for better modeling

- **Training (`train.py`)**
  - Implement and train models for:
    - Funding amount prediction (regression)
    - Deal/no-deal prediction (classification)
    - Shark participation prediction (multi-label)

- **Evaluation (`test.py`)**
  - Model performance analysis using appropriate metrics
  - Comparing model outputs and approaches

- **Pipeline Integration (`main.py`)**
  - Connecting all components into a complete end-to-end workflow

---

## Proposed Modeling Approach

A practical pipeline for this project is:

1. Load the raw Kaggle dataset
2. Clean and preprocess the data
3. Create task-specific targets
4. Split into train and test sets
5. Train models for:
   - funding amount prediction
   - deal/no-deal prediction
   - shark participation prediction
6. Evaluate and compare results
7. Visualize model outputs

---

## Evaluation

### For Funding Amount Prediction

Use regression metrics such as:

- MAE
- RMSE
- R²
- MAPE

### For Deal / No-Deal Prediction

Use classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### For Shark Participation Prediction

Use multi-label metrics such as:

- Micro F1
- Macro F1
- Hamming loss
- Per-shark accuracy

---

## How to Run

Once the data is placed in the expected location, the project can be run from the main script.

```bash
python scripts/main.py
```

Depending on how the scripts are implemented, you may also run them individually for debugging or evaluation.

---

## Notes

- The dataset is sourced from Kaggle, so authentication may be required for download.
- The repository is structured to support experimentation, model comparison, and future expansion.

---

## Future Improvements Checklist

- Add a Streamlit dashboard
- Add hyperparameter tuning
- Add feature importance and interpretability plots
- Create model zoo for performance reproducibility

---

## License

This project is released under the MIT License.
