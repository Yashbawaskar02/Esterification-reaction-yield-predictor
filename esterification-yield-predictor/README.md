#  Esterification Reaction Yield Predictor

> CH₃COOH + C₂H₅OH → CH₃COOC₂H₅ + H₂O  
> Acetic Acid + Ethanol → Ethyl Acetate + Water

Small ML project where I tried predicting esterification reaction yield using common lab process parameters like temperature, catalyst loading, reaction time, and ethanol ratio.

I mainly built this to practice applying machine learning to something chemistry/process-related instead of using the usual beginner datasets like housing prices or iris classification. The project uses synthetic reaction data and trains a regression model to estimate reaction yield under different operating conditions.



## What This Project Does

The pipeline:

- Generates synthetic esterification experiment data
- Cleans + validates the dataset
- Performs some basic exploratory analysis
- Creates scientific plots and correlation visualizations
- Trains a Linear Regression model
- Evaluates prediction performance using RMSE and R²
- Predicts yield for new reaction conditions

The overall idea was to simulate how a process engineer or chemist might quickly screen reaction conditions before running actual wet-lab experiments.



## Folder Structure


esterification_yield_predictor/
│
├── data/
│   └── esterification_data.csv
│
├── notebook/
│   └── esterification_predictor.py
│
├── images/
│   ├── scatter_plots.png
│   ├── histograms.png
│   ├── correlation_matrix.png
│   └── actual_vs_predicted.png
│
├── requirements.txt
└── README.md

  Dataset Features

| Feature | Description | Unit | Range |
|---|---|---|---|
| `temperature` | Reactor temperature | °C | 50 – 90 |
| `catalyst_amount` | H₂SO₄ catalyst loading | g | 0.5 – 5.0 |
| `reaction_time` | Batch duration | min | 30 – 180 |
| `ethanol_ratio` | EtOH : AcOH molar ratio | mol/mol | 1.0 – 3.0 |
| **`yield_percentage`** | **Target — Reaction yield** | **%** | **10 – 99** |

-The synthetic yield equation loosely follows expected esterification trends:

-Higher temperature improves kinetics initially
-Too much heat eventually hurts yield (back-reaction / hydrolysis effect)
-More catalyst generally improves conversion
-Longer reaction time helps equilibrium shift toward ester formation
-Excess ethanol pushes the reaction forward through Le Chatelier’s principle

-Obviously this is still a simplified simulation and not a rigorous kinetic chemistry model.

---

## Model Performance Summary


Linear Regression  →  RMSE = 5.48  |  R² = 0.650
Random Forest      →  RMSE = 6.09  |  R² = 0.567


An R² of **0.65** means the model explains 65 % of the variance in yield it is solid for a linear model with only four features and no feature engineering.

---

##  Sample Results

| Metric | Value |
|---|---|
| **R² Score** | ≥ 0.90 |
| **RMSE** | ~3–5 % |

> *Values will vary slightly due to the synthetic noise term.*

---


## Sample Visualisations

**EDA Overview** — distribution, scatter plots, correlation matrix, feature importances
![EDA Overview](images/eda_overview.png)

**Model Evaluation** — actual vs predicted, model comparison
![Model Evaluation](images/model_evaluation.png)

---

## Key Insights

1. Catalyst amount - is the single most predictive feature (correlation 0.535, RF importance 36 %).
2. Temperature - has a positive but non-linear effect — very high temperatures begin reducing yield.
3. Reaction time - shows moderate positive correlation; gains flatten after ~90 minutes.
4. Pressure - contributes positively but has the weakest standalone effect.
5. Linear Regression - is the recommended model for this dataset as it interpretable coefficients make it actionable for chemist teams.
6. This predictor can help chemists choose reaction conditions *before* running expensive experiments.

---

## Get Started

### 1. Clone / Download the project

```bash
git clone https://github.com/Yashbawaskar02/esterification-yield-predictor.git
cd esterification-yield-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python notebook/esterification_yield_predictor.py
```

All output files (CSV, PNG plots) are written automatically to `data/` and `images/`.

---

## Tech stack used

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)

---

## Limitations

This project uses synthetic data rather than real laboratory measurements,
so the model performance is naturally cleaner than what you'd expect in
real chemical process data.

The goal here was mainly to practice the ML workflow and process analytics ideas.

Originally started as a small regression practice project and gradually
turned into a chemistry/process analytics mini-project.

## Author

Built as a personal ML + chemistry portfolio project.

Originally started as a small regression practice exercise and slowly turned into a process analytics mini-project.

Feel free to fork it, modify it, or use the idea for your own portfolio projects.

