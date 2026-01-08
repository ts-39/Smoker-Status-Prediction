# Kaggle Competition: Smoking Status Prediction

## Overview
This repository contains my solution for a Kaggle binary classification competition
predicting **smoking status** from tabular health-related data.

Final results:
- 🥇 **Private Leaderboard:** 1st place
- 🥉 **Public Leaderboard:** 3rd place (Score: 0.88323)

The solution is based on a **cross-validated ensemble of five heterogeneous models**,  
designed to maximize **robustness, generalization, and prediction diversity** rather than
optimizing a single model aggressively.

- Task: Binary Classification  
- Target: `smoking`  
- Final approach: 5-model ensemble with Stratified K-Fold CV  

---

## Modeling Philosophy
Rather than heavily tuning a single model, this project emphasizes:

- stability across cross-validation folds
- complementary inductive biases
- ensemble-friendly prediction diversity
- avoidance of leaderboard overfitting

All models are trained using **Stratified K-Fold Cross Validation**, and
out-of-fold (OOF) predictions are used to evaluate both individual models and ensembles.

---

## Exploratory Data Analysis (EDA)
Initial EDA was conducted to understand:

- class imbalance in the target variable
- feature distributions and skewness
- correlations among numerical features

Several domain-inspired features were engineered and evaluated during this phase.

---

## Feature Engineering Experiments
In addition to baseline preprocessing, the following **hand-crafted features** were tested:

- **BMI**  
  `weightkg / (heightcm / 100)^2`
- **Waist-to-height ratio**  
  `waistcm / heightcm`
- **AST / ALT ratio**  
  `ast / alt`
- **LDL / HDL ratio**  
  `ldl / hdl`

### Outcome
Although these features are medically interpretable,  
**adding them consistently degraded cross-validation and Public LB performance**.

As a result, they were **excluded from the final model**.

### Key Insight
More features do not necessarily improve performance—  
especially when strong nonlinear models already capture similar interactions implicitly.

---

## Core Feature Processing
The final feature set includes:

- numerical features only
- log-transformed biochemical variables
- fold-wise preprocessing to prevent leakage
- model-specific transformations (e.g., RankGauss for TabNet)

---

## Individual Models

### 1. LightGBM (Standard)
A strong and stable baseline optimized for tabular data.

**Characteristics**
- Raw + log-transformed numerical features
- Stratified K-Fold CV
- Early stopping
- Moderate regularization

This model achieved the best standalone stability.

---

### 2. LightGBM (Regularized / Safe Variant)
A conservative LightGBM configuration to intentionally alter model behavior.

**Differences**
- Smaller `num_leaves`
- Larger `min_data_in_leaf`
- Stronger L1 / L2 regularization

Purpose:
- reduce overfitting risk
- introduce ensemble diversity

---

### 3. TabNet (CV-based)
A deep learning model for tabular data with explicit feature selection.

**Techniques**
- Stratified K-Fold CV
- RankGauss (QuantileTransformer) applied per fold
- Median imputation
- Early stopping

TabNet exhibited prediction patterns distinct from tree-based models,
making it valuable for ensembling.

---

### 4. XGBoost
A gradient boosting model included for architectural diversity.

**Highlights**
- Histogram-based tree method
- Conservative depth and child-weight settings
- Early stopping per fold

---

### 5. Tabular Residual MLP (ResMLP)
A neural network specifically designed for tabular inputs.

**Architecture**
- Input projection layer
- Multiple residual blocks (BatchNorm + SiLU)
- Dropout regularization
- Binary classification head

**Training**
- StandardScaler fitted per fold
- BCEWithLogitsLoss with class imbalance handling
- AdamW optimizer
- Cosine annealing learning rate schedule
- Early stopping based on validation AUC

While not the strongest standalone model, it contributed meaningful diversity.

---

## Final Ensemble Model
### Strategy
The final prediction is a **simple average of five models**:

```

ensemble_prediction =
(pred_lgb

* pred_lgb_safe
* pred_tabnet
* pred_xgb
* pred_resmlp) / 5

```

### Design Rationale
- Use prediction probabilities only
- Avoid stacking to reduce overfitting risk
- Prioritize reproducibility and CV stability
- Exploit model diversity over complexity

### Evaluation
- All base models trained with Stratified K-Fold CV
- OOF predictions used to evaluate ensemble AUC
- Final test predictions averaged across folds
- This ensemble achieved **Public LB score: 0.88323 (Rank 3)**

---

## Key Learnings
- Strong baselines (LightGBM) remain critical for tabular problems
- Domain-inspired features are not always beneficial
- Conservative models can improve ensemble robustness
- Performance gains often come from **diversity**, not complexity
- CV design matters more than minor hyperparameter tuning

---

## Reproducibility & Notes
- This repository focuses on the **final ensemble implementation**
- Intermediate experimental notebooks were used during exploration
- Data files are excluded according to Kaggle rules
- The emphasis is on clarity, stability, and practical modeling decisions

---

## Environment
- Python 3.x
- LightGBM
- XGBoost
- PyTorch
- PyTorch TabNet
- scikit-learn
- NumPy / pandas
```

---
