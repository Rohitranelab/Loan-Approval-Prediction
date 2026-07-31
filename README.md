<div align="center">

# 🏦 Loan Approval Prediction

### End-to-end Machine Learning system that predicts loan approval outcomes from applicant financial data

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-006ACC?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Models Used](#-models-used)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Example Predictions](#-example-predictions)
- [Configuration](#-configuration)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)
- [Why This Project Stands Out](#-why-this-project-stands-out)

---

## 📖 Project Overview

**Loan Approval Prediction** is a supervised binary classification project that predicts whether a loan application will be **approved** or **rejected**, based on an applicant's financial profile — income, credit score, requested loan amount, and years of employment.

> **Why it matters:** Manual loan underwriting is slow, inconsistent, and prone to human bias. A data-driven approval model helps financial institutions make faster, more consistent, and auditable lending decisions.

**Real-world applications**
- 🏦 Banks and NBFCs automating first-pass loan screening
- 💳 Fintech platforms offering instant credit decisions
- 📊 Risk teams building explainable, reproducible credit-scoring pipelines

**Expected users**
- Loan officers and credit analysts seeking a decision-support tool
- Data science teams benchmarking classification models on financial data
- Recruiters and engineers evaluating end-to-end ML project structure

---

## 🎬 Demo

> 📸 **Screenshot placeholder** — add a screenshot of the Streamlit app here (`assets/demo-screenshot.png`)

> 🎞️ **GIF placeholder** — add a short walkthrough GIF here (`assets/demo.gif`)

> 🌐 **Live deployment placeholder** — add your deployed Streamlit Cloud / Docker link here

---

## ✨ Features

- ✅ Clean, minimal data preprocessing pipeline
- ✅ Label encoding of the target variable
- ✅ Feature scaling with `StandardScaler`
- ✅ Comparison across **8 classification algorithms**
- ✅ Model selection based on accuracy, precision, recall, and F1-score
- ✅ Serialized model & scaler artifacts (`pickle`) for reuse
- ✅ Interactive **Streamlit** web app for real-time predictions
- ✅ **Dockerized** for one-command deployment
- ✅ Reproducible, notebook-driven workflow

---

## 🛠️ Tech Stack

**Languages**
- Python 3.13

**Libraries**
- Pandas, NumPy *(implicit via Pandas/Scikit-Learn)*
- Scikit-Learn (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `SVC`, `StandardScaler`, `LabelEncoder`)
- XGBoost (`XGBClassifier`)

**Frameworks**
- Streamlit (web application)

**Visualization**
- *Not implemented in the current notebook* (no plotting library imports present)

**Deployment**
- Docker (containerized Streamlit app)

**Version Control**
- Git & GitHub

---

## 📁 Project Structure

```
Loan-Approval-Prediction/
│
├── data/
│   └── loan_approval.csv          # Raw dataset (2,000 records)
│
├── model/
│   ├── loan_approval.pkl          # Trained GradientBoostingClassifier
│   └── scaling.pkl                # Fitted StandardScaler
│
├── notebooks/
│   └── Loan Approval Prediction.ipynb   # EDA, preprocessing, training & evaluation
│
├── app.py                         # Streamlit web application (inference UI)
├── requirements.txt                # Project dependencies
├── Dockerfile                      # Container build instructions
├── LICENSE                         # Apache 2.0 License
└── README.md                       # Project documentation
```

**Folder breakdown**
- **`data/`** — holds the source dataset used for training and evaluation.
- **`model/`** — stores the final trained model and the fitted scaler, both serialized with `pickle` for use at inference time.
- **`notebooks/`** — contains the full experimentation notebook: loading data, cleaning, encoding, scaling, training multiple models, and comparing results.
- **`app.py`** — a lightweight Streamlit front end that loads the saved model/scaler and serves live predictions.

---

## 🔄 Workflow

```
Data Collection
      ↓
Data Cleaning (drop irrelevant columns, check nulls)
      ↓
Label Encoding (target variable)
      ↓
Train/Test Split
      ↓
Feature Scaling (StandardScaler)
      ↓
Model Training (8 algorithms compared)
      ↓
Model Evaluation (Accuracy, Precision, Recall, F1)
      ↓
Best Model Selection (GradientBoostingClassifier)
      ↓
Model Serialization (pickle)
      ↓
Deployment (Streamlit + Docker)
```

---

## 🗂️ Dataset

| Detail | Description |
|---|---|
| **Source** | `data/loan_approval.csv` (included in repository) |
| **Samples** | 2,000 rows |
| **Raw columns** | `name`, `city`, `income`, `credit_score`, `loan_amount`, `years_employed`, `points`, `loan_approved` |
| **Features used for modeling** | `income`, `credit_score`, `loan_amount`, `years_employed` |
| **Dropped columns** | `name`, `city`, `points` (non-predictive / identifier columns) |
| **Target variable** | `loan_approved` (Boolean → encoded to 0/1) |
| **Class balance** | `False`: 1,121 · `True`: 879 |
| **Missing values** | None — verified with `df.isnull().sum()` |

---

## 🔍 Exploratory Data Analysis

The notebook performs a lightweight EDA focused on structural checks rather than visual analysis:

- Inspected data types and non-null counts via `df.info()`
- Verified **zero missing values** across all columns
- Reviewed target class distribution (`loan_approved.value_counts()`), showing a **mild class imbalance** (56% rejected vs. 44% approved)
- Dropped non-predictive identifier columns (`name`, `city`) and a redundant `points` column before modeling

> 📊 **Visualization placeholder** — no plots were generated in the current notebook. Consider adding distribution plots, correlation heatmaps, and class-balance charts here.

---

## 🧹 Data Preprocessing

| Step | Method |
|---|---|
| **Missing value handling** | Verified none present — no imputation required |
| **Column pruning** | Dropped `name`, `city`, `points` |
| **Target encoding** | `LabelEncoder` on `loan_approved` (`True`/`False` → `1`/`0`) |
| **Feature scaling** | `StandardScaler` fit on training data, applied to both train and test sets |
| **Feature selection** | Manual — retained `income`, `credit_score`, `loan_amount`, `years_employed` |
| **Outlier treatment** | *Not implemented* |
| **Train-test split** | 80% train / 20% test, `random_state=42` (1,600 / 400 samples) |

---

## 🤖 Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Non-linear baseline |
| Random Forest | Ensemble bagging classifier |
| AdaBoost | Ensemble boosting classifier |
| **Gradient Boosting** | **Final selected model** |
| K-Nearest Neighbors | Distance-based classifier |
| Support Vector Classifier (SVC) | Margin-based classifier |
| XGBoost | Gradient-boosted ensemble |

---

## 📈 Model Performance

All models were trained on the same scaled train/test split and evaluated on the held-out 20% test set (400 samples).

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 | **Gradient Boosting** | **0.9750** | 0.9626 | 0.9836 | 0.9730 |
| 🥈 | XGBoost | 0.9750 | 0.9727 | 0.9727 | 0.9727 |
| 🥉 | AdaBoost | 0.9725 | 0.9674 | 0.9727 | 0.9700 |
| 4 | Random Forest | 0.9700 | 0.9622 | 0.9727 | 0.9674 |
| 5 | Decision Tree | 0.9650 | 0.9617 | 0.9617 | 0.9617 |
| 6 | SVC | 0.9400 | 0.9297 | 0.9399 | 0.9348 |
| 7 | K-Nearest Neighbors | 0.9375 | 0.9341 | 0.9290 | 0.9315 |
| 8 | Logistic Regression | 0.8850 | 0.8785 | 0.8689 | 0.8736 |

> ✅ **`GradientBoostingClassifier`** was selected as the final production model based on the strongest overall balance of accuracy and recall, and was serialized to `model/loan_approval.pkl`.

> ROC AUC, RMSE, MAE, and R² were **not computed** in the current notebook (this is a classification task, so regression metrics are not applicable).

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Rohitranelab/Loan-Approval-Prediction.git

# Navigate into the project directory
cd Loan-Approval-Prediction

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Streamlit app locally

```bash
streamlit run app.py
```

The app will be available at **http://localhost:8501**.

### Run with Docker

```bash
# Build the image
docker build -t loan_approval .

# Run the container
docker run -p 8501:8501 loan_approval
```

### Retrain the model

Open and run `notebooks/Loan Approval Prediction.ipynb` end-to-end to reproduce preprocessing, training, evaluation, and re-export `loan_approval.pkl` / `scaling.pkl`.

---

## 🔮 Example Predictions

The Streamlit app accepts four applicant inputs and returns an approval decision with a confidence score:

| Income | Credit Score | Loan Amount | Years Employed | Prediction |
|---|---|---|---|---|
| 90,000 | 720 | 20,000 | 8 | ✅ Loan Approved |
| 30,000 | 400 | 45,000 | 1 | ❌ Loan Rejected |

---

## 🔧 Configuration

| Parameter | Location | Description |
|---|---|---|
| `test_size` | Notebook | Fraction of data held out for testing (default `0.2`) |
| `random_state` | Notebook | Seed for reproducible train/test splits (default `42`) |
| `n_estimators` | Notebook (model dict) | Number of estimators for ensemble models |
| `server.port` | `Dockerfile` | Streamlit app port (default `8501`) |
| Model/Scaler paths | `app.py` | `model/loan_approval.pkl`, `model/scaling.pkl` |

---

## 🗺️ Future Improvements

- 🧪 Add hyperparameter tuning (GridSearchCV / Optuna)
- 🔁 Introduce a CI/CD pipeline (GitHub Actions) for automated testing & deployment
- 📊 Add EDA visualizations (correlation heatmaps, feature distributions)
- 🧠 Add model explainability with SHAP / LIME
- 📈 Add cross-validation for more robust performance estimates
- 🗃️ Introduce a config file (YAML/JSON) instead of hardcoded parameters
- 🧪 Add unit tests for preprocessing and inference logic
- 📉 Address class imbalance with resampling techniques (SMOTE)

---

## 🤝 Contributing

Contributions are welcome and appreciated!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please open an issue first for major changes to discuss what you'd like to modify.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rohit Rane**

- GitHub: [@Rohitranelab](https://github.com/Rohitranelab)
- Email: ranerohit996@gmail.com

---

## 🙏 Acknowledgements

- Dataset used for training and evaluation (`data/loan_approval.csv`)
- [Scikit-Learn](https://scikit-learn.org/) for classification models and preprocessing utilities
- [XGBoost](https://xgboost.readthedocs.io/) for gradient-boosted tree modeling
- [Streamlit](https://streamlit.io/) for the interactive web application framework

---

## 🌟 Why This Project Stands Out

✔ End-to-end ML pipeline — from raw data to deployed application
✔ Clean, modular repository structure
✔ Production-ready inference app with Docker support
✔ Reproducible experiments (fixed random seed, versioned artifacts)
✔ Comparative benchmarking across 8 classification algorithms
✔ Well-documented, recruiter-friendly presentation
✔ Scalable foundation for further ML engineering work

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>