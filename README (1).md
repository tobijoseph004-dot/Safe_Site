# Predicting Severe Occupational Injuries in Construction Workplaces Using Machine Learning

## Project Objective
Develop a machine learning model to predict whether a workplace incident is likely
to result in a **Fatal** or **Nonfatal** outcome, enabling HSE (Health & Safety)
professionals to implement preventive measures and improve worker safety.

## Dataset
**Source:** OSHA (U.S. Occupational Safety and Health Administration) construction
incident records, 2015-2017.
**Size:** 4,847 real workplace incident records, 29 original columns.

## Analytical Technique: Classification
We chose **classification** because our target variable (Degree of Injury) is
categorical — Fatal or Nonfatal — not a continuous number. This rules out
regression, and we have labeled historical data with predefined outcomes to learn
from, which rules out clustering (used when no labels exist).

## Why Random Forest? (Justification vs. Alternatives)
We trained and compared four models on the same data:

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Logistic Regression | ~58% | ~0.58 | Simple linear baseline |
| Naive Bayes | 60.1% | 0.514 | Fast, but weak recall on Nonfatal (23%) |
| Gradient Boosting | 70.1% | 0.641 | Higher accuracy, but only 38% recall on Nonfatal — heavily favors the majority class |
| **Random Forest (tuned)** | **67.1%** | **0.656** | **Best balance across both outcomes** |

We selected Random Forest based on **Macro F1**, not raw accuracy, because Macro F1
weighs both classes equally. Gradient Boosting's higher accuracy came from
under-predicting Nonfatal cases — a real weakness for a safety tool, where missing
either outcome type has consequences. Random Forest with `class_weight='balanced'`
gave the most reliable performance across both Fatal and Nonfatal cases.

## Data Cleaning & Feature Selection
- Started with 4,847 rows and 29 columns.
- Selected 4 features that would realistically be known **before** an incident
  occurs: `Event type`, `Environmental Factor`, `Human Factor`, `Task Assigned`.
- Dropped `Construction End Use` and `Project Type` (~78-79% missing values).
- Dropped `fall_ht` (100% zero values, not usable).
- Dropped 9 rows with remaining missing values.
- Final clean dataset: **4,838 rows**.

### Important design decision: avoiding data leakage
We experimented with adding `Nature of Injury` and `Part of Body` as extra
features. This pushed accuracy up to **85%** — but we deliberately **rejected**
this approach. These columns describe the injury *after* it happened
(e.g., "Electrocution", "Amputation, Crushing") and are near-synonyms for the
outcome itself, not genuine predictive factors. Using them would be a form of
**data leakage**, and more importantly, this information doesn't exist yet when
an HSE officer is trying to assess a task *in advance* — which is the real-world
scenario this tool is built for. We chose the lower, honest 67.1% model because
it only uses information genuinely available before an incident occurs.

## Model Evaluation
**Final model: Random Forest (tuned)**
- Accuracy: 67.1%
- Macro F1: 0.656
- Fatal — Precision: 0.74, Recall: 0.72
- Nonfatal — Precision: 0.57, Recall: 0.59

**Confusion Matrix:**

|  | Predicted Fatal | Predicted Nonfatal |
|---|---|---|
| **Actual Fatal** | 427 | 165 |
| **Actual Nonfatal** | 153 | 223 |

The model catches the majority of both outcome types and does not collapse into
predicting only one class — errors are distributed roughly evenly in both
directions, meaning the model isn't just guessing the majority class.

## What the Results Mean
Predicting a Fatal vs. Nonfatal outcome from only situational factors (event
type, environment, human factor, task) is a genuinely hard problem — these four
factors alone don't capture everything that determines an outcome's severity, and
some overlap naturally exists between what leads to a fatal vs. nonfatal incident.
67% accuracy is meaningfully above chance (50%) and is in line with accuracy
ranges reported in real published occupational injury prediction research.

## Recommendation
This tool should be used as a **supporting signal**, not a replacement for
professional HSE judgment. When a task is flagged with a higher predicted
likelihood of a fatal outcome, HSE officers can use that as a prompt to apply
extra precautions — additional supervision, PPE checks, or task redesign —
before work begins, shifting safety management from reactive to proactive.

**Future improvements**, given more time or data:
- Incorporate near-miss reports (not just realized incidents) to catch more
  preventive signal.
- Add more granular environmental/task detail if available.
- Collect additional data to further balance the Fatal/Nonfatal ratio.

## Files in This Project
- `train_model.py` — full pipeline: load, clean, train, compare, evaluate, save
- `app.py` — Streamlit web app that loads the saved model for live predictions
- `model.pkl` — trained Random Forest model
- `encoders.pkl` — label encoders (converts text categories to numbers)
- `dropdown_options.pkl` — valid dropdown values used in the app
- `cleaned_data.csv` — the cleaned dataset used for training
- `confusion_matrix.png` — evaluation chart for the presentation
- `requirements.txt` — Python packages needed to run this project

## How to Run
```bash
pip install -r requirements.txt
python train_model.py    # optional: retrains the model from scratch
streamlit run app.py     # launches the interactive web app
```

## How to Deploy Live (required by supervisor)
1. Push this folder to a GitHub repository.
2. Go to share.streamlit.io, sign in, and connect the repository.
3. Select `app.py` as the entry point and deploy.
4. Streamlit will provide a public URL (e.g., yourapp.streamlit.app).

## Group Members
[Add all group member full names here per submission requirements]
