"""
=====================================================================
PREDICTING SEVERE OCCUPATIONAL INJURIES IN CONSTRUCTION WORKPLACES
=====================================================================
Project: Predicting Severe Occupational Injuries in Construction
         Workplaces Using Machine Learning

Objective: Develop a machine learning model to predict whether a
           workplace incident is likely to result in a Fatal or
           Nonfatal outcome, enabling HSE (Health & Safety)
           professionals to implement preventive measures and
           improve worker safety.

Dataset:   OSHA Construction Incident Data (2015-2017)
           Source: U.S. Occupational Safety and Health Administration
           4,847 real workplace incident records

This script documents the full pipeline:
  1. Load and inspect the data
  2. Clean the data (handle missing values, select features)
  3. Train and compare multiple classification models
  4. Evaluate using accuracy, precision, recall, F1, confusion matrix
  5. Save the final chosen model for use in the Streamlit app
=====================================================================
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# =====================================================================
# STEP 1: LOAD THE RAW DATA
# =====================================================================
# The raw file contains 4,847 incident records with 29 columns,
# including free-text descriptions, dates, and structured fields
# describing each incident (event type, environmental factors, etc.)

raw_df = pd.read_csv(
    r"C:\Users\olamiju\Downloads\OSHA_DATA.csv",
    low_memory=False
)
print(f"Raw data loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")


# =====================================================================
# STEP 2: SELECT FEATURES AND CLEAN THE DATA
# =====================================================================
# We deliberately choose only features that would be KNOWN BEFORE an
# incident occurs. This is critical for real-world usability: an HSE
# officer assessing a task in advance would know the event type,
# environmental conditions, human factors, and task assigned - but
# would NOT yet know the "Nature of Injury" or "Part of Body" affected,
# since those only exist AFTER an incident has already happened.
#
# We tested including "Nature of Injury" and "Part of Body" as an
# experiment: accuracy jumped from 67% to 85%. However, we rejected
# this approach because it constitutes DATA LEAKAGE - injury
# descriptions like "Electrocution" or "Amputation, Crushing" are
# near-synonyms for the outcome itself, not genuine predictive
# factors. Using them would make the model unusable for its intended
# preventive purpose, since that information isn't available before
# an incident occurs.

FEATURES = ['Event type', 'Environmental Factor', 'Human Factor', 'Task Assigned']
TARGET = 'Degree of Injury'

# We also checked 'Construction End Use', 'Project Type', and 'fall_ht'
# as candidate features but excluded them:
#   - fall_ht: 100% of values were 0 (not a usable feature)
#   - Construction End Use / Project Type: ~78-79% missing values
#     (too sparse to be a reliable predictor)

df = raw_df[FEATURES + [TARGET]].copy()

rows_before = len(df)
df = df.dropna()  # Remove the small number of rows (9) with missing values
rows_after = len(df)
print(f"Dropped {rows_before - rows_after} rows with missing values.")
print(f"Clean dataset: {rows_after} rows")

print("\nTarget variable distribution (Degree of Injury):")
print(df[TARGET].value_counts())
print(round(df[TARGET].value_counts(normalize=True) * 100, 1), "%")
# Fatal: ~61%, Nonfatal: ~39% - reasonably balanced, no need for
# aggressive class collapsing like an earlier dataset version required.

df.to_csv("cleaned_data.csv", index=False)


# =====================================================================
# STEP 2b: EXPLORATORY DATA VISUALIZATION
# =====================================================================
# Before modeling, we visualize the cleaned data to understand its
# structure and spot patterns. This produces data_exploration.png,
# showing: the Fatal/Nonfatal balance, the most common event types,
# the fatality rate by event type, and the most common human factors
# contributing to incidents.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

df['Degree of Injury'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#d62728', '#2ca02c'])
axes[0, 0].set_title('Fatal vs Nonfatal Incidents')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=0)

df['Event type'].value_counts().head(8).plot(kind='barh', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Top 8 Event Types')
axes[0, 1].invert_yaxis()

top_events = df['Event type'].value_counts().head(6).index
subset = df[df['Event type'].isin(top_events)]
fatal_rate = subset.groupby('Event type')['Degree of Injury'].apply(lambda x: (x == 'Fatal').mean() * 100).sort_values()
fatal_rate.plot(kind='barh', ax=axes[1, 0], color='indianred')
axes[1, 0].set_title('Fatal Rate (%) by Event Type (top 6 event types)')
axes[1, 0].set_xlabel('% Fatal')

df['Human Factor'].value_counts().head(8).plot(kind='barh', ax=axes[1, 1], color='darkorange')
axes[1, 1].set_title('Top 8 Human Factors')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig("data_exploration.png", dpi=150)
plt.close()
print("Saved data_exploration.png")


# =====================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =====================================================================
# Machine learning models need numbers, not text. We use LabelEncoder
# to convert each category (e.g., "Fall", "Slip", "Struck By") into a
# corresponding integer. We save these encoders so the Streamlit app
# can apply the SAME encoding to new user input later.

X = df[FEATURES].copy()
y = df[TARGET]

# Save the human-readable dropdown options for the Streamlit app
dropdown_options = {col: sorted(df[col].astype(str).unique().tolist()) for col in FEATURES}

encoders = {}
for col in FEATURES:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le


# =====================================================================
# STEP 4: TRAIN / TEST SPLIT
# =====================================================================
# We hold out 20% of the data as a test set the model never sees
# during training. This lets us fairly evaluate how well it
# generalizes to new, unseen incidents. `stratify=y` ensures the
# Fatal/Nonfatal ratio is preserved in both the train and test sets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")


# =====================================================================
# STEP 5: TRAIN AND COMPARE MULTIPLE MODELS
# =====================================================================
# Per the project guideline, we must justify our chosen technique
# against alternatives. We compare four classification approaches:
#
#   - Logistic Regression: a simple, interpretable linear baseline
#   - Random Forest: an ensemble of decision trees, handles
#     non-linear relationships between categorical features well
#   - Naive Bayes: a fast probabilistic baseline
#   - Gradient Boosting: a more powerful ensemble, builds trees
#     sequentially to correct previous errors
#
# We use `class_weight='balanced'` where available to give the
# minority class (Nonfatal, 39% of the data) fair weight during
# training, rather than letting the model default to the majority
# class to inflate accuracy.

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
    "Random Forest (tuned)": RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    ),
}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    results[name] = {"accuracy": acc, "macro_f1": f1_macro}
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f} | Macro F1: {f1_macro:.3f}")
    print(classification_report(y_test, preds, zero_division=0))


# =====================================================================
# STEP 6: CHOOSE THE FINAL MODEL
# =====================================================================
# We select based on MACRO F1, not raw accuracy. Macro F1 gives equal
# weight to both classes (Fatal and Nonfatal), whereas accuracy can be
# misleadingly inflated by a model that just favors the majority class.
#
# Gradient Boosting scored higher on raw accuracy (70.1%) but only
# caught 38% of actual Nonfatal cases (low recall) - it was
# essentially leaning on the majority class. For a safety application,
# a model that reliably misses one entire outcome type is a worse
# real-world tool than one with slightly lower accuracy but balanced
# performance across both outcomes.
#
# FINAL CHOICE: Random Forest (tuned)
#   Accuracy: 67.1%  |  Macro F1: 0.656
#   Fatal recall: 72%  |  Nonfatal recall: 59%

final_model_name = "Random Forest (tuned)"
final_model = models[final_model_name]
print(f"\n>>> FINAL MODEL SELECTED: {final_model_name} <<<")


# =====================================================================
# STEP 7: CONFUSION MATRIX (for the presentation)
# =====================================================================
final_preds = final_model.predict(X_test)
cm = confusion_matrix(y_test, final_preds, labels=["Fatal", "Nonfatal"])
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=["Actual Fatal", "Actual Nonfatal"],
                    columns=["Predicted Fatal", "Predicted Nonfatal"]))


# =====================================================================
# STEP 8: BUILD THE CONTEXT ENGINE (for guided/cascading dropdowns)
# =====================================================================
# Rather than showing the HSE officer every possible Environmental Factor
# and Human Factor in the entire dataset (many of which never actually
# occur together with a given event type), we build a lookup: for each
# Event Type, which Environmental Factors and Human Factors have
# historically been reported alongside it, ranked by frequency.
#
# This means the Streamlit app can filter its dropdowns dynamically based
# on the Event Type the officer selects first - a "context engine" that
# sits separately from the prediction model itself. The context engine
# does not predict anything; it only narrows the options shown to
# combinations that have actually occurred in the historical data.

context_map = {}
task_map = {}
for event in df["Event type"].unique():
    subset = df[df["Event type"] == event]
    context_map[event] = {
        "environmental_factors": subset["Environmental Factor"].value_counts().index.tolist(),
        "human_factors": subset["Human Factor"].value_counts().index.tolist(),
    }
    task_map[event] = subset["Task Assigned"].value_counts().index.tolist()

print("\nContext engine built: Environmental/Human factor options will be "
      "filtered per Event Type in the Streamlit app.")


# =====================================================================
# STEP 9: SAVE THE MODEL AND ALL SUPPORTING FILES FOR THE STREAMLIT APP
# =====================================================================
with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("dropdown_options.pkl", "wb") as f:
    pickle.dump(dropdown_options, f)
with open("context_map.pkl", "wb") as f:
    pickle.dump(context_map, f)
with open("task_map.pkl", "wb") as f:
    pickle.dump(task_map, f)

print("\nSaved: model.pkl, encoders.pkl, dropdown_options.pkl, "
      "context_map.pkl, task_map.pkl")
print("These are loaded directly by app.py to power the live, guided predictions.")
