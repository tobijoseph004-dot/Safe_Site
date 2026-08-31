# Safe_Site

# 🦺 Construction Workplace Risk Assessment Using Machine Learning

A machine learning and Streamlit application designed to help HSE professionals assess the **potential severity risk of workplace incidents** in construction settings.

The model learns from historical **Fatal and Nonfatal OSHA construction incident records** and uses event, environmental, human, and task-related factors to assess risk.

## 🚦 Risk Assessment

- 🔴 **High Risk** — Fatal probability ≥70%
- 🟠 **Medium Risk** — Fatal probability 40–69%
- 🟢 **Low Risk** — Fatal probability <40%

The application also provides **context-specific safety recommendations** based on the selected incident factors.

## 🤖 Model

**Tuned Random Forest Classifier**

- Accuracy: **67.1%**
- Macro F1: **0.656**

## 🛠️ Technologies

Python • Pandas • Scikit-learn • Matplotlib • Streamlit

## 📊 Dataset

**4,847 OSHA construction incident records (2015–2017)**

> This tool is a **decision-support system**, not a replacement for professional HSE judgment.

## 📸 Application Preview

### Risk Assessment Interface

![Risk Assessment Interface](screenshots/risk_assessment.png)

### Risk Prediction

![Risk Prediction](screenshots/risk_prediction.png)

### Safety Recommendations

![Safety Recommendations](screenshots/recommendations.png)

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Safe_Site.git
cd Safe_Site
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## 📊 Project Presentation

For a detailed explanation of the **dataset, exploratory analysis, model development, evaluation, findings, and recommendations**, see the project presentation.

[📥 View Project Presentation](presentation/Construction_Safety_Project.pptx)

## 👥 Team

**Oluwatobiloba • Abdul • Opeyemi**
