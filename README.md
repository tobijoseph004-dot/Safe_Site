# Safe_Site
# 🦺 Construction Workplace Risk Assessment Using Machine Learning

A machine learning and Streamlit application designed to help HSE professionals assess the **potential severity risk of workplace incidents** in construction settings.

The model learns from historical **Fatal and Nonfatal OSHA construction incident records** and uses event, environmental, human, and task-related factors to assess risk.

## 🚦 Risk Assessment

- 🔴 **High Risk** — Fatal probability ≥70%
- 🟠 **Medium Risk** — Fatal probability 40–69%
- 🟢 **Low Risk** — Fatal probability - <40%
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

## 👥 Team

Oluwatobiloba • Abdul • Opeyemi
