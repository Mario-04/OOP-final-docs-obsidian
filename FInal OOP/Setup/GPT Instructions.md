# Obsidian Project Setup - AutoML Assignment

## Overview
This Obsidian setup is designed to help you document and track the development of your AutoML assignment. Use the following sections to navigate through terminology, requirements, and instructions. You can add new notes for detailed content as you progress.

---

# Setup Instructions

## Tags
- **Tags to use**:
  - `#AutoML`
  - `#Artifact`
  - `#Dataset`
  - `#Metric`
  - `#Model`
  - `#Pipeline`
  - `#Feature`
  - `#ObsidianSetup`

---

# Terminology and Definitions

Each term below links to its dedicated note. Create separate notes with each term’s definition and examples.

- [[AutoML]]: Industry software or platform used to help train models without coding pipelines.
- [[Artifact]]: An abstract object referring to a stored asset, such as datasets or models.
- [[Metric]]: A function mapping predictions to ground truth, returning a real number.
- [[Dataset]]: A tabular data artifact split into training and test sets.
- [[Model]]: A function mapping input features to a target feature, including both parameters and hyperparameters.
- [[Feature]]: A measurable property (column in CSV) labeled as either `categorical` or `numerical`.
- [[Pipeline]]: A state machine that orchestrates stages like preprocessing, splitting, training, and evaluation.
- [[Adult Dataset]]: This is the dataset used to train the model on.

**Related Tags**: `#Terminology`, `#Definitions`

---

# Part 0: Initial Setup

### Steps
1. **Activate Environment**:
    ```bash
    conda activate <your env>
    pip install -r requirements.txt
    ```
2. **Run the Streamlit App**:
    ```bash
    python -m streamlit run app/Welcome.py
    ```
3. **Run Tests**:
    ```bash
    python -m autoop.tests.main
    ```

**Backlinks**: [[AutoML]], [[Streamlit]], [[Tests]]

---

# Part I: Core Library Requirements

- [x] **ML/Detect-Features**:
   - [x] Implement `detect_feature_types` in `autoop.functional.feature.detect_feature_types`.
- [x] **Artifact Class**:
   - [x] Implement in `autoop.core.ml.artifact`.
- [x] **Feature Class**:
   - [x] Implement in `autoop.core.ml.feature`.
- [x] **Metric Class**:
   - [x] Implement the metric class in `autoop.core.ml.metric` with the `__call__` method.
- [x] **Metric Extensions**:
   - [x] Add at least 6 metrics in `ML/metric/extensions`.
   - [x] Ensure at least 3 metrics are suitable for classification.
   - [x] Implement **Accuracy** metric for classification.
   - [x] Implement **Mean Squared Error** metric for regression.
   - [x] Do not use facades/wrappers for metric implementation; use libraries such as `numpy`.
- [ ] **Base Model Class**:
   - [ ] Implement the base model class in `autoop.core.ml.model`.
- [ ] **Model Extensions**:
   - [ ] Implement at least 3 classification models in `ML/model/extensions`.
   - [ ] Implement at least 3 regression models in `ML/model/extensions`.
   - [ ] You may use the facade pattern or wrappers on existing libraries for these models.
- [ ] **Pipeline Evaluation**:
   - [ ] Extend and modify the `execute` function in `ML/pipeline/evaluation` to return metrics for both the **evaluation** and **training sets**.


**Backlinks**: [[Artifact]], [[Feature]], [[Metric]], [[Model]], [[Pipeline]], [[Dataset]], [[Pydantic]], [[Tests]]

**Tags**: `#Requirements`, `#CoreLibrary`

---

# Part II: Streamlit App

1. **ST/Page/Datasets**:
   - Manage datasets on this page.
2. **ST/Datasets/Management/Create**:
   - Upload and convert CSV datasets using the `from_dataframe` method.
3. **ST/Modelling/Pipeline**:
   - Configure pipeline by selecting dataset split, input features, and metrics.

… *List other pages and options.*

**Backlinks**: [[Streamlit]], [[Pipeline]], [[Dataset]], [[Feature]]

**Tags**: `#Streamlit`, `#App`

---

# Part III: Bonus Additions

Explore ideas from the README file for bonus points. Track these ideas here and in linked notes as you develop them.

**Tags**: `#Bonus`, `#Ideas`

---

# Glossary

Add glossary terms and definitions in individual notes for easy reference across this vault.

---

# Final Thoughts

As you document each part of the project, link relevant terms to their definition notes, and use tags to organize your progress. You can also expand this document as you add more complex components or automation.

