# Construction Market & House Price Forecasting using PyTorch

An end-to-end Deep Learning pipeline designed to predict housing market prices based on multi-dimensional structural and economic attributes. This repository contains a Multilayer Perceptron (MLP) architecture implemented in **PyTorch**, featuring automated data preprocessing, feature scaling, and regularization techniques to prevent overfitting.

## 📊 Dataset Overview
The project processes 79 distinct predictive attributes from the Ames Housing framework, structural characteristics, and environmental configurations:
* **Structural Specs:** Dwelling type (`MSSubClass`), total living area (`GrLivArea`), and total rooms (`TotRmsAbvGrd`).
* **Quality Metrics:** Overall material finish (`OverallQual`) and heating quality (`HeatingQC`).
* **Zoning & Location:** Physical locations within Ames city limits (`Neighborhood`).

## 🛠️ Architecture & Methods
* **Core Framework:** PyTorch 2.x
* **Preprocessing Pipeline:** Handled with Pandas and Scikit-Learn. Categorical variables are converted via One-Hot Encoding, missing values are imputed, and all features undergo rigorous standardization (`StandardScaler`).
* **Neural Network Topology:**
  * Multi-layer Perceptron (MLP) with fully-connected layers.
  * **Batch Normalization** applied post-activation to stabilize and accelerate gradient descent.
  * **Dropout Regularization (20-30%)** to handle high-dimensional tabular noise.
  * L2 Regularization (Weight Decay of \(10^{-4}\)) configured via **Adam Optimizer** (\(lr = 0.001\)).

## 📈 Evaluation Results & Analysis
The pipeline utilizes an academic Train/Validation/Test split (85% development data with an internal 17.6% validation holdout, and 15% completely unseen test data) to ensure zero data leakage.

### Performance Metrics:
* **Mean Absolute Error (MAE):** \$18,383.23
* **R² Score (Coefficient of Determination):** 0.8799

### Training and Prediction Profiles
Below are the generated training loss convergence curves and the target distribution scatter plot:

![Model Results](model_results.png)

### Deep Technical Insight:
1. **Loss Curves Analysis:** The training profiles show an optimal exponential decay of MSE Loss on both training and validation sets within the first 15 epochs. Post epoch 20, the `Validation Loss` establishes a stable plateau in the 0.18–0.20 range without any upward divergence. This lack of generalization degradation proves that the combination of Batch Normalization, Dropout, and Weight Decay effectively constrained model capacity and successfully neutralized overfitting.
2. **Distribution Profile:** The prediction scatter plot demonstrates a tight density alignment along the \(y = x\) ideal reference line. The model achieves maximum predictive precision within the high-density market segment (\$100k–\$350k). A slight underestimation bias is visible only in the extreme premium tail (above \$500k), which represents a classic and expected deep learning boundary behavior due to the sparse representation of luxury properties in the training distribution.

---

## 🧠 Relevance to Biomedical AI (Transferable Skills)

**Why this pipeline demonstrates essential competency for Computational Biomedicine:**

The exact mathematical, architectural, and data engineering frameworks validated in this tabular forecasting repository transfer seamlessly to critical pipelines in Biomedical AI:

1. **High-Dimensional Tabular Diagnostics:** Patient registry datasets, epidemiological cohorts, and Electronic Health Records (EHR) share the exact structure of this dataset—sparse, mixed categorical-continuous tables with high missingness. This project demonstrates the ability to safely process and clean such data without introducing leakage.
2. **Deep Regularization for Low-Sample Environments:** Clinical trials and specialized biological screens often suffer from limited samples and severe overfitting risks. The defensive implementation of Batch Normalization, Dropout, and Weight Decay in this architecture is directly applicable to training stable clinical regression models.
3. **Biomarker and Pharmacokinetic Modeling:** Predicting a continuous market value evaluates the same foundational multi-variable regression mechanics used to quantify patient biomarkers, predict drug-target binding affinity concentrations, or model continuous physiological signal modulations.
