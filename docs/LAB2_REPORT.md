# LABORATORY EXPERIMENT 2
## Baseline Model Development and Comprehensive Evaluation

---

### 1. AIM
To construct, train, evaluate, and compare multiple candidate machine learning models on the Amazon Bestselling Books dataset, establish performance baselines, compute classification metrics, analyze misclassifications, and identify the optimal production champion model.

---

### 2. THEORY & BACKGROUND
Model selection requires empirical comparison across diverse inductive biases to establish the performance ceiling and variance trade-offs. The Amazon bestseller genre prediction is formulated as binary classification:
$$y \in \{0 (\text{Fiction}), 1 (\text{Non Fiction})\}$$
Input vector: $\mathbf{x} = [\text{User Rating}, \text{Reviews\_Log}, \text{Price}, \text{Year}] \in \mathbb{R}^4$.

The candidate algorithms evaluated include:
1. **Logistic Regression**: Linear decision boundary optimized via cross-entropy loss with L2 regularization.
2. **K-Nearest Neighbors (KNN)**: Non-parametric distance-based instance classifier.
3. **Decision Tree Classifier**: Greedy partitioning using Gini impurity.
4. **Random Forest Classifier**: Ensemble of bootstrap-aggregated decision trees with randomized feature subsets, reducing variance and overfitting.
5. **Support Vector Classifier (SVC)**: Maximum-margin hyperplane classification with RBF kernel.

Evaluation metrics utilized:
- **Accuracy**: Overall fraction of correct predictions: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision (Weighted)**: Proportion of positive identifications that were correct across classes: $\frac{TP}{TP + FP}$.
- **Recall (Weighted)**: Proportion of actual positives correctly identified: $\frac{TP}{TP + FN}$.
- **F1-Score (Weighted)**: Harmonic mean of precision and recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **ROC-AUC**: Area under receiver operating characteristic curve, measuring class separation capability across all decision thresholds.

---

### 3. SOFTWARE & SYSTEM SPECIFICATIONS
- **Language**: Python 3.12.6 AMD64
- **Environment**: Local Sandbox
- **Evaluation Framework**: scikit-learn metrics suite

---

### 4. LIBRARIES & DEPENDENCIES
- `scikit-learn` (>=1.3.0): Model implementations, stratified cross-validation, and metrics.
- `numpy` & `pandas`: Numerical evaluation and tabular results formatting.
- `matplotlib` & `seaborn`: Confusion matrix heatmap and ROC curve visualization.

---

### 5. IMPLEMENTATION DETAILS
1. **Stratified 5-Fold Cross-Validation**:
   Evaluated all candidate models on the 440 training samples using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. **Model Training**:
   Fitted each classifier on the standardized training partition.
3. **Metric Calculation**:
   Predicted labels and probabilities on the 110 unseen test samples. Computed weighted Precision, Recall, F1, Accuracy, and ROC-AUC.
4. **Error Analysis**:
   Constructed `error_analysis.csv` to isolate false positives and false negatives, examining publication titles and attributes associated with classification errors.

---

### 6. CODE FILES & SCRIPTS
- `src/train.py`: Baseline model training implementation.
- `src/evaluate.py`: Test set evaluation and error export generation.
- `src/mlops_pipeline/05_train_registry.py`: Production candidate training and registration.
- `src/mlops_pipeline/06_model_evaluation.py`: Comprehensive metrics and confusion matrix visualization.

---

### 7. EXECUTION COMMANDS
```bash
python src/train.py
python src/evaluate.py
python src/mlops_pipeline/06_model_evaluation.py
```

---

### 8. ACTUAL OUTPUTS & METRICS
Comparative test partition performance:

| Model Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.7909** | **0.7915** | **0.7909** | **0.7859** | **0.8740** |
| Decision Tree | 0.7455 | 0.7481 | 0.7455 | 0.7466 | 0.7433 |
| K-Nearest Neighbors | 0.7091 | 0.7103 | 0.7091 | 0.7011 | 0.7788 |
| Support Vector Machine (SVC) | 0.7182 | 0.7301 | 0.7182 | 0.7056 | 0.7984 |
| Logistic Regression | 0.6091 | 0.6123 | 0.6091 | 0.5970 | 0.6558 |

#### Confusion Matrix (Random Forest on Test Partition, $N=110$):
- True Fiction (0, Correct): 35
- False Non Fiction (Predicted 1, Actual 0): 13
- True Non Fiction (1, Correct): 52
- False Fiction (Predicted 0, Actual 1): 10

---

### 9. VERIFICATION & VALIDATION
1. **Quality Gate Compliance**: Random Forest achieved F1-weighted of `0.7859`, significantly surpassing the required minimum threshold of `0.70`.
2. **Artifact Verification**: Confirmed generated plots in `artifacts/plots/confusion_matrix.png` and `artifacts/plots/roc_curve.png`.
3. **Error Analysis Integrity**: Checked `outputs/error_analysis.csv` contains exactly 23 misclassified books out of 110 samples ($23/110 = 20.91\%$ error, matching $79.09\%$ accuracy).

---

### 10. RESULT & CONCLUSION
Random Forest was established as the clear top-performing baseline model, delivering 79.09% accuracy and 0.8740 ROC-AUC. It successfully captures non-linear price and review dynamics in bestseller classifications without overfitting.

