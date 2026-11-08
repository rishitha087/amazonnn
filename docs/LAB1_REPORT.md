# LABORATORY EXPERIMENT 1
## ML Project Initialization and Dataset Preparation

---

### 1. AIM
To initialize an enterprise-grade Machine Learning Operations (MLOps) workspace, configure reproducible environment parameters, perform comprehensive exploratory data analysis (EDA), engineer genre-specific numerical features, and establish deterministic data preparation and splitting pipelines for the Amazon Top-50 Bestselling Books dataset.

---

### 2. THEORY & BACKGROUND
Machine learning systems deployed in production depend fundamentally on high-quality, reproducible data preparation pipelines. The dataset used in this project comprises the Amazon Top 50 Bestselling Books spanning 2009 to 2019, containing 550 records and 7 attributes:
- `Name`: Title of the bestselling publication.
- `Author`: Author(s) of the book.
- `User Rating`: Customer evaluation score spanning 0.0 to 5.0.
- `Reviews`: Total customer reviews count (highly right-skewed).
- `Price`: Retail price of the publication in USD.
- `Year`: Year the book achieved top-50 bestseller ranking (2009–2019).
- `Genre`: Classification target (`Fiction` vs `Non Fiction`).

Data preparation for this tabular binary classification task requires:
1. **Handling Logarithmic Distribution**: Review counts span from 37 to 87,841. Applying logarithmic transformation $\text{Reviews\_Log} = \ln(1 + \text{Reviews})$ compresses skewness and stabilizes variance for gradient-based and distance-based estimators.
2. **Deterministic Partitioning**: Splitting the dataset with a stratified 80:20 ratio preserves class distribution (56% Non Fiction, 44% Fiction) and prevents data leakage across training and test partitions.
3. **Data Leakage Prevention**: Standardization parameters ($\mu, \sigma$) are computed strictly from the training partition and subsequently applied to transform the test partition.

---

### 3. SOFTWARE & SYSTEM SPECIFICATIONS
- **Operating System**: Windows 11 Enterprise (64-bit)
- **Runtime Environment**: Python 3.12.6 AMD64
- **Project Location**: `C:\Users\rishi\Desktop\mlopssprooo\amazooonnnn`
- **Configuration Management**: YAML (`config.yaml`)

---

### 4. LIBRARIES & DEPENDENCIES
- `pandas` (>=2.0.0): Structured data manipulation and inspection.
- `numpy` (>=1.24.0): Vectorized transformations and numerical array storage.
- `scikit-learn` (>=1.3.0): Data splitting, scaling transformers (`StandardScaler`), and ColumnTransformer.
- `joblib` (>=1.3.0): Transformer serialization and model persistence.
- `pyyaml` (>=6.0): Centralized configuration parsing.

---

### 5. IMPLEMENTATION DETAILS
1. **Raw Data Ingestion**:
   Loaded `data/raw/amazon_bestsellers.csv` and confirmed 550 rows and 7 columns with zero missing values across all features.
2. **Target Label Encoding**:
   Mapped categorical genre values to binary integers:
   $$\text{Genre\_Encoded} = \begin{cases} 0 & \text{if Fiction} \\ 1 & \text{if Non Fiction} \end{cases}$$
3. **Feature Transformation**:
   Engineered `Reviews_Log` column using `np.log1p(df['Reviews'])`.
4. **Stratified Splitting**:
   Partitioned 550 samples into 440 training samples (80%) and 110 testing samples (20%) using random seed `42` and target stratification.
5. **Standardization**:
   Fitted `StandardScaler` on `[User Rating, Reviews_Log, Price, Year]` from training set. Applied transformation to generate `X_train_final.npy` and `X_test_final.npy`.
6. **Artifact Persistence**:
   Serialized `scaler.pkl`, `preprocessor.pkl`, `dataset_metadata.json`, and raw train/test CSVs.

---

### 6. CODE FILES & SCRIPTS
- `src/mlops_pipeline/01_load_data.py`: Raw dataset loading, exploratory summary, and load reporting.
- `src/mlops_pipeline/02_validate_data.py`: Schema validation, datatype verification, and bound checks.
- `src/mlops_pipeline/03_preprocessing.py`: Feature engineering, train/test splitting, scaling, and artifact serialization.
- `src/mlops_pipeline/04_validate_outputs.py`: Shape, dimension, and NaN validation of preprocessed arrays.

---

### 7. EXECUTION COMMANDS
```bash
python src/mlops_pipeline/01_load_data.py
python src/mlops_pipeline/02_validate_data.py
python src/mlops_pipeline/03_preprocessing.py
python src/mlops_pipeline/04_validate_outputs.py
```

---

### 8. ACTUAL OUTPUTS & METRICS
```text
============================================================
STAGE 1: AMAZON BESTSELLERS DATA LOADING & EDA
============================================================
[INFO] Loaded Amazon Bestsellers dataset: amazon_bestsellers.csv
[INFO] Dataset shape: (550, 7)
[INFO] Columns: ['Name', 'Author', 'User Rating', 'Reviews', 'Price', 'Year', 'Genre']
[INFO] Missing values: None
[INFO] Genre distribution: {'Non Fiction': 310, 'Fiction': 240}
[SUCCESS] Data load report saved to artifacts/data_load_report.json

============================================================
STAGE 2: RAW DATA SCHEMA & QUALITY VALIDATION
============================================================
[SUCCESS] Schema validation passed! All 7 columns valid across 550 records.

============================================================
STAGE 3: FEATURE ENGINEERING, SCALING & PREPROCESSING
============================================================
[SUCCESS] Preprocessing completed. Train: (440, 4), Test: (110, 4)

============================================================
STAGE 4: VALIDATING PREPROCESSED ARRAYS & OUTPUTS
============================================================
[SUCCESS] Processed arrays validated: X_train=(440, 4), X_test=(110, 4)
Validation report saved to: artifacts/output_validation_report.json
```

---

### 9. VERIFICATION & VALIDATION
1. **Dimension Conformance**: Verified `X_train_final.npy` has shape `(440, 4)` and `X_test_final.npy` has shape `(110, 4)`.
2. **Integrity Checks**: Confirmed 0 NaN values, 0 infinite values, and strictly valid binary labels $\{0, 1\}$.
3. **Deterministic Output**: Re-running preprocessing produces bit-for-bit identical NumPy arrays when using seed 42.

---

### 10. RESULT & CONCLUSION
Lab 1 successfully established an automated, reproducible data ingestion and preprocessing pipeline for the Amazon Bestseller dataset. Raw data was validated, feature-engineered, scaled, and persisted as production-ready NumPy and joblib artifacts with complete metadata traceability.

