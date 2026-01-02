# Titanic-Survival-Analysis
END-TO-END DATA ANALYSIS AND MACHINE LEARNING PROJECT ON THE TITANIC DATASET

#  Titanic Survival Analysis

This project is an end-to-end **data analysis and machine learning pipeline**
built on the Titanic dataset.  
The goal is to analyze the factors affecting passenger survival and to build
optimized classification models.

---

##  Dataset
- **train.csv**: 891 passengers with survival labels
- **test.csv**: 418 passengers without labels

---

##  Exploratory Data Analysis (EDA)

- Dataset structure and data types
- Missing value analysis
- Numerical statistics using `describe()`
- Categorical feature distributions
- Survival rate comparisons by:
  - Gender
  - Passenger class
  - Embarkation port
  - Cabin deck
- Correlation analysis with the target variable (**Survived**)

---

##  Data Preprocessing

### Missing Value Handling
- **Age** → filled with median
- **Cabin** → transformed into `CabinDeck`, missing values labeled as `U (Unknown)`
- **Embarked** → filled with mode
- **Fare** → filled with median

### Feature Engineering
- `YakinSayisi = SibSp + Parch`
- `FamilySize = YakinSayisi + 1`
- `CabinDeck` extracted from the first letter of the cabin
- Noise removal and data normalization
- Removal of high-noise features (`Name`, `Ticket`, `Cabin`)

---

##  Encoding

- **Label Encoding**
  - Sex: male → 0, female → 1
  - Embarked: S → 0, C → 1, Q → 2
  - CabinDeck: A–G, T, U → numeric values

- **One-Hot Encoding**
  - Applied to family size bins to avoid ordinal bias

---

##  Modeling

### Baseline Models
- Logistic Regression
- Random Forest

### Evaluation Metrics
- Accuracy
- F1-score
- ROC-AUC
- Confusion Matrix

---

##  Cross Validation
- 5-Fold Stratified Cross Validation
- Mean and standard deviation of evaluation metrics reported

---

##  Optimization

- Hyperparameter tuning using **RandomizedSearchCV**
- Optimization metric: **ROC-AUC**
- Threshold tuning to maximize **F1-score**

 **Performance Improvement**
- Accuracy: 0.79 → 0.80
- F1-score: 0.73 → 0.74
- ROC-AUC: 0.86 → 0.87

Final model: **Optimized Random Forest**

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

##  Author
**Gizem Fatma Kılıç**  
Software Engineering Student

