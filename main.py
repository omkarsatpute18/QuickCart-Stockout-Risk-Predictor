# ============================================================
# QUICKCART STOCKOUT RISK PREDICTION
# ============================================================
#
# BUSINESS PROBLEM
# ----------------
# Predict whether a SKU in a particular store on a particular
# day is:
#
#       Safe
#       At-Risk
#       Imminent
#
# PROJECT TYPE
# ------------
# Supervised Multi-Class Classification
#
# DATASET
# -------
# 12 Stores
# 60 SKUs
# 15 Suppliers
# 30 Days
# 21,600 Inventory Records
#
# MODELS
# ------
# 1. Majority Classifier
# 2. Multinomial Logistic Regression
# 3. Random Forest
#
# MAIN METRIC
# -----------
# Imminent Recall
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Change this path according to your folder structure.

DATA_PATH = r"C:\Users\omkars\Desktop\data science\QuickCart Warehouse Inventory Stockout Risk Predictor"

STORES_FILE = f"{DATA_PATH}/dim_stores.csv"
SKUS_FILE = f"{DATA_PATH}/dim_skus.csv"
SUPPLIERS_FILE = f"{DATA_PATH}/dim_suppliers.csv"
EVENTS_FILE = f"{DATA_PATH}/dim_events.csv"
INVENTORY_FILE = f"{DATA_PATH}/fact_inventory_daily.csv"


# ============================================================
# 3. LOAD THE FIVE TABLES
# ============================================================
#
# The project contains:
#
# dim_stores
# dim_skus
# dim_suppliers
# dim_events
# fact_inventory_daily
#
# The fact table is the main table.
# The other tables provide additional information.
#
# ============================================================

stores = pd.read_csv(
    STORES_FILE,
    na_values=["N/A", "missing", "--", "NA", "null"],
    keep_default_na=True
)

skus = pd.read_csv(
    SKUS_FILE,
    na_values=["N/A", "missing", "--", "NA", "null"],
    keep_default_na=True
)

suppliers = pd.read_csv(
    SUPPLIERS_FILE,
    na_values=["N/A", "missing", "--", "NA", "null"],
    keep_default_na=True
)

events = pd.read_csv(
    EVENTS_FILE,
    na_values=["N/A", "missing", "--", "NA", "null"],
    keep_default_na=True
)

inventory = pd.read_csv(
    INVENTORY_FILE,
    na_values=["N/A", "missing", "--", "NA", "null"],
    keep_default_na=True
)


print("\n================ DATA LOADED ================\n")

print("Stores:", stores.shape)
print("SKUs:", skus.shape)
print("Suppliers:", suppliers.shape)
print("Events:", events.shape)
print("Inventory:", inventory.shape)


# ============================================================
# 4. BASIC DATA VALIDATION
# ============================================================
#
# Expected:
#
# Stores       = 12
# SKUs         = 60
# Suppliers    = 15
# Events       = 30
# Inventory    = 21,600
#
# ============================================================

print("\n================ EXPECTED ROW COUNTS ================\n")

print("Stores expected:", 12)
print("Stores actual  :", len(stores))

print("SKUs expected:", 60)
print("SKUs actual  :", len(skus))

print("Suppliers expected:", 15)
print("Suppliers actual  :", len(suppliers))

print("Events expected:", 30)
print("Events actual  :", len(events))

print("Inventory expected:", 21600)
print("Inventory actual  :", len(inventory))


# ============================================================
# 5. CHECK DATE COLUMN
# ============================================================

inventory["date"] = pd.to_datetime(
    inventory["date"],
    errors="coerce"
)

events["date"] = pd.to_datetime(
    events["date"],
    errors="coerce"
)

print("\nDate range:")
print(inventory["date"].min())
print(inventory["date"].max())

print("\nUnique dates:")
print(inventory["date"].nunique())


# ============================================================
# 6. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n================ TARGET DISTRIBUTION ================\n")

target_counts = inventory["stockout_risk"].value_counts()

print(target_counts)

target_percent = (
    inventory["stockout_risk"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nTarget percentages:")
print(target_percent.round(2))


# ============================================================
# 7. DATA QUALITY CHECK
# ============================================================

print("\n================ MISSING VALUES ================\n")

print(
    inventory.isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)


# ============================================================
# 8. CLEAN CITY DISPLAY
# ============================================================
#
# The project specification mentions inconsistent casing
# in city_display.
#
# Example:
#
# Pune
# PUNE
# pune
#
# We standardize it.
#
# ============================================================

if "city_display" in stores.columns:

    stores["city_display"] = (
        stores["city_display"]
        .astype("string")
        .str.strip()
        .str.title()
    )


# ============================================================
# 9. CLEAN SUPPLIER RELIABILITY
# ============================================================
#
# Some reliability values are "N/A".
#
# Convert them to numeric.
#
# Non-numeric values become NaN.
#
# ============================================================

suppliers["reliability_score"] = pd.to_numeric(
    suppliers["reliability_score"],
    errors="coerce"
)

print("\nSupplier reliability missing values:")
print(
    suppliers["reliability_score"]
    .isna()
    .sum()
)


# ============================================================
# 10. IMPUTE SUPPLIER RELIABILITY
# ============================================================
#
# The project specification suggests using the category
# median.
#
# Because suppliers can supply one or more categories,
# we use the median reliability across suppliers that share
# the supplied-category information where practical.
#
# For a simple and robust implementation, we use the
# overall median if category-level imputation is not possible.
#
# ============================================================

supplier_median = suppliers["reliability_score"].median()

suppliers["reliability_score"] = (
    suppliers["reliability_score"]
    .fillna(supplier_median)
)


# ============================================================
# 11. MERGE THE TABLES
# ============================================================
#
# Start with the inventory fact table.
#
# Then attach:
#
# stores       → store_id
# skus         → sku_id
# suppliers    → supplier_id
# events       → date
#
# IMPORTANT:
# The final row count should remain 21,600.
#
# ============================================================

df = inventory.copy()


# -----------------------------
# Merge stores
# -----------------------------

df = df.merge(
    stores,
    on="store_id",
    how="left",
    validate="many_to_one"
)


# -----------------------------
# Merge SKUs
# -----------------------------

df = df.merge(
    skus,
    on="sku_id",
    how="left",
    validate="many_to_one",
    suffixes=("", "_sku")
)


# -----------------------------
# Merge suppliers
# -----------------------------

df = df.merge(
    suppliers,
    on="supplier_id",
    how="left",
    validate="many_to_one",
    suffixes=("", "_supplier")
)


# -----------------------------
# Merge events
# -----------------------------

df = df.merge(
    events,
    on="date",
    how="left",
    validate="many_to_one",
    suffixes=("", "_event")
)


# ============================================================
# 12. VERIFY MERGE
# ============================================================

print("\n================ AFTER MERGING ================\n")

print("Final shape:", df.shape)

print("Expected rows:", 21600)
print("Actual rows  :", len(df))

assert len(df) == 21600, \
    "ERROR: Row count changed after merging!"


# ============================================================
# 13. FEATURE ENGINEERING
# ============================================================
#
# This is one of the most important parts of the project.
#
# Raw variables are converted into business-meaningful
# predictive features.
#
# ============================================================


# ------------------------------------------------------------
# 13.1 Days of Cover
# ------------------------------------------------------------
#
# Formula from project specification:
#
# closing_stock / sales_velocity_7d
#
# Meaning:
# How many days current stock can theoretically support
# the observed sales velocity.
#
# ------------------------------------------------------------

df["days_of_cover"] = (
    df["closing_stock"] /
    df["sales_velocity_7d"].replace(0, np.nan)
)


# ------------------------------------------------------------
# 13.2 Reorder Gap
# ------------------------------------------------------------
#
# Formula:
#
# reorder_point - closing_stock
#
# Positive:
# Current stock is below reorder point.
#
# Negative:
# Current stock is above reorder point.
#
# ------------------------------------------------------------

df["reorder_gap"] = (
    df["reorder_point"] -
    df["closing_stock"]
)


# ------------------------------------------------------------
# 13.3 Days of Cover Ratio
# ------------------------------------------------------------
#
# Formula:
#
# days_of_cover / expected lead time
#
# ------------------------------------------------------------

df["days_of_cover_ratio"] = (
    df["days_of_cover"] /
    df["lead_time_days_expected"].replace(0, np.nan)
)


# ============================================================
# 14. HISTORICAL REORDER FEATURE
# ============================================================
#
# IMPORTANT:
#
# We don't want today's reorder decision to predict today's
# stockout risk because that could create leakage.
#
# Instead we use the PREVIOUS day's reorder behavior.
#
# ============================================================

df = df.sort_values(
    ["store_id", "sku_id", "date"]
).reset_index(drop=True)


df["previous_reorder"] = (
    df.groupby(["store_id", "sku_id"])["reorder_placed"]
    .shift(1)
)


# Convert previous reorder to numeric indicator.

df["previous_reorder_flag"] = (
    df["previous_reorder"]
    .map({
        "Y": 1,
        "N": 0
    })
)


# ------------------------------------------------------------
# Recent reorder
# ------------------------------------------------------------
#
# Example:
# If a reorder happened recently, mark it as 1.
#
# Here we use previous reorder as a simple historical signal.
#
# ------------------------------------------------------------

df["is_recent_reorder"] = (
    df["previous_reorder_flag"]
    .fillna(0)
)


# ============================================================
# 15. TIME FEATURES
# ============================================================

df["day_of_month"] = df["date"].dt.day

df["day_of_week"] = df["date"].dt.dayofweek


# ============================================================
# 16. FESTIVAL TIME FEATURE
# ============================================================
#
# Project specification:
# Diwali Week = Oct 22–26, 2026
#
# We create a feature describing the position inside
# the festival period.
#
# -1 = outside festival period
#  0 = festival start
#  1 = second festival day
# ...
#
# ============================================================

festival_start = pd.Timestamp("2026-10-22")
festival_end = pd.Timestamp("2026-10-26")

df["days_since_festival_start"] = np.where(
    df["date"].between(
        festival_start,
        festival_end
    ),
    (
        df["date"] -
        festival_start
    ).dt.days,
    -1
)


# ============================================================
# 17. SUPPLIER RELIABILITY GROUP
# ============================================================

df["reliability_group"] = pd.cut(
    df["reliability_score"],
    bins=[-np.inf, 0.75, 0.85, np.inf],
    labels=[
        "<0.75",
        "0.75-0.85",
        ">=0.85"
    ],
    right=False
)


# ============================================================
# 18. FEATURE ENGINEERING CHECK
# ============================================================

print("\n================ ENGINEERED FEATURES ================\n")

engineered_features = [
    "days_of_cover",
    "reorder_gap",
    "days_of_cover_ratio",
    "previous_reorder",
    "is_recent_reorder",
    "day_of_month",
    "day_of_week",
    "days_since_festival_start",
    "reliability_group"
]

print(df[engineered_features].head())


# ============================================================
# 19. CHECK ENGINEERED FEATURE MISSING VALUES
# ============================================================

print("\nMissing values in engineered features:")

print(
    df[engineered_features]
    .isna()
    .sum()
)


# ============================================================
# 20. EXPLORATORY DATA ANALYSIS
# ============================================================


# ------------------------------------------------------------
# Target Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="stockout_risk"
)

plt.title("Stockout Risk Distribution")
plt.xlabel("Risk Class")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Festival vs Non-Festival
# ------------------------------------------------------------

festival_risk = pd.crosstab(
    df["is_festival_week"],
    df["stockout_risk"],
    normalize="index"
) * 100

print("\nFestival risk distribution:")
print(festival_risk.round(2))


# ------------------------------------------------------------
# Imminent rate by festival status
# ------------------------------------------------------------

festival_imminent = (
    df.groupby("is_festival_week")["stockout_risk"]
    .apply(lambda x: (x == "Imminent").mean() * 100)
)

print("\nImminent rate by festival status:")
print(festival_imminent)


# ============================================================
# 21. PERISHABLE ANALYSIS
# ============================================================

perishable_imminent = (
    df.groupby("is_perishable")["stockout_risk"]
    .apply(lambda x: (x == "Imminent").mean() * 100)
)

print("\nImminent rate by perishability:")
print(perishable_imminent)


# ============================================================
# 22. SUPPLIER RELIABILITY ANALYSIS
# ============================================================

supplier_reliability_analysis = (
    df.groupby("reliability_group",
               observed=False)["stockout_risk"]
    .apply(lambda x: (x == "Imminent").mean() * 100)
)

print("\nImminent rate by supplier reliability:")
print(supplier_reliability_analysis)


# ============================================================
# 23. CATEGORY ANALYSIS
# ============================================================

category_analysis = (
    df.groupby("category")["stockout_risk"]
    .apply(lambda x: (x == "Imminent").mean() * 100)
    .sort_values(ascending=False)
)

print("\nImminent rate by category:")
print(category_analysis)


# ============================================================
# 24. STORE ANALYSIS
# ============================================================

store_analysis = (
    df.groupby(
        ["store_id", "city"]
    )["stockout_risk"]
    .apply(lambda x: (x == "Imminent").mean() * 100)
    .sort_values(ascending=False)
)

print("\nImminent rate by store:")
print(store_analysis)


# ============================================================
# 25. RISK GROUP FEATURE ANALYSIS
# ============================================================

risk_group_means = (
    df.groupby("stockout_risk")[
        [
            "days_of_cover",
            "days_of_cover_ratio",
            "reorder_gap",
            "sales_velocity_7d",
            "closing_stock"
        ]
    ]
    .mean()
)

print("\nAverage features by risk group:")
print(risk_group_means)


# ============================================================
# 26. CHECK TARGET LEAKAGE
# ============================================================
#
# Target:
#
# stockout_risk
#
# Must NOT appear in X.
#
# We also remove identifiers because IDs generally shouldn't
# be treated as meaningful numerical predictors.
#
# ============================================================

TARGET = "stockout_risk"

DROP_COLUMNS = [
    TARGET,
    "date",
    "store_id",
    "sku_id",
    "supplier_id"
]


# ============================================================
# 27. TIME-BASED TRAIN/TEST SPLIT
# ============================================================
#
# Training:
# Oct 1–23
#
# Testing:
# Oct 24–30
#
# IMPORTANT:
# No random train_test_split here.
#
# ============================================================

TRAIN_END = pd.Timestamp("2026-10-23")
TEST_START = pd.Timestamp("2026-10-24")


train_df = df[
    df["date"] <= TRAIN_END
].copy()


test_df = df[
    df["date"] >= TEST_START
].copy()


print("\n================ TRAIN / TEST SPLIT ================\n")

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

print(
    "Training dates:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print(
    "Testing dates:",
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# ============================================================
# 28. CREATE X AND y
# ============================================================

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]


X_test = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = test_df[TARGET]


print("\nX_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("\ny_train distribution:")
print(y_train.value_counts())

print("\ny_test distribution:")
print(y_test.value_counts())


# ============================================================
# 29. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


categorical_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()


print("\n================ FEATURES ================\n")

print("Numeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 30. PREPROCESSING
# ============================================================
#
# NUMERIC:
#   Missing value → median
#   Scaling
#
# CATEGORICAL:
#   Missing value → most frequent
#   One-hot encoding
#
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 31. MODEL 1 — MAJORITY CLASSIFIER
# ============================================================

baseline_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]
)


baseline_model.fit(
    X_train,
    y_train
)


baseline_predictions = baseline_model.predict(
    X_test
)


# ============================================================
# 32. MODEL 2 — LOGISTIC REGRESSION
# ============================================================
#
# class_weight="balanced" helps the model pay attention
# to minority classes.
#
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = logistic_model.predict(
    X_test
)


# ============================================================
# 33. MODEL 3 — RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


rf_predictions = random_forest_model.predict(
    X_test
)


# ============================================================
# 34. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    macro_precision = precision_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    imminent_precision = precision_score(
        y_true,
        predictions,
        labels=["Imminent"],
        average="macro",
        zero_division=0
    )

    imminent_recall = recall_score(
        y_true,
        predictions,
        labels=["Imminent"],
        average="macro",
        zero_division=0
    )

    imminent_f1 = f1_score(
        y_true,
        predictions,
        labels=["Imminent"],
        average="macro",
        zero_division=0
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Imminent Precision": imminent_precision,
        "Imminent Recall": imminent_recall,
        "Imminent F1": imminent_f1
    }


# ============================================================
# 35. EVALUATE ALL MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "Majority Classifier",
        y_test,
        baseline_predictions
    )
)

results.append(
    evaluate_model(
        "Multinomial Logistic Regression",
        y_test,
        logistic_predictions
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions
    )
)


results_df = pd.DataFrame(results)


# Convert metrics to percentages

results_display = results_df.copy()

metric_columns = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Imminent Precision",
    "Imminent Recall",
    "Imminent F1"
]

results_display[metric_columns] = (
    results_display[metric_columns] * 100
)


print("\n================ MODEL COMPARISON ================\n")

print(
    results_display.round(2).to_string(
        index=False
    )
)


# ============================================================
# 36. CLASSIFICATION REPORTS
# ============================================================

print("\n================ LOGISTIC REGRESSION ================\n")

print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0
    )
)


print("\n================ RANDOM FOREST ================\n")

print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# ============================================================
# 37. CONFUSION MATRIX — LOGISTIC REGRESSION
# ============================================================

cm_logistic = confusion_matrix(
    y_test,
    logistic_predictions,
    labels=[
        "Safe",
        "At-Risk",
        "Imminent"
    ]
)


plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    xticklabels=[
        "Safe",
        "At-Risk",
        "Imminent"
    ],
    yticklabels=[
        "Safe",
        "At-Risk",
        "Imminent"
    ]
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 38. CONFUSION MATRIX — RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    rf_predictions,
    labels=[
        "Safe",
        "At-Risk",
        "Imminent"
    ]
)


plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    xticklabels=[
        "Safe",
        "At-Risk",
        "Imminent"
    ],
    yticklabels=[
        "Safe",
        "At-Risk",
        "Imminent"
    ]
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 39. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
#
# IMPORTANT:
#
# Random Forest sees transformed features after:
#
#   preprocessing
#       ↓
#   one-hot encoding
#
# Therefore, the number of feature names is larger than
# the number of original columns.
#
# ============================================================

rf_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)


rf_classifier = (
    random_forest_model
    .named_steps["classifier"]
)


feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)


feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance":
        rf_classifier.feature_importances_

})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n================ RANDOM FOREST FEATURE IMPORTANCE ================\n")

print(
    feature_importance_df
    .head(20)
)


# ============================================================
# 40. PLOT TOP RANDOM FOREST FEATURES
# ============================================================

top_features = (
    feature_importance_df
    .head(15)
    .sort_values(
        "Importance"
    )
)


plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top Random Forest Features"
)

plt.xlabel("Importance")

plt.tight_layout()
plt.show()


# ============================================================
# 41. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================
#
# Logistic Regression provides coefficients that can help
# interpret which transformed features are associated with
# each class.
#
# IMPORTANT:
#
# These are model associations, NOT causal effects.
#
# ============================================================

logistic_preprocessor = (
    logistic_model
    .named_steps["preprocessor"]
)


logistic_classifier = (
    logistic_model
    .named_steps["classifier"]
)


logistic_feature_names = (
    logistic_preprocessor
    .get_feature_names_out()
)


coefficient_matrix = (
    logistic_classifier
    .coef_
)


classes = (
    logistic_classifier
    .classes_
)


logistic_coefficients = pd.DataFrame(
    coefficient_matrix.T,
    index=logistic_feature_names,
    columns=classes
)


# ------------------------------------------------------------
# Imminent coefficients
# ------------------------------------------------------------

imminent_coefficients = (
    logistic_coefficients["Imminent"]
    .sort_values(ascending=False)
)


print(
    "\n================ POSITIVE IMMINENT COEFFICIENTS ================\n"
)

print(
    imminent_coefficients
    .head(15)
)


print(
    "\n================ NEGATIVE IMMINENT COEFFICIENTS ================\n"
)

print(
    imminent_coefficients
    .tail(15)
)


# ============================================================
# 42. BUSINESS INSIGHT SUMMARY
# ============================================================

print("\n================ BUSINESS INSIGHTS ================\n")


# Festival

festival_rates = (
    df.groupby("is_festival_week")["stockout_risk"]
    .apply(
        lambda x:
        (x == "Imminent").mean() * 100
    )
)

print("\nFestival effect:")
print(festival_rates.round(2))


# Perishable

perishable_rates = (
    df.groupby("is_perishable")["stockout_risk"]
    .apply(
        lambda x:
        (x == "Imminent").mean() * 100
    )
)

print("\nPerishable effect:")
print(perishable_rates.round(2))


# Category

print("\nCategory Imminent rates:")

print(
    category_analysis.round(2)
)


# Store

print("\nStore Imminent rates:")

print(
    store_analysis.round(2)
)


# ============================================================
# 43. FINAL MODEL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

for _, row in results_display.iterrows():

    print(
        f"\n{row['Model']}"
    )

    print(
        f"Accuracy: "
        f"{row['Accuracy']:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{row['Macro F1']:.2f}%"
    )

    print(
        f"Imminent Recall: "
        f"{row['Imminent Recall']:.2f}%"
    )

    print(
        f"Imminent Precision: "
        f"{row['Imminent Precision']:.2f}%"
    )


# ============================================================
# 44. FINAL PROJECT CONCLUSION
# ============================================================

print("""
============================================================
PROJECT CONCLUSION
============================================================

This project developed a supervised multi-class classification
system for predicting daily stockout risk for SKU-store
combinations.

The workflow included:

1. Loading five relational tables.
2. Validating the expected row counts.
3. Cleaning data-quality issues.
4. Joining the tables.
5. Engineering inventory and demand features.
6. Checking for potential data leakage.
7. Using a time-based train/test split.
8. Training a majority baseline.
9. Training Multinomial Logistic Regression.
10. Training Random Forest.
11. Evaluating Accuracy, Macro F1 and Imminent Recall.
12. Interpreting feature importance and model coefficients.
13. Performing business-level stockout risk analysis.

The key business objective was to identify Imminent stockout
situations early enough to support proactive inventory decisions.

The most important lesson from this project was that model
performance depends not only on the algorithm, but also on
correct data preparation, feature engineering, temporal
validation and leakage prevention.
""")


# ============================================================
# 45. OPTIONAL: SAVE RESULTS
# ============================================================

results_display.to_csv(
    "model_comparison_results.csv",
    index=False
)


feature_importance_df.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)


risk_group_means.to_csv(
    "risk_group_feature_means.csv"
)


print(
    "\nResults saved successfully."
)

# ============================================================
# END OF PROJECT
# ============================================================