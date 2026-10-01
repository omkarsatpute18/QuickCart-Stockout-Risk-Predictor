import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

df = pd.read_csv("Quickcart_Warehouse_Inventory_Stockout_Risk_Predictor.csv")


df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"]).copy()

df["day_of_month"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

X = df.drop(columns=["stockout_risk"])
y = df["stockout_risk"]

drop_columns = [
    "stockout_risk",
    "date",
    "store_id",
    "sku_id",
    "supplier_id"
]

X = df.drop(columns=drop_columns)
y = df["stockout_risk"]

# Training and testing dataset using timestamp comparisons.
train_df = df[df["date"] <= pd.Timestamp("2026-10-23")].copy()
test_df = df[df["date"] >= pd.Timestamp("2026-10-24")].copy()



X_train = train_df.drop(columns=drop_columns)
y_train = train_df["stockout_risk"]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df["stockout_risk"]



# print("Training shape:", X_train.shape)
# print("Testing shape :", X_test.shape)

# print("\nTraining target:")
# print(y_train.value_counts())

# print("\nTesting target:")
# print(y_test.value_counts())


"""
Our Dataset's categorical and numerical values are already converted to its proper format so
do not need to perform action like onehotencoding
"""

#MODEL 1
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)


def evaluate_model(model_name, y_true, y_pred):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    imminent_recall = recall_score(
        y_true,
        y_pred,
        labels=["Imminent"],
        average=None,
        zero_division=0
    )[0]

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Macro Precision    : {precision:.4f}")
    print(f"Macro Recall       : {recall:.4f}")
    print(f"Macro F1           : {f1:.4f}")
    print(f"Imminent Recall    : {imminent_recall:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Imminent Recall": imminent_recall
    }

# baseline_results = evaluate_model(
#     "Majority Classifier",
#     y_test,
#     y_pred_baseline
# )



random_forest_model = Pipeline(
    steps=[
        
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

y_pred_rf = random_forest_model.predict(
    X_test
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf
)


categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns

print("Categorical columns:")
print(categorical_columns.tolist())