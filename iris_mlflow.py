import os
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------
# Load Dataset
# --------------------------

iris = load_iris()

X = iris.data
y = iris.target


# --------------------------
# Split Dataset
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------
# MLflow Experiment
# --------------------------
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

mlflow.set_experiment("Iris_Classification")


# --------------------------
# Model Parameters
# --------------------------

n_estimators = 100
max_depth = 5


# --------------------------
# Start MLflow Run
# --------------------------

with mlflow.start_run():

    # Create Model

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train Model

    model.fit(X_train, y_train)

    # Prediction

    prediction = model.predict(X_test)

    # --------------------------
    # Calculate Metrics
    # --------------------------

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    # --------------------------
    # MLflow Logging
    # --------------------------

    # Log Parameters

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log Metrics

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # --------------------------
    # Log Model Artifact
    # --------------------------

    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        serialization_format="skops",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    # --------------------------
    # Display Results
    # --------------------------

    print("----------------------")
    print("Experiment Completed")
    print("----------------------")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("----------------------")
    print("MLflow Run ID:", mlflow.active_run().info.run_id)
    print("----------------------")
