import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def run_iris_project():
    print("--- Task 1: Iris Species Prediction (Decision Tree) ---")

    # 1. Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Preprocessing
    # Simulation: In a real scenario, data might have missing values.
    # We use SimpleImputer to fill missing values with the mean (just in case).
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Normalize/Scale features (Good practice for many models, though Trees are robust to it)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode Labels (Iris target is already numeric 0,1,2, but this demonstrates the step)
    # If y was ['setosa', 'versicolor'...], we would use LabelEncoder()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y) 

    # Split into Train and Test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Decision Tree Classifier
    print("Training Decision Tree Model...")
    clf = DecisionTreeClassifier(random_state=42, max_depth=3) # max_depth prevents overfitting
    clf.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    # 'macro' average calculates metrics for each label, and finds their unweighted mean.
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f"\nModel Evaluation:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    run_iris_project()
