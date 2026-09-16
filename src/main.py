"""
Train and evaluate the final DT+KNN pipeline using the
hyperparameters selected in model_selection.py.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

#===============================================
# Configuration
#===============================================
CLASS_LABELS = ['in', 'noise', 'normal', 'other', 'out']
NUM_CLASSES = len(CLASS_LABELS)
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
BEST_DEPTH = 32  # Selected Decision Tree depth
BEST_K = 1  # Selected number of KNN neighbors
INITIAL_FEATURES = ['lrate'] + [f'{i}HZ' for i in range(0, 5120, 10)]  # Initial feature set

#===============================================
# Functions
#===============================================
# Fit feature scaler
def get_scaler(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler

# Select the most important features
def get_top_features(x_train, y_train, top_n):
    model = DecisionTreeClassifier(max_depth=BEST_DEPTH, criterion='gini', random_state=42)
    model.fit(x_train, y_train)

    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:top_n]

    return indices

# Select the optimal number of features
def find_best_feature_count(
    x_train,
    y_train,
    groups,
    top_n_range=range(80, 130),
):
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []
    tr_scores = []
    best_accuracy = 0.0
    best_n = None

    for top_n in top_n_range:
        val_fold_scores = []
        tr_fold_scores = []

        for train_index, val_index in kf.split(x_train, y_train, groups):
            x_tr, x_val = x_train[train_index], x_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            # Select features using only the training portion of this fold.
            indices = get_top_features(x_tr, y_tr, top_n)
            x_tr_selected = x_tr[:, indices]
            x_val_selected = x_val[:, indices]

            # Fit preprocessing only on the training portion of this fold.
            scaler = get_scaler(x_tr_selected)
            x_tr_scaled = scaler.transform(x_tr_selected)
            x_val_scaled = scaler.transform(x_val_selected)

            knn_model = KNeighborsClassifier(n_neighbors=BEST_K)
            knn_model.fit(x_tr_scaled, y_tr)

            y_tr_predict = knn_model.predict(x_tr_scaled)
            tr_matrix = build_confusion_matrix(y_tr, y_tr_predict)
            _, _, tr_accuracy = evaluate_metrics(tr_matrix)
            tr_fold_scores.append(tr_accuracy)

            y_val_predict = knn_model.predict(x_val_scaled)
            val_matrix = build_confusion_matrix(y_val, y_val_predict)
            _, _, val_accuracy = evaluate_metrics(val_matrix)
            val_fold_scores.append(val_accuracy)

        mean_val_score = np.mean(val_fold_scores)
        mean_tr_score = np.mean(tr_fold_scores)
        val_scores.append(mean_val_score)
        tr_scores.append(mean_tr_score)

        print(
            f"[Group CV] Top {top_n} features, "
            f"Training Accuracy: {mean_tr_score:.4f}, "
            f"Validation Accuracy: {mean_val_score:.4f}"
        )

        if mean_val_score > best_accuracy:
            best_accuracy = mean_val_score
            best_n = top_n

    plt.figure(figsize=(10, 6))
    plt.plot(list(top_n_range), tr_scores, "b-", label="Training Accuracy")
    plt.plot(list(top_n_range), val_scores, "r-", label="Validation Accuracy")
    plt.xlabel("Top-N Feature Count")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree + KNN: Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Best feature count: {best_n}")

    # After model selection, fit feature selection once on the full training set.
    return get_top_features(x_train, y_train, best_n)


# Build confusion matrix
def build_confusion_matrix(y_test, y_pred):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for true, pred in zip(y_test, y_pred):
        actual_index = CLASS_TO_INDEX[true]
        pred_index = CLASS_TO_INDEX[pred]
        matrix[actual_index, pred_index] += 1

    return matrix

# Compute macro recall, macro precision, and accuracy
def evaluate_metrics(matrix):
    recalls = []
    precisions = []

    for index in range(NUM_CLASSES):
        TP = matrix[index, index]
        FP = np.sum(matrix[:, index]) - TP
        FN = np.sum(matrix[index, :]) - TP

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        recalls.append(recall)
        precisions.append(precision)

    recall = np.mean(recalls)
    precision = np.mean(precisions)
    accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)

    return recall, precision, accuracy


#===============================================
# Main workflow
#===============================================
def main():
    # Disable scientific notation for matrix output
    np.set_printoptions(suppress=True, precision=0)

    # Load data
    project_root = Path(__file__).resolve().parents[1]
    train_data = pd.read_csv(project_root / "classification_training_data.csv")
    test_data = pd.read_csv(project_root / "classification_testing_data.csv")
    x_train = train_data[INITIAL_FEATURES].values
    y_train = train_data['leaktype'].values
    groups = pd.factorize(
        pd.MultiIndex.from_frame(train_data[['site', 'sid', 'ldate']])
    )[0]
    x_test = test_data[INITIAL_FEATURES].values
    y_test = test_data['leaktype'].values

    #===============================================
    # Select the optimal feature count using Decision Tree importance
    #===============================================
    best_indices = find_best_feature_count(x_train, y_train, groups)
    selected_features = [INITIAL_FEATURES[i] for i in best_indices]
    print(f"Selected features: {selected_features}")

    # Select the final features using the full training set
    x_train_selected = x_train[:, best_indices]
    x_test_selected = x_test[:, best_indices]

    #===============================================
    # Train KNN and evaluate on the held-out test set
    #===============================================
    # Scale features
    scaler = get_scaler(x_train_selected)
    x_train_selected_scaled = scaler.transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)

    # Train model
    knn_model = KNeighborsClassifier(n_neighbors=BEST_K)
    knn_model.fit(x_train_selected_scaled, y_train)

    # Evaluate on the held-out test set
    y_predict = knn_model.predict(x_test_selected_scaled)
    matrix = build_confusion_matrix(y_test, y_predict)
    print(matrix)
    recall, precision, accuracy = evaluate_metrics(matrix)
    print("recall: %.2f, precision: %.2f, accuracy: %.2f" %(recall, precision, accuracy))


# Entry point
if __name__ == "__main__":
    main()
