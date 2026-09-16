"""
Select KNN and Decision Tree hyperparameters using
group-aware cross-validation on the training set.
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_LABELS = ['in', 'noise', 'normal', 'other', 'out']
NUM_CLASSES = len(CLASS_LABELS)
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}

# Fit feature scaler
def get_scaler(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler

# Select the optimal K for KNN and plot CV performance
def find_optimal_k(x_train, y_train, groups, k_range=range(1, 10, 2)):
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []
    tr_scores = []

    for k in k_range:
        val_fold_scores = []
        tr_fold_scores = []

        for train_index, val_index in kf.split(x_train, y_train, groups):
            x_tr, x_val = x_train[train_index], x_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            # Fit preprocessing only on the training portion of this fold.
            scaler = get_scaler(x_tr)
            x_tr_scaled = scaler.transform(x_tr)
            x_val_scaled = scaler.transform(x_val)

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(x_tr_scaled, y_tr)

            y_tr_predict = model.predict(x_tr_scaled)
            tr_matrix = build_confusion_matrix(y_tr, y_tr_predict)
            _, _, tr_accuracy = evaluate_metrics(tr_matrix)
            tr_fold_scores.append(tr_accuracy)

            y_val_predict = model.predict(x_val_scaled)
            val_matrix = build_confusion_matrix(y_val, y_val_predict)
            _, _, val_accuracy = evaluate_metrics(val_matrix)
            val_fold_scores.append(val_accuracy)

        mean_val_score = np.mean(val_fold_scores)
        mean_tr_score = np.mean(tr_fold_scores)
        val_scores.append(mean_val_score)
        tr_scores.append(mean_tr_score)

        print(
            f"[Group CV] K={k}, "
            f"Training Accuracy: {mean_tr_score:.4f}, "
            f"Validation Accuracy: {mean_val_score:.4f}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), tr_scores, "b-", label="Training Accuracy")
    plt.plot(list(k_range), val_scores, "r-", label="Validation Accuracy")
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.title("KNN: Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    best_k = list(k_range)[np.argmax(val_scores)]
    print(f"Best K: {best_k}")

    return best_k


# Select the optimal Decision Tree depth and plot CV performance
def find_optimal_depth(x_train, y_train, groups, depth_range=range(30, 50, 2)):
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []  # Mean validation scores
    tr_scores = []  # Mean training scores

    for depth in depth_range:
        val_fold_scores = []  # Validation scores for each fold
        tr_fold_scores = []  # Training scores for each fold

        for train_index, val_index in kf.split(x_train, y_train, groups):
            x_tr, x_val = x_train[train_index], x_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            # Train model
            model = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
            model.fit(x_tr, y_tr)

            # Training performance
            y_tr_predict = model.predict(x_tr)
            tr_matrix = build_confusion_matrix(y_tr, y_tr_predict)
            _, _, tr_accuracy = evaluate_metrics(tr_matrix)
            tr_fold_scores.append(tr_accuracy)

            # Validation performance
            y_val_predict = model.predict(x_val)
            val_matrix = build_confusion_matrix(y_val, y_val_predict)
            _, _, val_accuracy = evaluate_metrics(val_matrix)
            val_fold_scores.append(val_accuracy)

        # Compute mean performance for this depth
        mean_val_score = np.mean(val_fold_scores)
        mean_tr_score = np.mean(tr_fold_scores)
        val_scores.append(mean_val_score)
        tr_scores.append(mean_tr_score)
        print(f"[Group CV] Depth={depth}, Training Accuracy: {mean_tr_score:.4f}, Validation Accuracy: {mean_val_score:.4f}")

    # Plot CV performance
    plt.figure(figsize=(10, 6))
    plt.plot(list(depth_range), tr_scores, 'b-', label='Training Accuracy')
    plt.plot(list(depth_range), val_scores, 'r-', label='Validation Accuracy')
    plt.xlabel('Depth value')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree: Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Report the best depth
    best_depth = list(depth_range)[np.argmax(val_scores)]
    print(f"Best depth: {best_depth}")

    return best_depth

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


# Main workflow
def main():
    project_root = Path(__file__).resolve().parents[1]
    train_data = pd.read_csv(
        project_root / "Project_2_Classification_training_data.csv"
    )

    features = ["lrate"] + [f"{i}HZ" for i in range(0, 5120, 10)]
    x_train = train_data[features].values
    y_train = train_data["leaktype"].values

    groups = pd.factorize(
        pd.MultiIndex.from_frame(train_data[["site", "sid", "ldate"]])
    )[0]

    find_optimal_k(x_train, y_train, groups)
    find_optimal_depth(x_train, y_train, groups)


# Entry point
if __name__ == "__main__":
    main()
