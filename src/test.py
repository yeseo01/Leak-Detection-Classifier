'''
1차 특징 추출을 찾으며 KNN과 DT 최적의 모델을 찾는 파일
'''



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt

CLASS_LABELS = ['in', 'noise', 'normal', 'other', 'out']
CLASS_N = len(CLASS_LABELS)
class_to_index = {Class: index for index, Class in enumerate(CLASS_LABELS)}

# 데이터 스케일링 함수    
def get_scaler(x_train, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'normalize':
        scaler = Normalizer()
    else:
        raise ValueError("지원되지 않는 스케일링 방식입니다.")

    scaler.fit(x_train)

    return scaler

# KNN 최적의 K 찾기 -> 그래프 출력
def find_optimal_k(x_train, y_train, k_range=range(1, 10, 2)):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = [] # 검증 데이터 성능을 저장할 리스트
    tr_scores = [] # 훈련 데이터 성능을 저장할 리스트
    
    for k in k_range:
        val_fold_scores = [] # 각 fold의 검증 데이터 성능을 저장할 리스트
        tr_fold_scores = [] # 각 fold의 훈련 데이터 성능을 저장할 리스트

        for train_index, val_index in kf.split(x_train, y_train):
            x_tr, x_val = x_train[train_index], x_train[val_index]    
            y_tr, y_val = y_train[train_index], y_train[val_index]
            
            # 모델 학습
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(x_tr, y_tr)
            
            # 훈련 데이터 성능
            y_tr_predict = model.predict(x_tr)
            tr_matrix = Confusion_Matrix(y_tr, y_tr_predict)
            _, _, tr_accuracy = Evaluation(tr_matrix)
            tr_fold_scores.append(tr_accuracy)
            
            # 검증 데이터 성능
            y_val_predict = model.predict(x_val)    
            val_matrix = Confusion_Matrix(y_val, y_val_predict)
            _, _, val_accuracy = Evaluation(val_matrix)
            val_fold_scores.append(val_accuracy)
        
        # 각 K에 대한 평균 성능 계산
        mean_val_score = np.mean(val_fold_scores) # 검증 데이터 성능
        mean_tr_score = np.mean(tr_fold_scores) # 훈련 데이터 성능
        val_scores.append(mean_val_score) # 각 K에 대한 검증 데이터 성능
        tr_scores.append(mean_tr_score) # 각 K에 대한 훈련 데이터 성능

        print(f"[K-Fold on training set] K={k}, Training Accuracy: {mean_tr_score:.4f}, Validation Accuracy: {mean_val_score:.4f}")
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), tr_scores, 'b-', label='Training Accuracy')
    plt.plot(list(k_range), val_scores, 'r-', label='Validation Accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.title('KNN: Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 최적 K 출력
    best_k = list(k_range)[np.argmax(val_scores)]
    print(f"Best K: {best_k}")

    return best_k

# Decision Tree 최적의 depth 찾기 -> 그래프 출력
def find_optimal_depth(x_train, y_train, depth_range=range(30, 50, 2)):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = [] # 검증 데이터 성능을 저장할 리스트
    tr_scores = [] # 훈련 데이터 성능을 저장할 리스트

    for depth in depth_range:
        val_fold_scores = [] # 각 fold의 검증 데이터 성능을 저장할 리스트
        tr_fold_scores = [] # 각 fold의 훈련 데이터 성능을 저장할 리스트

        for train_index, val_index in kf.split(x_train, y_train):
            x_tr, x_val = x_train[train_index], x_train[val_index]    
            y_tr, y_val = y_train[train_index], y_train[val_index]
            
            # 모델 학습
            model = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
            model.fit(x_tr, y_tr)

            # 훈련 데이터 성능
            y_tr_predict = model.predict(x_tr)
            tr_matrix = Confusion_Matrix(y_tr, y_tr_predict)
            _, _, tr_accuracy = Evaluation(tr_matrix)
            tr_fold_scores.append(tr_accuracy)
            
            # 검증 데이터 성능  
            y_val_predict = model.predict(x_val)
            val_matrix = Confusion_Matrix(y_val, y_val_predict)
            _, _, val_accuracy = Evaluation(val_matrix)
            val_fold_scores.append(val_accuracy)  # accuracy를 기준으로 평가

        # 각 depth에 대한 평균 성능 계산
        mean_val_score = np.mean(val_fold_scores)
        mean_tr_score = np.mean(tr_fold_scores)
        val_scores.append(mean_val_score) # 각 depth에 대한 검증 데이터 성능
        tr_scores.append(mean_tr_score) # 각 depth에 대한 훈련 데이터 성능
        print(f"[K-Fold on training set] Depth={depth}, Training Accuracy: {mean_tr_score:.4f}, Validation Accuracy: {mean_val_score:.4f}")
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(list(depth_range), tr_scores, 'b-', label='Training Accuracy')
    plt.plot(list(depth_range), val_scores, 'r-', label='Validation Accuracy')
    plt.xlabel('Depth value')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree: Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 최적 depth 출력
    best_depth = list(depth_range)[np.argmax(val_scores)]
    print(f"Best depth: {best_depth}")

    return best_depth

# 혼동 행렬 생성 함수   
def Confusion_Matrix(y_test, y_pred):
    matrix = np.zeros((CLASS_N, CLASS_N))
    
    for true, pred in zip(y_test, y_pred):
        actual_index = class_to_index[true]
        pred_index = class_to_index[pred]
        matrix[pred_index, actual_index] += 1
    
    return matrix

# 성능 평가 함수
def Evaluation(matrix):
    recalls = []
    precisions = []
    
    for index, Class in enumerate(CLASS_LABELS):
        TP = matrix[index, index]
        FP = np.sum(matrix[index, :]) - TP
        FN = np.sum(matrix[:, index]) - TP
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        recalls.append(recall)
        precisions.append(precision)
    
    recall = np.mean(recalls)
    precision = np.mean(precisions)
    accuracy = np.sum(np.diag(matrix)) / np.sum(matrix)
    
    return recall, precision, accuracy


# 메인 함수
def main():
    # 지수 표기법으로 출력되지 않게 설정
    np.set_printoptions(suppress=True, precision=0)

    # 데이터 로드
    train_data = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_2/Project_2_Classification_training_data.csv')
    test_data = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_2/Project_2_Classification_testing_data.csv')
    x_train = train_data[['lrate'] + [f'{i}HZ' for i in range(0, 5120, 10)]].values
    y_train = train_data['leaktype'].values
    x_test = test_data[['lrate'] + [f'{i}HZ' for i in range(0, 5120, 10)]].values
    y_test = test_data['leaktype'].values

    #===============================================
    # 최적의 KNN model 생성 
    #===============================================
    # 데이터 스케일링
    scaler = get_scaler(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 최적의 K 찾기
    best_k = find_optimal_k(x_train_scaled, y_train) 

    # 모델 학습
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(x_train_scaled, y_train)

    # 테스트 성능 출력
    y_predict = knn_model.predict(x_test_scaled)
    matrix = Confusion_Matrix(y_test, y_predict)
    print(matrix)
    recall, precision, accuracy = Evaluation(matrix)
    print("recall: %.2f, precision: %.2f, accuracy: %.2f" %(recall, precision, accuracy))

    #===============================================
    # 최적의 Decision Tree model 생성
    #===============================================
    # 최적의 depth 찾기
    best_depth = find_optimal_depth(x_train, y_train)

    # 모델 학습
    decision_tree_model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)
    decision_tree_model.fit(x_train_scaled, y_train)

    # 테스트 성능 출력
    y_predict = decision_tree_model.predict(x_test_scaled)
    matrix = Confusion_Matrix(y_test, y_predict)
    print(matrix)
    recall, precision, accuracy = Evaluation(matrix)
    print("recall: %.2f, precision: %.2f, accuracy: %.2f" %(recall, precision, accuracy))

   

# 메인 함수 실행
if __name__ == "__main__":
    main()