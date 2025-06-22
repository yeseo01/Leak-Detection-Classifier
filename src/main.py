'''
test.py에서 찾은 최적의 모델과 최적의 특징을 사용해, 
DT+KNN 모델을 사용해 2차 특징 추출하고 최적의 모델을 찾는 파일
'''


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt

#===============================================
# 필요한 변수 설정
#===============================================
CLASS_LABELS = ['in', 'noise', 'normal', 'other', 'out']
CLASS_N = len(CLASS_LABELS)
class_to_index = {Class: index for index, Class in enumerate(CLASS_LABELS)}
BEST_DEPTH = 40 # Decision Tree 최적의 depth
BEST_K = 1 # KNN 최적의 K
selected_features_1 = ['lrate'] + [f'{i}HZ' for i in range(0, 5120, 10)] # 1차 특징 추출

#===============================================
# 함수 정의
#===============================================
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

# 중요 특징 추출 함수
def get_top_features(x_train, y_train, top_n):
    model = DecisionTreeClassifier(max_depth=BEST_DEPTH, criterion='gini', random_state=42)
    model.fit(x_train, y_train)

    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:top_n]

    return indices

# 최적의 특징 개수 찾기 
def find_best_feature_count(x_train, y_train, top_n_range=range(80, 130, 1)):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = [] # 검증 데이터 성능을 저장할 리스트
    tr_scores = [] # 훈련 데이터 성능을 저장할 리스트
    best_accuracy = 0 # 최적의 성능을 저장할 변수
    best_indices = None # 최적의 특징 인덱스를 저장할 변수

    for top_n in top_n_range:
        indices = get_top_features(x_train, y_train, top_n)
        x_train_selected = x_train[:, indices]

        val_fold_scores = [] # 각 fold의 검증 데이터 성능을 저장할 리스트
        tr_fold_scores = [] # 각 fold의 훈련 데이터 성능을 저장할 리스트
        # KFold 적용
        for train_index, val_index in kf.split(x_train_selected, y_train):
            x_tr, x_val = x_train_selected[train_index], x_train_selected[val_index]    
            y_tr, y_val = y_train[train_index], y_train[val_index]

            # 데이터 스케일링
            scaler = get_scaler(x_tr)
            x_tr_scaled = scaler.transform(x_tr)
            x_val_scaled = scaler.transform(x_val)

            # 모델 학습
            knn_model = KNeighborsClassifier(n_neighbors=BEST_K)
            knn_model.fit(x_tr_scaled, y_tr)

            # 훈련 데이터 성능 출력
            y_tr_predict = knn_model.predict(x_tr_scaled)
            tr_matrix = Confusion_Matrix(y_tr, y_tr_predict)
            _, _, tr_accuracy = Evaluation(tr_matrix)
            tr_fold_scores.append(tr_accuracy)

            # 검증 데이터 성능 출력
            y_val_predict = knn_model.predict(x_val_scaled)
            val_matrix = Confusion_Matrix(y_val, y_val_predict)
            _, _, val_accuracy = Evaluation(val_matrix)
            val_fold_scores.append(val_accuracy)

        # 각 특징 개수에 대한 평균 성능 계산
        mean_val_score = np.mean(val_fold_scores)
        mean_tr_score = np.mean(tr_fold_scores)
        val_scores.append(mean_val_score) # 각 특징 개수에 대한 검증 데이터 성능
        tr_scores.append(mean_tr_score) # 각 특징 개수에 대한 훈련 데이터 성능
        print(f"[K-Fold on training set] Top {top_n} features, Training Accuracy: {mean_tr_score:.4f}, Validation Accuracy: {mean_val_score:.4f}")

        if mean_val_score > best_accuracy:
            best_accuracy = mean_val_score
            best_n = top_n
            best_indices = indices

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(list(top_n_range), tr_scores, 'b-', label='Training Accuracy')
    plt.plot(list(top_n_range), val_scores, 'r-', label='Validation Accuracy')
    plt.xlabel('Top-N Feature Count')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree + KNN: Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 최적 특징 개수와 해당하는 인덱스 찾기
    print(f"Best feature count: {best_n}")

    return best_indices

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


#===============================================
# 메인 함수
#===============================================
def main():
    # 지수 표기법으로 출력되지 않게 설정
    np.set_printoptions(suppress=True, precision=0)

    # 데이터 로드
    train_data = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_2/Project_2_Classification_training_data.csv')
    test_data = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_2/Project_2_Classification_testing_data.csv')
    x_train = train_data[selected_features_1].values
    y_train = train_data['leaktype'].values
    x_test = test_data[selected_features_1].values
    y_test = test_data['leaktype'].values

    #===============================================
    # Decision Tree -> 최적의 특징 찾기
    #===============================================
    best_indices = find_best_feature_count(x_train, y_train)
    selected_features_2 = [selected_features_1[i] for i in best_indices]
    print(f"Selected features: {selected_features_2}")

    # 최적 특징 추출
    x_train_selected = x_train[:, best_indices]
    x_test_selected = x_test[:, best_indices]

    #===============================================
    # KNN -> 테스트 데이터 성능 확인
    #===============================================
    # 데이터 스케일링
    scaler = get_scaler(x_train_selected)
    x_train_selected_scaled = scaler.transform(x_train_selected)
    x_test_selected_scaled = scaler.transform(x_test_selected)

    # 모델 학습
    knn_model = KNeighborsClassifier(n_neighbors=BEST_K)
    knn_model.fit(x_train_selected_scaled, y_train)

    # 테스트 성능 출력
    y_predict = knn_model.predict(x_test_selected_scaled)
    matrix = Confusion_Matrix(y_test, y_predict)
    print(matrix)
    recall, precision, accuracy = Evaluation(matrix)
    print("recall: %.2f, precision: %.2f, accuracy: %.2f" %(recall, precision, accuracy))


# 메인 함수 실행
if __name__ == "__main__":
    main()