
# 기본 패키지들 호출

import numpy as np
import os

np.random.seed(42)

# 그래프 그릴떄 쓰이는 패키지
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# mnist data 정렬 ( sklearn 0.2 부터는 fetch_openml 명령어만 먹힘 )
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([target, i] for i, target in enumerate(mnist.target[60000:])))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)
print(mnist)

mnist.target = mnist.target.astype(np.int8)
sort_by_target(mnist)

print(mnist["data"], mnist["target"])

# data의 행, 열 수 표시
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

# MNIST 모형
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
# save_fig("more_digits_plot")

# 36001 번째 이미지 확인
some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image, cmap= mpl.cm.binary, interpolation="nearest")
# plt.axis("off")

# 36001 번째 이미지가 실제로 무엇인지 확인
print(y[36000])

X_train, X_test, y_train, y_test = X[:60000], X[60000: ], y[:60000], y[60000:]

# 0 ~ 59999 까지 랜덤 배열
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

print("shuffle_index")
print(shuffle_index)
print(shuffle_index.shape)

print("X_train")
print(X_train[shuffle_index])
print(X_train[shuffle_index].shape)

print("y_train")
print(y_train[shuffle_index])
print(y_train[shuffle_index].shape)

# y_train_5에는 train_set의 타깃 변수가 들어있다.
y_train_5 = (y_train == 5)
print(y_train_5.shape)
print("y_train 안에꺼")

# for j in y_train_5:
#     print(j)

# y_test_5 에는 test_set의 타깃 변수가 들어있다.
y_test_5 = (y_test == 5)
print(y_test_5.shape)
print("y_test 안에꺼")

# for i in y_test_5:
#     print(i)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
# 여기서의 X_train은 X_train의 shuffle 버전, y_train_5는 y_train의 shuffle 버전에서 label 배열.
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit]))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# K겹 교차검증 ( 3개로 나눠주는 객체 생성, 비율 맞도록 )
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

from sklearn.model_selection import cross_val_score

print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.base import BaseEstimator

# 5가 아닌것을 찾는 경우의 정확도
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# Data imbalance problem -> 영화 데이터 분석 할 때에 데이터 불균형 문제 있던 경우와 동일

# cross_val_predict -> K_fold_cross_validation 결과의 predict를 반환해줌
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)

from sklearn.metrics import confusion_matrix

# 행 -> True 값 / 열 -> Predict 값
print(confusion_matrix(y_train_5, y_train_pred))

from sklearn.metrics import precision_score, recall_score

# 여기서 Precision ( 정밀도 )의 의미 -> 5로 판별된 이미지중 precision*100% 만이 정확하다.
print(precision_score(y_train_5, y_train_pred))

# 여기서 recall ( 재현율 )의 의미 -> 전체 숫자 5의 갯수의 recall * 100% 만이 5로 분류되었다.
print(recall_score(y_train_5, y_train_pred))

# 정밀도와 재현율 둘 다 고려해야하니까, 두개를 합친 조화평균을 자주 사용 -> F1 score로 불린다.
from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))

# 무조건 F1_score가 높다고 좋은것은 아니다. ( 1종오류 / 2종오류를 생각 해 보아야 한다 )
# 정밀도 / 재현율 Trade off 관계 -> 임계값을 찾아야 하는 문제 발생

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

threshold = 200000
y_some_digit_pred_2 = (y_scores > threshold)
print(y_some_digit_pred_2)

y_scores_2 = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

# thresholds엔 임계값이 여러개 담겨 져 있다.
# precisions, recalls엔 각 임계값에 해당하는 값들이 들어 가 있다.
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_2)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="recalls")
    plt.xlabel("thresholds")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# precision 90 이상 분류기 -> 재현율이 너무 낮다면 의미 없으니 정밀도 달성시엔 재현율을 얼마로 잡는지도 고려해야한다 !
y_train_pred_90 = ( y_scores_2 > 70000 )
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

# Roc curve -> 음성으로 분류된 것들중에 실제 양성인 경우 대 양성으로 분류된 것들중에 실제 양성인 경우의 비율
# 1 - 특이도 / 재현율

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores_2)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')

# Roc curve에서 나타나는 직선 -> 랜덤 분류기로 분류 했을 때 이다.
# 랜덤 분류기란 트레이닝셋의 범주형 데이터 비율을 따라 무작위로 예측하는것 ( 찍는거 )
# -> 실제 클래스와 비슷한 비율로 범주가 나뉘어서 ROC 곡선이 y = x 꼴이 됨
# plot_roc_curve(fpr, tpr)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_train_5, y_scores_2))

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

# predict proba -> 샘플이 행, 클래스가 열일때 샘플이 주어진 클래스에 속할 확률을 담은 배열을 줌
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random_forest")
plt.legend(loc="lower right")

print(roc_auc_score(y_train_5, y_scores_forest))

sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

some_digit_score = sgd_clf.decision_function([some_digit])

# 가장 높은 점수로 예측함
print(some_digit_score)

# 가장 높은 점수를 가진 index 값
print(np.argmax(some_digit_score))

print(sgd_clf.classes_)
print(sgd_clf.classes_[5])

from sklearn.multiclass import OneVsOneClassifier

# Ono vs One을 사용하도록 강제로 할당해줌 -> MNIST 문제에서는 45개의 class가 생김 ( SGD 분류기 )
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
ovo_clf.fit(X_train, y_train)

print(ovo_clf.predict([some_digit]))
print(len(ovo_clf.estimators_))

# randomforest 분류기
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

# Randomforest에서 각 클래스에 속할 확률 print
print(forest_clf.predict_proba([some_digit]))

# 분류기 평가 -> K fold Cross Val Score
# -> 0.8 정도 나오는데, 각 범주별의 특성을 반영해서 찍었을때 확률이 0.9 이므로 더 높여야함
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# 각 변수 Scaling -> 정확도 높여 줄 수 있음
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# 오차 행렬 검사
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
confusion_mat = confusion_matrix(y_train, y_train_pred)
print(confusion_mat)

# matplotlib 활용 ploting ( 값 높을수록 흰색 )
# 확인결과 -> 5에 해당하는 class가 어둡다
# 원인 1. class 5인 경우가 다른 숫자인 경우보다 적다
# 원인 2. class 5를 잘 구분해내지 못한다.
plt.matshow(confusion_mat, cmap=plt.cm.gray)

# 에러 비율로 보자
# class 8, 9에 해당하는 행이 밝다 -> 8, 9가 분류가 잘 안된다 ( 오류가 높다 )
# class 1은 잘 된다 ( 어둡다 )
# 에러 matrix는 대칭은아니다.
# 개선방안 ?
# 1. Training data 추가 수집
# 2. 분류기에 도움 될만한 특성 추가 ( 숫자의 동심원 갯수라는 feature ( ex_ 8은 동심원이 2개 ) )
# 3. 동심원 같은 패턴이 드러나도록 이미지 전처리 가능
row_sums = confusion_mat.sum(axis=1, keepdims=True)
norm_confusion_mat = confusion_mat / row_sums

cl_a, cl_b = 3, 5
# 3인데 3으로, 3인데 5로, 5인데 3으로, 5인데 5로 분류 한것들 순서
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

np.fill_diagonal(norm_confusion_mat, 0)
plt.matshow(norm_confusion_mat, cmap=plt.cm.gray)

plt.figure(figsize=(8,8))

# subplot -> 2 * 2 행렬의 n번째 ( -> 밑 순서 )
# 플롯팅 해 보았을 때, 왜 분류를 잘못하는지 이해 불가능한것 꽤 많음
# -> 3, 5 자체가 숫자가 비슷 하게 생기기도 했음
# -> 이미지 전처리를 해 준다면 좀 더 정확할것
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# save_fig("error_analysis_digits_plot")

# 다중 레이블 분류 ( ex_ 사진 하나에 사람 여러명 있을 때, 사람들 분류 모델 )
from sklearn.neighbors import KNeighborsClassifier

# 첫번째 조건 -> class 값이 7 이상인가
# 두번째 조건 -> 홀수인가
# 이 두가지를 한번에 분류하는 분류기 생성
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

print(knn_clf.predict([some_digit]))

# 모든 label에 대한 F1 평균 점수 return
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

# 위의 코드는 모든 label이 가중치가 같다고 가정하고 진행 한 것
# support ( 지지도 )에 가중치를 주려면 average="weighted"로 주면 됨
# 여러 가중치 방법들이 존재 ( 공식문서 참조 할 것 )

# 다중 출력 분류 -> 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화

# 노이즈 끼게 함 ( train, test 이미지에 )
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

# target 변수가 pixel 값이니 이게 들어가는게 맞음
y_train_mod = X_train
y_test_mod = X_test

# 아무인덱스나 주고
some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
# save_fig("noisy_digit_example_plot")

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
# save_fig("cleaned_digit_example_plot")

plt.show()
