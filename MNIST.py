
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

# 36001 번째 이미지 확인
some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap= mpl.cm.binary, interpolation="nearest")
plt.axis("off")

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

y_train_5 = (y_train == 5)
print(y_train_5.shape)
print("y_train 안에꺼")

# for j in y_train_5:
#     print(j)

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
print(cross_val_score(never_5_clf))

# Data imbalance problem -> 영화 데이터 분석 할 때에 데이터 불균형 문제 있던 경우와 동일

from sklearn.model_selection import cross_val_score






plt.show()
