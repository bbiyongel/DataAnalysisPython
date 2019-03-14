import os
import tarfile

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from zlib import crc32

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from six.moves import urllib

from pandas.plotting import scatter_matrix

from sklearn.impute import SimpleImputer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

# head 부분
print(housing.head())

# 전체 csv data에 대한 information
print(housing.info())

# 범주형으로 추정되는 변수의 count 정보
print(housing["ocean_proximity"].value_counts())

# housing data에 대한 기초통계량
print(housing.describe())

# 히스토그램 show
#housing.hist(bins=50, figsize=(20, 15))
#plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

# ???
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

#random_state = seed number -> seed_num이 같으면 같은 난수가 생성된다.
#https://code.i-harness.com/ko-kr/q/15973e3
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#housing["income_cat"].hist(bins=50, figsize=(20, 15))
#plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace=True)

housing = strat_train_set.copy()

# housing.plot(kind = "scatter", x = "longitude", y = "latitude")
# housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)

housing.plot(kind="scatter", x="longitude", y= "latitude", alpha = 0.4, s= housing["population"]/100, label = " population", figsize = (10,7), c="median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)

plt.legend()
# plt.show()

corr_matrix = housing.corr()
print(corr_matrix)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing.dropna(subset=["total_bedrooms"]) # 해당 구역 제거 ( N/A 인 부분 )
housing.drop("total_bedrooms", axis=1) # 전체 특성 제거
median = housing["total_bedrooms"].median() #Training 셋의 median값 으로 N/A값 채움, Test 셋평가할때도 해당 값 사용
housing["total_bedrooms"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy= "median")

#median 값은 수치형 데이터에만 적용되기 때문에 범주형 데이터 삭제
housing_num = housing.drop("ocean_proximity", axis = 1 )

imputer.fit(housing_num)

print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))

housing_cat = housing["ocean_proximity"]
print(housing_cat.head(10))

housing_cat_encoded, housing_cat_categories = housing_cat.factorize()
print(housing_cat_encoded[:10])

print(housing_cat_categories)

print(housing["ocean_proximity"].head(10))

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

# from sklearn.preprocessing import CategoricalEncoder

# cat_encoder = CategoricalEncoder()

cat_encoder = OneHotEncoder(sparse=False)
housing_cat_reshaped = housing_cat.values.reshape(-1,1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
print("asdfasdf")
print(housing_cat_1hot)

print(cat_encoder.categories_)
print(cat_encoder.categories)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

print('aasdfasdfasdf')
print(housing_num_tr)

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list= [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("예측 : ", lin_reg.predict(some_data_prepared))
print("레이블 : ", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regresssion")
print(lin_rmse)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree")
print(tree_rmse)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("Cross-Validation")
print(tree_rmse_scores)

def display_scores(scores):
    print("Scores :", scores)
    print("Mean :", scores.mean())
    print("Standard Deviation :", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

for_mse = mean_squared_error(housing_labels, housing_predictions)
for_rmse = np.sqrt(for_mse)

print("Random Forest")
print(for_rmse)

for_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
for_rmse_scores = np.sqrt(-for_scores)
display_scores(for_rmse_scores)

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators' : [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap' : [False], 'n_estimators' : [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(n_estimators=10)

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse= True))

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

from sklearn.svm import SVR

# for i in range(1, 11):
#     svm_reg = SVR(kernel="linear", C=i*0.1)
#     svm_reg.fit(housing_prepared, housing_labels)
#     housing_predictions_svm = svm_reg.predict(housing_prepared)
#
#     for_mse_svm = mean_squared_error(housing_predictions_svm, housing_labels)
#     for_rmse_svm = np.sqrt(for_mse_svm)
#     print("svm, kernel = linear C = " + str(i))
#     print(for_rmse_svm)

# for i in range(1, 11):
#     svm_reg = SVR(kernel="rbf", C=i*0.1)
#     svm_reg.fit(housing_prepared, housing_labels)
#     housing_predictions_svm = svm_reg.predict(housing_prepared)
#
#     for_mse_svm = mean_squared_error(housing_predictions_svm, housing_labels)
#     for_rmse_svm = np.sqrt(for_mse_svm)
#     print("svm, kernel = rbf C = " + str(i))
#     print(for_rmse_svm)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

n_iter_search = 20

random_forest_reg = RandomForestRegressor(n_estimators=20)

param_dist= {
    "max_depth" : [3, None],
    "max_features" : randint(1,11),
    "min_samples_split" : randint(2,11),
    "bootstrap" : [True, False]
    # "criterion" : ["gini", "entropy"]
}

random_search = RandomizedSearchCV(random_forest_reg, param_distributions=param_dist, n_iter=n_iter_search, cv=5)
random_search.fit(housing_prepared, housing_labels)

print(random_search.best_params_)
print(random_search.best_estimator_)

cvres_random_search = random_search.cv_results_
for mean_score, params in zip(cvres_random_search["mean_test_score"], cvres_random_search["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = random_search.best_estimator_.feature_importances_
print(feature_importances)

# plt.show()

