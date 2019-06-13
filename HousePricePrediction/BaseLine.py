
import os
import pandas as pd
import matplotlib as matplt  # ploting tools
import matplotlib.pyplot as plt  # Visulization
import seaborn as sns  # BoxPlot 그리기 위해
import numpy as np
import warnings

warnings.filterwarnings('ignore')

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

# Set Data Path
KAGGLE_DATA_PATH = os.path.join("HousePricePrediction/datasets")

def load_kaggle_data(kaggle_data_path=KAGGLE_DATA_PATH):
    train_csv_path = os.path.join(kaggle_data_path, "train.csv")
    test_csv_path = os.path.join(kaggle_data_path, "test.csv")
    return pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)


# Loading Data
train, test = load_kaggle_data()

# id 컬럼 삭제
train = train.drop("id", axis=1)
test = test.drop("id", axis=1)

# Shape of Data
print("train.sahpe = ", train.shape)
print("test.shape = ", test.shape)

# oulier 탐색 / 제거

# sqft_lot15 outlier 탐색
sqft_lot15_threshold = 800000
train_temp_copy_del_sqft_lot15 = train.__deepcopy__()
train_temp_copy_del_sqft_lot15['sqft_lot15'] = train_temp_copy_del_sqft_lot15['sqft_lot15'].map(
    lambda x: np.nan if x < sqft_lot15_threshold else x)
train_temp_copy_del_sqft_lot15 = train_temp_copy_del_sqft_lot15.dropna(axis=0)
print(train_temp_copy_del_sqft_lot15)

# sqft_lot15의 outlier 제거
train['sqft_lot15'] = train['sqft_lot15'].map(lambda x: np.nan if x > sqft_lot15_threshold else x)
train = train.dropna(axis=0)

# sqft_lot outlier 탐색
sqft_lot_threshold = 1500000
train_temp_copy_del_sqft_lot = train.__deepcopy__()
train_temp_copy_del_sqft_lot['sqft_lot'] = train_temp_copy_del_sqft_lot['sqft_lot'].apply(
    lambda x: np.nan if x < sqft_lot_threshold else x)
train_temp_copy_del_sqft_lot = train_temp_copy_del_sqft_lot.dropna(axis=0)
print(train_temp_copy_del_sqft_lot)

# sqft_lot의 outlier 제거
train['sqft_lot'] = train['sqft_lot'].map(lambda x: np.nan if x > sqft_lot_threshold else x)
train = train.dropna(axis=0)

# sqft_living outlier 탐색
sqft_living_threshold = 12000
train_temp_copy_del_sqft_living = train.__deepcopy__()
train_temp_copy_del_sqft_living['sqft_living'] = train_temp_copy_del_sqft_living['sqft_living'].apply(
    lambda x: np.nan if x < sqft_living_threshold else x)
train_temp_copy_del_sqft_living = train_temp_copy_del_sqft_living.dropna(axis=0)
print(train_temp_copy_del_sqft_living)

# sqft_lot의 outlier 제거
train = train.drop(8912)
print(train.isnull().sum())

# sqft_above outlier 탐색
# 앞에서 이미 삭제된듯
sqft_above_threshold = 8000
train_temp_copy_del_sqft_above = train.__deepcopy__()
train_temp_copy_del_sqft_above['sqft_above'] = train_temp_copy_del_sqft_above['sqft_above'].apply(
    lambda x: np.nan if x < sqft_above_threshold else x)
train_temp_copy_del_sqft_above = train_temp_copy_del_sqft_above.dropna(axis=0)
print(train_temp_copy_del_sqft_above)

# grade outlier 탐색
grade_price_threshold = 6500000
train_temp_copy_del_grade_price = train.__deepcopy__()
train_temp_copy_del_grade_price['grade'] = train_temp_copy_del_grade_price['grade'].apply(
    lambda x: np.nan if x != 11 else x)
train_temp_copy_del_grade_price['price'] = train_temp_copy_del_grade_price['price'].apply(
    lambda x: np.nan if x < grade_price_threshold else x)
train_temp_copy_del_grade_price = train_temp_copy_del_grade_price.dropna(axis=0)
print(train_temp_copy_del_grade_price)
train = train.drop(2775)

# bathroom outlier 탐색
# 2775 앞에서 지워짐
bathroom_price_threshold = 6500000
train_temp_copy_del_bathroom_price = train.__deepcopy__()
train_temp_copy_del_bathroom_price['bathrooms'] = train_temp_copy_del_bathroom_price['bathrooms'].apply(
    lambda x: np.nan if x != 4.5 else x)
train_temp_copy_del_bathroom_price['price'] = train_temp_copy_del_bathroom_price['price'].apply(
    lambda x: np.nan if x < bathroom_price_threshold else x)
train_temp_copy_del_bathroom_price = train_temp_copy_del_bathroom_price.dropna(axis=0)
print(train_temp_copy_del_bathroom_price)

cnt = 0
# Date Data Check ( T000000가 없는것은 없는지 )
for i in train["date"]:
    if i[8:] == "T000000":
        continue
    else:
        cnt = cnt + 1

if cnt == 0:
    print("'T000000'을 포함하지 않은것은 없습니다.")
else:
    print("포함되지 않은것이 있습니다.")

# 127495.22247481202
# 0.15874729786956207
# 0.1586730572667452

# Feature Engineering
for df in [train,test]:
    df['date'] = df['date'].apply(lambda x: x[0:8]).astype(int)
    df['total_room_num'] = df['bedrooms'] + df['bathrooms']
    df['room_num_per_floors'] = (df['total_room_num']) / df['floors']
    df['sqft_living_per_floors'] = df['sqft_living'] / df['floors']
    df['sqft_living_15_per_floors'] = df['sqft_living15'] / df['floors']
    df['garden'] = df['sqft_lot'] - df['sqft_living']
    df['garden'] = df['sqft_lot15'] - df['sqft_living15']
    df['is_renovated'] = df['yr_renovated'] - df['yr_built']
    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

train['per_price'] = train['price']/train['sqft_lot']
zipcode_price = train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()
train = pd.merge(train,zipcode_price,how='left',on='zipcode')
test = pd.merge(test,zipcode_price,how='left',on='zipcode')

# train['per_price_15'] = train['price']/train['sqft_lot_15']
# zipcode_price = train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()
# df_train = pd.merge(train,zipcode_price,how='left',on='zipcode')
# df_test = pd.merge(test,zipcode_price,how='left',on='zipcode')

for df in [train, test]:
    df['mean'] = df['mean'] * df['sqft_lot']
    df['var'] = df['var'] * df['sqft_lot']


# Shape of Data
print("train.sahpe = ", train.shape)
print("test.shape = ", test.shape)

from sklearn.model_selection import StratifiedShuffleSplit

# 나누려고하는 범주의 갯수를 확인하자 ( 갯수가 2보다 작으면 ( 1이면 ) 나누어 질 수 없다 -> 상식적인것 )
# n_split -> iteration 횟수

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(train, train['condition']):
    train_split = train.loc[train_index]
    train_test_split = train.loc[test_index]

from sklearn.model_selection import train_test_split as train_test_lib

# Random split
train_split, train_test_split = train_test_lib(train, test_size=0.2, random_state=42)

# Shape of Data
print("train_split.sahpe = ", train_split.shape)
print("train_test_split.shape = ", train_test_split.shape)

# Data's Head
print(train_split.head(10))
print(train_test_split.head(10))

# 결측치 탐색
print(train_split.isnull().sum())
print(train_test_split.isnull().sum())

# 목적변수 histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(train_split['price'])
# plt.show()

# log 변환
train_split['price'] = np.log(train_split['price'])
train['price'] = np.log(train['price'])
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(train_split['price'])


# plt.show()

# 이산형 변수 플롯팅 ( boxplot )
def discrete_data_box_plot(columname):
    # print(train_split[columname].value_counts())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=columname, y="price", data=train_split)
    plt.title("Box Plot" + columname)
    # plt.show()


# 연속형 변수 vs Price 플롯팅 ( scatter plot )
def show_target_scatter_plot(colum_name):
    plt.scatter(train_split[colum_name], train_split['price'])
    plt.xlabel(colum_name)
    plt.ylabel('price')
    plt.title('Price VS ' + colum_name)


# 이산형 변수 List
attributes_discrete = [
    'yr_built', 'yr_renovated', 'bedrooms', 'bathrooms',
    'floors', 'waterfront', 'view', 'condition', 'grade'
]

# 이산형 변수 BoxPlot
for i in attributes_discrete:
    discrete_data_box_plot(i)
    # plt.show()

# 연속형 변수 List
attributes_continuous = [
    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
    'lat', 'long', 'sqft_living15', 'sqft_lot15'
]

# 연속형 변수 vs Price( target value ) Scatter Plot
for i in attributes_continuous:
    show_target_scatter_plot(i)
    # plt.show()

# 폐기 코드
# for i in range(0,len(train_split['yr_renovated'])):
#     if train_split['yr_renovated'][i] == 0:
#         train_split['yr_renovated'][i] = train_split['yr_built'][i]

# 재건축년도가 0일 경우 건축 년도로 변환
for df in [train_split, train_test_split]:
    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)
    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

from sklearn.linear_model import LinearRegression

train_columns = [c for c in train_split.columns if c not in ['price', 'per_price']]

lin_reg = LinearRegression()
lin_reg.fit(train_split[train_columns], train_split['price'])

pred = lin_reg.predict(train_test_split[train_columns])
adjusted_pred = np.exp(pred)

from sklearn.metrics import mean_squared_error

print(train_test_split.isnull().any())
print(np.where(np.isnan(train_test_split)))
lin_mse = mean_squared_error(train_test_split['price'], adjusted_pred)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeCV

param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.015,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4950}

y_reg = train_split['price']

# prepare fit model with cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_split))
predictions = np.zeros(len(train_test_split))
predictions_submission = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

# run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_split)):
    trn_data = lgb.Dataset(train_split.iloc[trn_idx][train_columns],
                           label=y_reg.iloc[trn_idx])  # , categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_split.iloc[val_idx][train_columns],
                           label=y_reg.iloc[val_idx])  # , categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                    early_stopping_rounds=100)
    oof[val_idx] = clf.predict(train_split.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)
    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    # predictions
    predictions += clf.predict(train_test_split[train_columns], num_iteration=clf.best_iteration) / folds.n_splits
    predictions_submission += clf.predict(test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

# oof -> training set의 예측결과 y_reg -> training set의 가격변수
cv = np.sqrt(mean_squared_error(np.exp(oof), np.exp(y_reg)))
print(cv)

# predictions -> test_set 예측결과
cv_test = np.sqrt(mean_squared_error(train_test_split['price'], np.exp(predictions)))
print(cv_test)

np.savetxt('predict.csv', np.exp(predictions_submission), delimiter=',')
# print(pred)

# 제출용 코드
y_reg = train['price']

# prepare fit model with cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

# run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx][train_columns],
                           label=y_reg.iloc[trn_idx])  # , categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][train_columns],
                           label=y_reg.iloc[val_idx])  # , categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=500,
                    early_stopping_rounds=100)
    oof[val_idx] = clf.predict(train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)
    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    # predictions
    predictions += clf.predict(test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

cv = np.sqrt(mean_squared_error(oof, y_reg))
print(cv)

##plot the feature importance
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

np.savetxt('predict.csv', np.exp(predictions), delimiter=',')
print(pred)
