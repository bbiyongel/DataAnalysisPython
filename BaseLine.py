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
KAGGLE_DATA_PATH = os.path.join("kaggle_competion")

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

from sklearn.model_selection import StratifiedShuffleSplit

# 나누려고하는 범주의 갯수를 확인하자 ( 갯수가 2보다 작으면 ( 1이면 ) 나누어 질 수 없다 -> 상식적인것 )
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,  random_state=42)
for train_index, test_index in split.split(train, train['condition']):
    train_split = train.loc[train_index]
    train_test_split = train.loc[test_index]

# Shape of Data
print("train_split.sahpe = ", train.shape)
print("train_test_split.shape = ", train_test_split.shape)

# Data's Head
# print(train_split.head(10))
# print(train_test_split.head(10))

# Date Data Check ( T000000가 없는것은 없는지 )
cnt = 0
for i in train_split["date"]:
    if i[8:] == "T000000":
        continue
    else:
        cnt = cnt + 1

if cnt == 0:
    print("'T000000'을 포함하지 않은것은 없습니다.")
else:
    print("포함되지 않은것이 있습니다.")

# date 변경
train_split['date'] = train_split['date'].apply(lambda x: x[0:8])
train_test_split['date'] = train_test_split['date'].apply(lambda x: x[0:8])

# 결측치 탐색
# print(train_split.isnull().sum())
# print(train_test_split.isnull().sum())

# 목적변수 histogram
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

# for i in range(0,len(train_split['yr_renovated'])):
#     if train_split['yr_renovated'][i] == 0:
#         train_split['yr_renovated'][i] = train_split['yr_built'][i]

# 재건축년도가 0일 경우 건축 년도로 변환 
for df in [train_split,train_test_split]:
    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)
    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

from sklearn.linear_model import LinearRegression

train_columns = [c for c in train_split.columns if c not in ['price']]

lin_reg = LinearRegression()
lin_reg.fit(train_split[train_columns], train_split['price'])

pred = lin_reg.predict(train_test_split[train_columns])
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(train_test_split['price'], pred)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# np.savetxt('predict.csv',pred,delimiter=',')
# print(pred)
