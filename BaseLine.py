
import os
import pandas as pd

# Set Data Path
KAGGLE_DATA_PATH = os.path.join("datasets", "kaggle_competion")

def load_kaggle_data(kaggle_data_path=KAGGLE_DATA_PATH):
    train_csv_path = os.path.join(kaggle_data_path, "train.csv")
    test_csv_path = os.path.join(kaggle_data_path, "test.csv")
    return pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)

# Loading Data
train, test = load_kaggle_data()

# # Shape of Data
# print("train.sahpe = ", train.shape)
# print("test.shape = ", test.shape)
#
# # Data's Head
# print(train.head(10))
# print(test.head(10))
#
# cnt = 0
# for i in train["date"]:
#     if i[8:] == "T000000":
#         continue
#     else:
#         cnt = cnt + 1
#
# if cnt == 0:
#     print("'T000000'을 포함하지 않은것은 없습니다.")
# else:
#     print("포함되지 않은것이 있습니다.")
