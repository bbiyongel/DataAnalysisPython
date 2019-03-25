import pandas as pd
import os
import matplotlib as plt
import seaborn as sns
# Loading packages
import pandas as pd #Analysis
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis
from scipy.stats import norm #Analysis
from sklearn.preprocessing import StandardScaler #Analysis
from scipy import stats #Analysis
import warnings
warnings.filterwarnings('ignore')

# import gc
# import plotly.graph_objs as go
# import plotly.offline as py
# from plotly import tools
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

# Set Data Path
KAGGLE_DATA_PATH = os.path.join("datasets", "kaggle_competion")

def load_kaggle_data(kaggle_data_path=KAGGLE_DATA_PATH):
    train_csv_path = os.path.join(kaggle_data_path, "train.csv")
    test_csv_path = os.path.join(kaggle_data_path, "test.csv")
    print(train_csv_path)
    return pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)

# Loading Data
train, test = load_kaggle_data()
















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

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="waterfront", y="price", data=train)
plt.title("Box Plot")
plt.show()

def discrete_data_box_plot(columname):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=columname, y="price", data=train)
    plt.title("Box Plot")
    plt.show()

def show_target_scatter_plot(colum_name):
    plt.scatter(train[colum_name], train['price'])
    plt.xlabel(colum_name)
    plt.ylabel('price')
    plt.title('Price VS '+ colum_name)

attributes = [
    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
    'lat', 'long', 'sqft_living15', 'sqft_lot15'
]

# for i in attributes:
#     show_target_scatter_plot(i)
#     plt.show()


# # bedrooms Category counts
# print(train["yr_built"].value_counts())
# discrete_data_box_plot("yr_built")
#
# # bedrooms Category counts
# print(train["yr_renovated"].value_counts())
# discrete_data_box_plot("yr_renovated")

# # bedrooms Category counts
# print(train["bedrooms"].value_counts())
# discrete_data_box_plot("bedrooms")
#
# # bathrooms Category counts
# print(train["bathrooms"].value_counts())
# discrete_data_box_plot("bathrooms")
#
# # floors Category counts
# print(train["floors"].value_counts())
# discrete_data_box_plot("floors")
#
# # waterfront Category counts
# print(train["waterfront"].value_counts())
# discrete_data_box_plot("waterfront")
#
# # view Category counts
# print(train["view"].value_counts())
# discrete_data_box_plot("view")
#
# # condition Category counts
# print(train["condition"].value_counts())
# discrete_data_box_plot("condition")
#
# # grade Category counts
# print(train["grade"].value_counts())
# discrete_data_box_plot("grade")
#
# ids = test['id']
# target = train['price'].values
# data = pd.concat([train.drop(['price'], axis=1), test])

from pandas.plotting import scatter_matrix

# scatter_matrix(train[attributes], figsize=(10, 15))

# fig, axes = plt.subplots(4, 2, figsize=(10, 15), dpi=100)
#
# columns = [
#     'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
#     'lat', 'long', 'sqft_living15', 'sqft_lot15'
# ]
#
# for i, col in enumerate(columns):
#     sns.distplot(train[col].values, hist_kws={'alpha': 1}, ax=axes.flat[i])
#     axes.flat[i].set_xlabel('')
#     axes.flat[i].set_title(col, fontsize=14)
#     axes.flat[i].grid(axis='y')
#     axes.flat[i].tick_params(axis='x', rotation=90)
#
# fig.suptitle('')
# # fig.delaxes(axes.flat[7])
# fig.tight_layout()
# plt.show()


# print("train.sahpe = ",train.shape)
# print("test.shape = ", test.shape)
# print("data.shape = ", data.shape)
# print("target.shape = ", target.shape)
#
# print(train.info())
# print(train.isnull().sum())

# print(train["date"].head(30))



# print(train["bedrooms"].drop_duplicates())



# import matplotlib.pyplot as plt
#
#histogram
# f, ax = plt.subplots(figsize=(8, 6))
# sns.distplot(train['price'])
# plt.show()
#
#
# def draw_scatter(df, col_name, axes):
#     df.plot(kind='scatter', x='long', y='lat', c=col_name,
#             cmap=plt.get_cmap('plasma'), colorbar=False, alpha=0.1, ax=axes)
#     axes.set(xlabel='longitude', ylabel='latitude')
#     axes.set_title(col_name, fontsize=13)
#     return axes
#
#
# fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 9), dpi=100)
#
# draw_scatter(train, 'price_q', axes.flat[0])
# draw_scatter(train, 'yr_built_q', axes.flat[1])
# draw_scatter(train, 'sqft_living_q', axes.flat[2])
# fig.suptitle('Seattle Housing Prices', fontsize=15)
# fig.delaxes(axes.flat[3])
# fig.tight_layout()
# fig.subplots_adjust(top=0.9)
# # plt.show()
#

# train['price_q'] = pd.qcut(train['price'], q=10, labels=list(range(10))).astype(int)
# train['yr_built_q'] = pd.qcut(train['yr_built'], q=10, labels=list(range(10))).astype(int)
# train['sqft_living_q'] = pd.qcut(train['sqft_living'], q=10, labels=list(range(10))).astype(int)
#
# from bokeh.models.mappers import ColorMapper, LinearColorMapper
# from bokeh.plotting import gmap
# from bokeh.models import GMapOptions, HoverTool, ColumnDataSource
# from bokeh.io import output_notebook, show, output_file
# from matplotlib import *
# from bokeh.palettes import Plasma10
#
# output_notebook()
# output_file("gmap.html")
#
# api_key = 'AIzaSyCafbWU4mSLjLTXcYdn75men73JtToqYWU'
#
# map_options = GMapOptions(lat=47.5112, lng=-122.257, map_type='roadmap', zoom=10)
# p = gmap(api_key, map_options, title='Seattle Housing Prices')
#
# source = ColumnDataSource(
#     data=dict(
#         lat=train['lat'].tolist(),
#         long=train['long'].tolist(),
#         color=train['price_q'].tolist(),
#         price=train['price'].tolist()
#     )
# )
#
# color_mapper = LinearColorMapper(palette=Plasma10)
# p.circle(x='long', y='lat',
#          fill_color={'field': 'color', 'transform': color_mapper},
#          fill_alpha=0.3, line_color=None, source=source)
#
# hover = HoverTool(
#     tooltips=[
#         ('lat', '@lat'),
#         ('long', '@long'),
#         ('price', '@price')
#     ]
# )
#
# p.add_tools(hover)
# show(p)
