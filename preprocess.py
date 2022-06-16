import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gradio as gr
from scipy.stats import stats
from sklearn.decomposition import PCA

"""
Recommend data preprocessing steps for any data set.
"""


class Handling_Data:

    def __init__(self, data):
        self.data_pca = None
        self.data_outliers = None
        self.message_output = [] # This is the message output of the program
        # gradio app that tracks each step of the preprocessing process the data undergoes
        self.data_missing = 0
        self.data = data  # dataframe
        self.columns = self.data.columns  # columns of the dataframe
        self.data_type = type(data)  # data type
        self.data_shape = data.shape  # data shape
        self.data_columns = data.columns  # data columns
        self.data_info = data.info()  # data info
        self.data_describe = data.describe()  # data describe
        self.data_corr = data.corr()  # data correlation matrix
        self.data_hist = data.hist(bins=50, figsize=(20, 15))  # data histogram
        self.data_boxplot = data.boxplot(figsize=(20, 15))  # data boxplot
        self.data_kde = data.plot(kind='kde', figsize=(20, 15))  # data kde
        self.data_scatter = data.plot(kind='scatter', figsize=(20, 15))  # data scatter

    def __str__(self):
        return str(self.columns)  # get string of column name

    def __getitem__(self, key):
        return self.data[key]  # This returns the dataframe with the specified column

    def __setitem__(self, key, value):
        self.data[key] = value  # This sets the dataframe with the specified column to the specified value

    def __len__(self):
        return len(self.data)  # This returns the length of the dataframe

    def __contains__(self, item):
        return item in self.data  # This returns whether the dataframe contains the specified column

    def __getattr__(self, name):
        return getattr(self.data, name)  # This returns the dataframe with the specified attribute

    def __dir__(self):
        return dir(self.data)  # This returns the dataframe as a list of attributes

    def __add__(self, other):
        return self.data + other  # This returns the dataframe with the addition of the specified dataframe

    def __eq__(self, other):
        return self.data == other  # This returns the dataframe with the equality of the specified

    def __ne__(self, other):
        return self.data != other  # This returns the dataframe with the inequality of the specified

    def __gt__(self, other):
        return self.data > other  # This returns the dataframe with the greater than of the specified

    def __lt__(self, other):
        return self.data < other  # This returns the dataframe with the less than of the specified

    """
    A function that takes in a dataset using GradIO and returns a dataframe.
    """

    def take_data(self, data):
        print(data)
        dataframe = pd.read_csv(data.name, delimiter=',')
        dataframe.head(10)  # Prints the first 10 rows of the dataframe
        dataframe.fillna(0, inplace=True)  # Fills in missing values with 0 if any

        row1 = dataframe.iloc[[0], :]  # Prints the first row of the dataframe

        return row1, dataframe

    """
    A  function that inspects the dataframe and returns the data type, shape, columns, info, describe, correlation matrix, histogram, boxplot, kde, and scatter plot.
    Needed Information to perform the data preprocessing recommendations.
    """

    def inspect_data(self, data):
        print(data)
        dataframe = pd.read_csv(data.name, delimiter=',')
        dataframe.head(10)  # Prints the first 10 rows of the dataframe
        self.data_info = dataframe.info() if any else print('No Info')
        self.data_describe = dataframe.describe() if any else print('No Describe')

        # Check data dimensions
        self.data_shape = dataframe.shape if any else print('Dataset is empty')

        # Check for missing values
        self.data_missing = dataframe.isnull().sum().sum() if any else print('No Missing Values')

        # if the missing values are between 0 and 10% of the total values, then we can drop them
        if self.data_missing <= (dataframe.shape[0] * dataframe.shape[1] * 0.1):
            dataframe.dropna(inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 10% and 20% of the total values, then we can replace them with the mean of the column
        elif self.data_missing <= (dataframe.shape[0] * dataframe.shape[1] * 0.2):
            dataframe.fillna(dataframe.mean(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 20% and 30% of the total values, then we can replace them with the median of the column
        elif self.data_missing <= (dataframe.shape[0] * dataframe.shape[1] * 0.3):
            dataframe.fillna(dataframe.median(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 30% and 40% of the total values, then we can replace them with the mode of the column
        elif self.data_missing <= (dataframe.shape[0] * dataframe.shape[1] * 0.4):
            dataframe.fillna(dataframe.mode(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 40% and the rest of the total values, then we can replace them with the most common value of the column
        else:
            dataframe.fillna(dataframe.value_counts().idxmax(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        self.data_columns = dataframe.columns if any else print('No Columns')

        # if there are any columns with only one value, then we can drop them
        for column in dataframe.columns:
            if dataframe[column].nunique() == 1:  # if the column has only one value, then we can drop it
                dataframe.drop(column, axis=1, inplace=True)  # drop the column

            # if there are columns with no values, then we can drop them
            elif dataframe[column].nunique() == 0:
                dataframe.drop(column, axis=1, inplace=True)

            # if there are columns no name, let's rename them to 'Unnamed' + column number
            elif dataframe[column].name is None:
                dataframe[column].name = 'Unnamed' + self.__str__()

            else:
                pass

        # Finding outliers in the dataframe
        self.data_outliers = dataframe.describe()['75%'] - dataframe.describe()[
            '25%']  # This returns the dataframe with the outliers of the specified dataframe (75% - 25%)

        # Check if data is normally distributed
        self.data_norm = stats.normaltest(dataframe) if any else print('Dataset is not normally distributed')

        # Check if data needs PCA
        if self.data_norm[1] < 0.05:  # if the p-value is less than 0.05, then we can perform PCA
            self.data_pca = PCA(n_components=2).fit_transform(dataframe)  # perform PCA on the dataframe
        else:
            print('Dataset does not need PCA')


class Pre_processing_Options:
    pass  # This is a placeholder for the preprocessing options


class Recommend_Data_Preprocessing:
    pass  # This is a placeholder for the recommendations for data preprocessing


class Recommend_ML_Algorithms:
    pass  # This is a placeholder for the recommendations for machine learning algorithms
