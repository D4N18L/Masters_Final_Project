import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gradio as gr
from scipy.stats import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Handling_Data:

    def __init__(self, data):
        self.normalized_data = None
        self.pca = None
        self.data_reduced = None
        self.data_transformed = None
        self.data_duplicates_df = None
        self.data_regression = None
        self.data = data
        self.data_duplicates = None
        self.data_pca = None
        self.data_outliers = None
        self.message_output = []  # This is the message output of the program
        # gradio app that tracks each step of the preprocessing process the data undergoes
        self.data_missing = 0
        self.columns = self.data.columns
        self.data_type = type(data)  # data type
        self.data_shape = data.shape  # data shape
        self.data_columns = data.columns  # data columns
        self.data_info = data.info()  # data info
        self.data_describe = data.describe()  # data describe
        self.data_corr = data.corr()  # data correlation matrix
        self.data_hist = data.hist(bins=50, figsize=(20, 15))  # data histogram
        self.data_boxplot = data.boxplot(figsize=(20, 15))  # data boxplot
        self.data_kde = data.plot(kind='kde', figsize=(20, 15))  # data kde
        # self.data_scatter = data.plot(kind='scatter', figsize=(20, 15))  # data scatter

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

    def load_data(self, file_name):
        """
        This function loads the data from the specified file name
        :param file_name: the name of the file to be loaded
        :return: the dataframe of the loaded data
        """
        self.data = pd.read_csv(file_name)
        return self.data

    def data_cleaning(self):
        """
        This function cleans the data by  handling missing values , smoothing the noisy data ,
        resolving inconsistencies in the data and removing outliers and duplicates
        :return: the cleaned dataframe
        """
        self.data.head(10)
        self.data_info = self.data.info()
        print("Tip- Data Info:", self.data_info)
        print("\n")

        # Description of the dataset
        self.data_describe = self.data.describe() if any else print('No Describe')
        print('Tip - Description of the dataset:', self.data_describe)
        print("\n")

        self.data_shape = self.data.shape if any else print('Dataset is empty')
        print('Tip - Shape of the dataset:', self.data_shape)
        print("\n")

        # Check if data set has a probability distribution
        if self.data_shape[0] > 1 and self.data_shape[1] > 1:
            print('Tip - Data has a probability distribution')
        else:
            print('Tip - Data has no probability distribution')

        print("\n")

        self.data_type = type(self.data)
        print('Fact - Data type:', self.data_type)
        print("\n")

        """
        Handling Outliers
        """

        if self.data.shape[0] > 1000:  # Check if the data is larger than 1000 rows
            print("The next step is to check for outliers in the dataframe")

            print("The best step to detect outliers with this dataset size is to use IQR")
            # Calculate the IQR

            print("\n")
            self.data_numerical = self.data._get_numeric_data()
            self.data_numerical_columns = self.data_numerical.columns
            print("Numerical columns:", self.data_numerical_columns)
            print("\n")

            self.data_numerical_IQR = self.data_numerical.quantile(0.75) - self.data_numerical.quantile(0.25)
            print("IQR:", self.data_numerical_IQR)
            print("\n")

            self.data_numerical_outliers = self.data_numerical[(self.data_numerical > self.data_numerical_IQR * 1.5) | (
                    self.data_numerical < self.data_numerical_IQR * 0.5)]
            print("Outliers:", self.data_numerical_outliers)
            print("\n")

            self.data_numerical_outliers_count = self.data_numerical_outliers.shape[0]
            print("Number of outliers:", self.data_numerical_outliers_count)
            print("\n")

            self.data_numerical_outliers_percentage = self.data_numerical_outliers_count / self.data_numerical.shape[
                0] * 100
            print("Percentage of outliers:", self.data_numerical_outliers_percentage)
            print("\n")

            if self.data_numerical_outliers_percentage > 0.1:
                print("The data has outliers")
                print("\n")
                print("The next step is to remove the outliers")
                self.data_numerical_outliers_removed = self.data_numerical.drop(self.data_numerical_outliers.index)
                print("Outliers removed:", self.data_numerical_outliers_removed)
                print("\n")
                self.data = self.data_numerical_outliers_removed
                print("Dataframe with outliers removed:", self.data)
                print("\n")

            else:
                pass

        # Check if the dataset is less than 1000 rows
        elif self.data.shape[0] < 1000:
            print("The next step is to check for outliers in the dataframe")

            print("The best step to detect outliers with this dataset size is to use IQR")

            # Calculate Z-score
            # Perform z-score only on numerical columns and not categorical columns
            print("\n")
            self.data_numerical = self.data._get_numeric_data()
            self.data_numerical_columns = self.data_numerical.columns
            print("Numerical columns:", self.data_numerical_columns)
            print("\n")

            # Calculate the z-score for each column
            for column in self.data_numerical_columns:
                self.data_numerical[column] = (self.data_numerical[column] - self.data_numerical[column].mean()) / \
                                              self.data_numerical[column].std()
                print("Z-score for column:", column, "is:", self.data_numerical[column])
                print("\n")

                threshold = 2  # threshold for the z-score
                self.data_outliers = self.data_numerical[(self.data_numerical[column] > threshold) | (
                        self.data_numerical[
                            column] < -threshold)]  # Check if any values are outside the bounds and save them to self.data_outliers
                print("Outliers:", self.data_outliers)
                print("\n")

                if self.data_outliers.empty:
                    print("No outliers found")
                else:
                    print("Number of outliers found:", self.data_outliers.shape[0])
                    print("Tip - Outliers can be removed by dropping the rows")

            self.data.drop(self.data_outliers.index, inplace=True)
            print("\n")
            print("All outliers removed")

        """
        Handling Columns
        """

        self.data_columns = self.data.columns
        print('Fact - Columns:', self.data_columns)
        print("\n")

        for column in self.data_columns:  # Loop through each column in the dataframe
            # print("Column:", column)
            # print("\n")

            if self.data[column].nunique() == 1:  # if the column has only one value, then we can drop it
                print('Tip - Column:', column, 'has only one value, so it would be best to drop it')
                self.data.drop(column, axis=1, inplace=True)  # drop the column

            # if there are columns with no values, then we can drop them
            elif self.data[column].nunique() == 0:
                print('Tip - Column:', column, 'has no values, so it would be best to drop it')
                self.data.drop(column, axis=1, inplace=True)

            # Clean column names
            # if there are columns no name, let's rename them to the column number
            elif self.data[column].name is None or self.data[column].name == '' or 'Unnamed' in self.data[column].name:
                print('Tip - Column:', column, 'has no name, so it would be best to rename it')
                self.data.rename(columns={column: str(column)}, inplace=True)
                # print('Tip - Column:', column, 'has been renamed to:', str(column))
            else:
                print('Fact - Column:', column, 'has the name:', self.data[column].name)

        """
        Handling missing values
        """

        print("\n")
        # Data Cleaning
        print('The next step is to check for missing values in the dataframe depending on the number of missing values')
        self.data_missing = self.data.isnull().sum().sum() if any else print('No Missing Values')
        # print(self.data_missing + ' missing values')

        # if there are no missing values, then the data is clean
        if self.data_missing == 0:
            print('No missing values detected')

        # if the missing values are between 0 and 10% of the total values, then we can drop them
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.1):
            print(
                "Amount of missing values is less than 10% of the total values : The best option is to drop the missing values")
            self.data.dropna(inplace=True)  # Drops the missing values
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 10% and 20% of the total values, then we can replace them with the mean of the column
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.2):
            print(
                "Amount of missing values is between 10% and 20% of the total values : The best option is to replace the missing values with the mean of the column")
            self.data.fillna(self.data.mean(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'
            # return json.dumps(self.message_output + ' and replaced with the mean of the column')

        # if the missing values are between 20% and 30% of the total values, then we can replace them with the median of the column
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.3):
            print(
                "Amount of missing values is between 20% and 30% of the total values : The best option is to replace the missing values with the median of the column")
            self.data.fillna(self.data.median(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 30% and 40% of the total values, then we can replace them with the mode of the column
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.4):
            print(
                "Amount of missing values is between 30% and 40% of the total values : The best option is to replace the missing values with the mode of the column")
            self.data.fillna(self.data.mode(), inplace=True)
            self.message_output = f'Dropped {self.data_missing} missing values'

        # if the missing values are between 40% and the rest of the total values, then we can replace them with the most common value of the column
        else:
            self.data.fillna(self.data.value_counts().idxmax(), inplace=True)
            print(
                "Amount of missing values is greater than 40% of the total values : The best option is to replace the missing values with the most common value of the column")
            self.message_output = f'Dropped {self.data_missing} missing values'

        """
        Handling Duplicates
        """
        self.data_duplicates = self.data.duplicated().sum()  # Check if there are any duplicates in the dataframe in any column
        print("\n")
        print('Fact - Duplicates:', self.data_duplicates)

        if self.data_duplicates > 0:  # Check if there are any duplicates in the dataframe
            print("There are " + str(self.data_duplicates) + " duplicates in the dataframe")

            self.data_duplicates_df = self.data[self.data.duplicated()]  # Get the duplicates from the dataframe
            self.data_duplicates_df.to_csv('duplicates.csv')  # Export the duplicates to a csv file
            self.data = self.data.drop_duplicates()  # Drop the duplicates from the dataframe
            print('After the duplicates are removed from the dataframe'
                  '\n' + 'The dataframe  should have ' + str(len(self.data)) + ' rows')

        return self.data, self.message_output, self.data_columns, self.data_duplicates_df, self.data_outliers, self.data_regression, self.data_duplicates

    def data_integration(self):
        """
        This function is used to integrate the data into the database
        """
        self.data_intergrated = self.data.copy()
        self.data.to_sql('data', self.engine, if_exists='replace', index=False)
        print('Data has been integrated into the database')
        return self.data

    def data_transformation(self):
        """
        This function is used to transform the data
        """

        self.normalized_data = self.data.copy()

        # If the dataset is not normalized, then normalize the data
        if self.normalized_data.mean().mean() == 0:
            self.normalized_data = (self.normalized_data - self.normalized_data.mean()) / self.normalized_data.std()
            print('Data has been normalized')
        else:
            print('Data is already normalized')

    def data_reduction(self):
        """
        This function is used to reduce the data
        """

        # if the data is completely numeric, then we can perform PCA
        if self.data.dtypes.isin(['int64', 'float64']).all():  # Check if the data is numeric
            self.data_reduced = self.data.copy()
            self.data_reduced = self.data_reduced.drop(['id'],
                                                       axis=1) if any else self.data_reduced  # Drop the id column
            # ignore date columns if they exist
            self.data_reduced = self.data_reduced.drop(['date'], axis=1) if any else self.data_reduced

            # if the data is numeric, then we can perform PCA
            self.pca = PCA(n_components=2)  # Create a PCA object
            self.pca.fit(self.data_reduced)  # Fit the PCA object to the data
            self.data_reduced = self.pca.transform(self.data_reduced)  # Transform the data
            self.data_reduced = pd.DataFrame(self.data_reduced) if self.data_reduced.ndim == 1 else self.data_reduced
            self.data_reduced.columns = ['PC1', 'PC2']  # Update the column names
            self.data_reduced['id'] = self.data['id']  # Add the id column back to the data
            self.data_reduced['date'] = self.data['date']  # Add the date column back to the data
            self.data = self.data_reduced  # Update the data

            print('Data has been reduced')
        else:
            print('Data is not numeric, PCA cannot be performed')

        return self.data, self.data_reduced

    def data_visualization(self):
        """
        This function is used to visualize the data
        """

        # if dataset has an X axis and a Y axis, then we can perform visualization
        if self.data.shape[1] == 2:
            self.data_visualized = self.data.copy()
            self.data_visualized = self.data_visualized.drop(['id'], axis=1) if any else self.data_visualized

            # ignore date columns if they exist
            self.data_visualized = self.data_visualized.drop(['date'], axis=1) if any else self.data_visualized

            self.data_visualized.plot(kind='scatter')  # Plot the data
            plt.show()  # Show the plot
            print('Data has been visualized')
        else:
            print('Data does not have an X axis and a Y axis, visualization cannot be performed')


class Choose_ML_Model:
    """
    User picks the type of machine learning classification to run the data on
    With this , the number of possible machine learning algorithms lowers.
    """

    print('\n' + 'Choose Machine Learning Type')

    def __init__(self, ML_choice):
        self.ML_choice = ML_choice
        self.ML_report = None

    def choose_ml_type(self):
        # choose between supervised, reinforcement and unsupervised learning
        self.ML_choice = input('Type of Machine Learning: \n'
                               '1. Supervised Learning \n'
                               '2. Reinforcement Learning \n'
                               '3. Unsupervised Learning \n'
                               'Enter your choice: ')

        if self.ML_choice == '1':
            self.ML_choice = 'Supervised Learning'
        elif self.ML_choice == '2':
            self.ML_choice = 'Reinforcement Learning'
        elif self.ML_choice == '3':
            self.ML_choice = 'Unsupervised Learning'
        else:
            print('Invalid choice, please try again')
            self.choose_ml_type()  # call the function again

            report = 'You chose ' + self.ML_choice + ' as your machine learning type'
            self.ML_report.append(report)

        return self.ML_choice

    def choose_ml_learning(self):
        # choose the type of machine learning algorithm depending on the type of machine learning chosen above
        if self.ML_choice == '1':  # supervised learning
            self.ML_learning = input('Types of Supervised Learning: \n'
                                     '1. Classification \n'
                                     '2. Regression \n'
                                     'Enter your choice: ')

            if self.ML_learning == '1':
                self.ML_learning = 'Classification'
            elif self.ML_learning == '2':
                self.ML_learning = 'Regression'
            else:
                print('Invalid choice, please try again')
                self.choose_ml_learning()  # call the function again


        elif self.ML_choice == '2':  # unsupervised learning
            self.ML_learning = input('Types of Unsupervised Learning: \n'
                                     '1. Clustering \n'
                                     '2. Dimensionality Reduction \n'
                                     'Enter your choice: ')

            if self.ML_learning == '1':
                self.ML_learning = 'Clustering'
            elif self.ML_learning == '2':
                self.ML_learning = 'Dimensionality Reduction'
            else:
                print('Invalid choice, please try again')
                self.choose_ml_learning()

        elif self.ML_choice == '3':  # reinforcement learning
            self.ML_learning = input('Types of Reinforcement Learning: \n'
                                     '1. Reinforcement Learning \n'
                                     'Enter your choice: ')

            if self.ML_learning == '1':
                self.ML_learning = 'Reinforcement Learning'
            else:
                print('Invalid choice, please try again')
                self.choose_ml_learning()

        else:
            print('Invalid choice, please try again')
            self.choose_ml_type()

            report = 'You chose ' + self.ML_learning + ' as your machine learning problem'
            self.ML_report.append(report)

        return self.ML_learning


class estimate_machine_learning:
    """

    """


if __name__ == '__main__':
    data_check = Handling_Data(data=pd.read_csv('datasets/ds_salaries.csv'))
    # data_check.load_data()
    data_check.data_cleaning()
    data_check.data_transformation()
    data_check.data_reduction()
    data_check.data_visualization()

    # Either choose the type of machine learning or let the program choose for you

    # Perform the machine learning algorithm and show the user the results it would get if it was trained on the data set.
