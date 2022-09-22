import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, MeanShift
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, SGDRegressor, ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
import argparse as ap
import json
from pprint import pprint as nice_print


class Merge_Datasets:

    def __init__(self, x_data, y_data):
        self.merged_data = None
        self.x_data = x_data
        self.y_data = y_data

    """
    Merge datasets into one dataframe with the y data as the target and the x data as the features.
    """

    def merge_datasets(self):
        self.merged_data = pd.concat([self.x_data, self.y_data], axis=1)
        return self.merged_data


class Handling_Data:

    def __init__(self, data):
        self.data_numerical_columns = None
        self.data_numerical_IQR = None
        self.data_visualized = None
        self.data_numerical_outliers_count = None
        self.data_numerical = None
        self.data_intergrated = None
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
        self.DP_Report = list()  # Data Preprocessing Report

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

    def data_cleaning(self):
        """
        This function cleans the data by  handling missing values , smoothing the noisy data ,
        resolving inconsistencies in the data and removing outliers and duplicates

        :return: the cleaned dataframe
        """

        self.DP_Report.append('\n ' + '-- Data Cleaning --' + '\n')

        self.data.head(10)
        self.data_info = self.data.info()
        self.DP_Report.append('Data Information: \n' + str(self.data_info))

        # Description of the dataset
        self.data_describe = self.data.describe() if any else print('No Description')

        self.DP_Report.append('Data Description: \n' + str(self.data_describe))

        self.data_shape = self.data.shape if any else print('Dataset is empty')
        self.DP_Report.append('Data Shape: \n' + str(self.data_shape))

        if self.data_shape[0] > 1 and self.data_shape[
            1] > 1:  # checks rows and columns for values that are greater than 1 to avoid
            self.DP_Report.append('Data has a probability distribution: \n')
        else:
            self.DP_Report.append('Data does not have a probability distribution: \n')

        self.data_type = type(self.data)
        self.DP_Report.append('Data Type: \n' + str(self.data_type))

        """
        Handling Outliers
        """

        self.DP_Report.append('\n -- Handling Outliers -- \n')

        if self.data.shape[0] > 1000:  # if the data is larger than 1000 rows
            self.DP_Report.append('Data has more than 1000 rows: \n')

            self.data_numerical = self.data._get_numeric_data()
            self.data_numerical_columns = self.data_numerical.columns
            self.DP_Report.append('Data Numerical Columns: \n' + str(self.data_numerical_columns))

            # Outlier is checked using the interquartile range (IQR)

            self.data_numerical_IQR = self.data_numerical.quanxxtile(0.75) - self.data_numerical.quantile(0.25)
            self.DP_Report.append('Data Numerical IQR: \n' + str(self.data_numerical_IQR))

            self.data_numerical_outliers = self.data_numerical[(self.data_numerical > self.data_numerical_IQR * 1.5) | (
                    self.data_numerical < self.data_numerical_IQR * 0.5)]
            self.DP_Report.append('Data Numerical Outliers: \n' + str(self.data_numerical_outliers))
            # print("Outliers:", self.data_numerical_outliers)

            self.data_numerical_outliers_count = self.data_numerical_outliers.shape[0]
            self.DP_Report.append('Data Numerical Outliers Count: \n' + str(self.data_numerical_outliers_count))
            # print("Number of outliers:", self.data_numerical_outliers_count)

            self.data_numerical_outliers_percentage = self.data_numerical_outliers_count / self.data_numerical.shape[
                0] * 100
            self.DP_Report.append(
                'Data Numerical Outliers Percentage: \n' + str(self.data_numerical_outliers_percentage))

            if self.data_numerical_outliers_percentage > 0.1:  # if the percentage of outliers is greater than 0.1
                self.DP_Report.append('Data Numerical Outliers Percentage is greater than 10%: \n')
                # print("The data has outliers")

                self.DP_Report.append('The next step is to remove the outliers: \n')

                self.data_numerical_outliers_removed = self.data_numerical.drop(self.data_numerical_outliers.index)
                self.DP_Report.append('Data Numerical Outliers Removed: \n' + str(self.data_numerical_outliers_removed))
                # print("Outliers removed:", self.data_numerical_outliers_removed)

                self.data = self.data_numerical_outliers_removed
                self.DP_Report.append('Dataframe with outliers removed: \n' + str(self.data))


            else:  # if the percentage of outliers is less than 0.1
                self.DP_Report.append('Data Numerical Outliers Percentage is less than 10%: \n')
                pass


        # Check if the dataset is less than 1000 rows
        elif self.data.shape[0] < 1000:
            self.DP_Report.append('Data has less than 1000 rows: \n')

            self.DP_Report.append('The next step is to check for outliers in the dataframe using IQR: \n')

            self.data_numerical = self.data._get_numeric_data()

            self.data_numerical_columns = self.data_numerical.columns
            self.DP_Report.append('Data Numerical Columns: ' + '\n' + str(self.data_numerical_columns))

            for column in self.data_numerical_columns:
                self.data_numerical[column] = (self.data_numerical[column] - self.data_numerical[column].mean()) / \
                                              self.data_numerical[
                                                  column].std()  # the z score checks the outliers in the data by checking the standard deviation of the data and removing the outliers that are more than 3 standard deviations from the mean

                self.DP_Report.append('Z-score for column: \n' + str(column) + str(self.data_numerical[column]))

                threshold = 2  # threshold for the z-score is set to 2 standard deviations from the mean of the data to remove outliers
                self.data_outliers = self.data_numerical[
                    (self.data_numerical[column] > threshold) | (self.data_numerical[column] < -threshold)]
                self.DP_Report.append('Data Outliers: \n' + str(self.data_outliers))

                if self.data_outliers.empty:
                    self.DP_Report.append('Data Outliers is empty:' + '\n')

                else:
                    self.DP_Report.append('Data Outliers is not empty:' + '\n')

            self.data.drop(self.data_outliers.index, inplace=True)
            self.DP_Report.append('Dataframe with outliers removed:' + '\n' + str(self.data))

        """
        Handling Columns
        """

        self.DP_Report.append('\n' + ' -- Handling Columns -- ' + '\n')

        self.data_columns = self.data.columns
        self.DP_Report.append('Data Columns:' + '\n' + str(self.data_columns))

        for column in self.data_columns:  # Loop through each column in the dataframe
            if self.data[column].nunique() == 1:
                self.DP_Report.append(
                    'Tip - Column:' + '\n' + str(column) + 'has only one unique value, so it would best to remove it')
                self.data.drop(column, axis=1, inplace=True)  # drop the column

            # if there are columns with no values, then we can drop them
            elif self.data[column].nunique() == 0:
                self.DP_Report.append(
                    'Tip - Column:' + '\n' + str(column) + 'has no unique values, so it would best to drop it')
                self.data.drop(column, axis=1, inplace=True)

            # if there are columns no name, let's rename them to the column number
            elif self.data[column].name is None or self.data[column].name == '' or 'Unnamed' in self.data[column].name:
                self.DP_Report.append(
                    'Tip - Column:' + '\n' + str(column) + 'has no name, so it would best to rename it')
                self.data.rename(columns={column: str(column)}, inplace=True)
            else:
                self.DP_Report.append('Column:' + '\n' + str(column) + 'has the name: \n' + str(self.data[column].name))

        """
        Handling missing values
        """

        self.DP_Report.append('\n' + ' -- Handling Missing Values -- ' + '\n')

        self.data_missing = self.data.isnull().sum().sum() if any else print('No Missing Values')
        self.DP_Report.append('Data Missing: \n' + str(self.data_missing))
        # print(self.data_missing + ' missing values')

        # if there are no missing values, then the data is clean
        if self.data_missing == 0:
            self.DP_Report.append('Tip - There are no missing values in the dataframe')
            self.DP_Report.append('The next step is to check for duplicate rows: \n')

        # if the missing values are between 0 and 10% of the total values, then we can drop them
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.1):
            self.DP_Report.append(
                'Amount of missing values is less than 10% of the total values : The best option is to drop the missing values')
            self.data.dropna(inplace=True)

        # if the missing values are between 10% and 20% of the total values, then we can replace them with the mean of the column
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.2):
            self.DP_Report.append(
                'Amount of missing values is between 10% and 20% of the total values : The best option is to replace the missing values with the mean of the column')
            self.data.fillna(self.data.mean(), inplace=True)

        elif self.data_missing <= (self.data.shape[0] * self.data.shape[
            1] * 0.3):  # if the missing values are between 20% and 30% of the total values, then we can replace them with the median of the column
            # The median is a better option than the mean because it is not affected by outliers in the data and it is more robust than the mean in this case
            self.DP_Report.append(
                'Amount of missing values is between 20% and 30% of the total values : The best option is to replace the missing values with the median of the column')
            self.data.fillna(self.data.median(), inplace=True)

        # if the missing values are between 30% and 40% of the total values, then we can replace them with the mode of the column .
        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.4):
            self.DP_Report.append(
                'Amount of missing values is between 30% and 40% of the total values : The best option is to replace the missing values with the mode of the column')
            self.data.fillna(self.data.mode(), inplace=True)

        # if the missing values are between 40% and the rest of the total values, then we can replace them with the most common value of the column
        else:
            self.DP_Report.append(
                'Amount of missing values is between 40% and the rest of the total values : The best option is to replace the missing values with a randmisation of the remaining values in the column')
            self.data.fillna(self.data.sample(frac=1).values, inplace=True)  # randomise the values in the column

        """
        Handling Duplicates
        """

        self.DP_Report.append('\n' + ' -- Handling Duplicates -- ' + '\n')

        self.data_duplicates = self.data.duplicated().sum()
        self.DP_Report.append('Data Duplicates:' + '\n' + str(self.data_duplicates))

        if self.data_duplicates > 0:  # Check if there are any duplicates in the dataframe
            self.DP_Report.append('Tip - There are ' + str(self.data_duplicates) + 'duplicates in the dataframe')
            self.data_duplicates_df = self.data[self.data.duplicated()]
            self.data_duplicates_df.to_csv('duplicates.csv')
            self.data = self.data.drop_duplicates()
            self.DP_Report.append(
                'After the duplicates have been removed, the dataframe should have ' + str(len(self.data)) + 'rows')

        return self.data, self.message_output, self.data_columns, self.data_duplicates_df, self.data_outliers, self.data_regression, self.data_duplicates

    def data_integration(self):
        """
        This function is used to integrate the data into the database
        """
        self.data_intergrated = self.data.copy()
        self.data.to_sql('data', self.engine, if_exists='replace', index=False)
        self.DP_Report.append('Data has been integrated into the database')

        self.data = self.data_intergrated

        return self.data

    def data_transformation(self):
        """
        This function is used to transform the data
        """

        self.normalized_data = self.data.copy()

        # If the dataset is not normalized, then normalize the data
        if self.normalized_data.mean().mean() == 0:
            self.normalized_data = (self.normalized_data - self.normalized_data.mean()) / self.normalized_data.std()
            self.DP_Report.append('Data has been normalized')
        else:
            self.DP_Report.append('Data is already normalized')

        self.data = self.normalized_data.copy()

        return self.data

    def data_reduction(self):
        """
        This function is used to reduce the data
        """

        # if the data is completely numeric, then we can perform PCA
        if self.data.dtypes.isin(['int64', 'float64']).all():  # Check if the data is numeric
            self.data_reduced = self.data.copy()
            self.data_reduced = self.data_reduced.drop(['id'],
                                                       axis=1) if any else self.data_reduced  # Drop the id column
            self.data_reduced = self.data_reduced.drop(['date'], axis=1) if any else self.data_reduced

            self.pca = PCA(n_components=2)  # Create a PCA object
            self.pca.fit(self.data_reduced)  # Fit the PCA object to the data
            self.data_reduced = self.pca.transform(self.data_reduced)  # Transform the data
            self.data_reduced = pd.DataFrame(self.data_reduced) if self.data_reduced.ndim == 1 else self.data_reduced
            self.data_reduced.columns = ['PC1', 'PC2']  # Rename the columns
            self.data_reduced['id'] = self.data['id']  # Add the id column back to the data
            self.data_reduced['date'] = self.data['date']  # Add the date column back to the data
            self.data = self.data_reduced  # Update the data

            self.DP_Report.append('Data has been reduced using PCA')
        else:
            self.DP_Report.append('Data is not numeric, so PCA cannot be reduced')

        return self.data, self.data_reduced

    def data_visualization(self):
        """
        This function is used to visualize the data
        """

        if self.data.shape[1] == 2:
            self.data_visualized = self.data.copy()
            self.data_visualized = self.data_visualized.drop(['id'], axis=1) if any else self.data_visualized

            # ignore date columns if they exist
            self.data_visualized = self.data_visualized.drop(['date'], axis=1) if any else self.data_visualized

            self.data_visualized.plot(kind='scatter')  # Plot the data
            plt.show()  # Show the plot
            self.DP_Report.append('Data has been visualized')
        else:
            self.DP_Report.append('Data is not numeric, so visualization cannot be performed')

        return self.data, self.data_visualized

    def data_report(self):
        """
        This function is used to create a report of the data
        """
        # nice_print(self.DP_Report)
        # save report to file
        with open('data_report.csv', 'w') as f:
            for item in self.DP_Report:
                f.write("%s\n" % item)

        return self.DP_Report

    def export_clean_data(self):
        """
        This function is used to export the clean data
        """
        self.data.to_csv('clean_data.csv')
        self.DP_Report.append('Clean data has been exported')

        return self.data


class Choose_ML_Model:
    """
    User picks the type of machine learning classification to run the data on
    With this , the number of possible machine learning algorithms lowers.
    """

    def __init__(self):
        self.clean_dataset = pd.read_csv('clean_data.csv')
        self.x = self.clean_dataset.iloc[:, :-1].values  # Get the independent variables
        self.y = self.clean_dataset.iloc[:, -1].values  # Get the dependent variables
        self.features_important = None
        self.ML_learning = ''
        self.categories_known = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.ML_estimator = None
        self.estimation = None
        self.ML_choice = None
        self.cat_quant = ['categorical', 'quantitative']
        self.user_choice = None
        self.ML_report = []

    def impute_dependent_variable(self):
        """
        This function is used to impute the dependent variable if a user does not have a dependent variable
        Add a check to see if the dependent variable is already imputed AND ask the user which one to use
        If not imputed, then attempt to impute the dependent variable with the information given by the user
        """

        # Check if the dependent variable is needed
        dependent_variable_needed = input('Does the data need a dependent variable? (y/n): ')
        if dependent_variable_needed == 'n':
            self.ML_report.append('The data does not need a dependent variable')
            return self.y
        else:
            # Make a new column for the dependent variable
            self.clean_dataset['class'] = None
            num_classes = int(input('How many classes are there? (int): '))

            if num_classes == 1:
                # dependent variable can not be 1 , it has to be 2 or more
                print('The dependent variable can not be 1, it has to be 2 or more')
                self.impute_dependent_variable()

            elif num_classes == 2:
                pass

            elif num_classes > 2:
                pass

            else:
                pass

    def split_data(self):
        """
        This function is used to split the data into training and testing sets
        """

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=0.2,
                                                                                random_state=0)

        return self.x_train, self.x_test, self.y_train, self.y_test

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
            self.choose_ml_type()

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

    def ml_estimator(self):

        """
        The machine learning estimator finds the right machine learning algorithm to use depending on the dataset.
        """
        # if there is more than 50 samples in the cleaned dataset, then i begin estimating a machine learning algorithm
        if self.clean_dataset.shape[0] > 50:

            # ask the user if the data is categorical or quantitative

            ask_user = input(' Would you like to predict a category or quantity \n'
                             'Type 1 for category \n'
                             'or'
                             'Type 2 for quantity \n'
                             'Enter your choice: ')

            if ask_user == '1':
                self.user_choice = 'categorical'
            elif ask_user == '2':
                self.user_choice = 'quantitative'
            else:
                print('Invalid choice, please try again')
                self.ml_estimator()  # call the function again

            if self.user_choice == self.cat_quant[0]:  # if the user chooses categorical

                if self.clean_dataset.shape[1] > 2:
                    self.ML_estimator = 'Classification'
                    if self.clean_dataset < 100000:

                        self.estimation = LinearSVC()  # Create a LinearSVC object
                        self.estimation.fit(self.x, self.y)  # Fit the LinearSVC object to the data
                        report = 'This is a classification problem, the machine learning algorithm to use would be LinearSVC'
                        print(report)
                        self.ML_report.append(report)

                    else:
                        self.estimation = SVC()  # Create a SVC object
                        self.estimation.fit(self.x, self.y)  # Fit the SVC object to the data
                        report = 'This is a classification problem, the machine learning algorithm to use would be SVC'
                        print(report)
                        self.ML_report.append(report)

                        """
                       
                            if self.clean_dataset.dtypes != 'object':  # if the dataset is not text data
                                report = 'The dataset is not text data'
                                print(report)
                                self.ML_report.append(report)
                                try:
                                    self.estimation = KNeighborsClassifier()  # Create a KNeighborsClassifier object
                                    self.estimation.fit(self.x,
                                                        self.y)  # Fit the KNeighborsClassifier object to the data
                                    report = 'This is a classification problem, the machine learning algorithm to use would be KNeighborsClassifier'
                                    print(report)
                                    self.ML_report.append(report)
                                except:  # if this does not work then try svc
                                    self.estimation = SVC()  # Create a SVC object
                                    self.estimation.fit(self.x, self.y)  # Fit the SVC object to the data
                                    report = 'This is a classification problem, the machine learning algorithm to use would be SVC'
                                    print(report)
                                    self.ML_report.append(report)
                            else:
                                # try naive bayes
                                self.estimation = GaussianNB()  # Create a GaussianNB object
                                self.estimation.fit(self.x, self.y)  # Fit the GaussianNB object to the data
                                report = 'This is a classification problem, the machine learning algorithm to use would be GaussianNB'
                                self.ML_report.append(report)
                    else:
                        # try SGD classifier
                        try:
                            self.estimation = SGDClassifier()  # Create a SGDClassifier object
                            self.estimation.fit(self.x, self.y)  # Fit the SGDClassifier object to the data
                            report = 'This is a classification problem, the machine learning algorithm to use would be SGDClassifier'
                            self.ML_report.append(report)
                        except:
                            # try kernel approximation
                            pass
                else:  # if the dataset is not labelled then we have as a clustering problem
                    self.ML_estimator = 'Clustering'
                    # are any categories known?
                    ask_categories = input('Are any categories known? \n'
                                           'Type 1 for yes \n'
                                           'or'
                                           'Type 2 for no \n'
                                           'Enter your choice: ')
                    if ask_categories == '1':
                        self.categories_known = 'yes'
                    elif ask_categories == '2':
                        self.categories_known = 'no'
                    else:
                        print('Invalid choice, please try again')
                        self.ml_estimator()

                    if self.categories_known == 'yes':
                        # check if dataset is less than 10000 samples
                        if self.clean_dataset.shape[0] < 10000:
                            # try kmeans
                            try:
                                self.estimation = KMeans()  # Create a KMeans object
                                self.estimation.fit(self.x, self.y)  # Fit the KMeans object to the data
                                report = 'This is a clustering problem, the machine learning algorithm to use would be KMeans'
                                self.ML_report.append(report)
                            except:

                                self.estimation = SpectralClustering()  # Create a SpectralClustering object
                                self.estimation.fit(self.x, self.y)  # Fit the SpectralClustering object to the data
                                report = 'This is a clustering problem, the machine learning algorithm to use would be SpectralClustering'
                                self.ML_report.append(report)
                        else:
                            # try minibatch kmeans
                            try:
                                self.estimation = MiniBatchKMeans()  # Create a MiniBatchKMeans object
                                self.estimation.fit(self.x, self.y)  # Fit the MiniBatchKMeans object to the data
                                report = 'This is a clustering problem, the machine learning algorithm to use would be MiniBatchKMeans'
                                self.ML_report.append(report)

                            except:
                                # last option did not work so reverted to kmeans
                                self.estimation = KMeans()  # Create a KMeans object
                                self.estimation.fit(self.x, self.y)  # Fit the KMeans object to the data
                                report = 'This is a clustering problem, the machine learning algorithm to use would be KMeans'
                                self.ML_report.append(report)
                    else:
                        # check if dataset is less than 10000 samples
                        if self.clean_dataset.shape[0] < 10000:
                            # try mean shift or vbgmm
                            try:
                                self.estimation = MeanShift()  # Create a MeanShift object
                                self.estimation.fit(self.x, self.y)  # Fit the MeanShift object to the data
                                report = 'This is a clustering problem, the machine learning algorithm to use would be MeanShift'
                                self.ML_report.append(report)
                            except:
                                # TODO: make a function that uses the data on vgbmm
                                pass
                        else:
                            # tough luck
                            pass
            else:
                if self.user_choice == self.cat_quant[1]:
                    self.ML_estimator = 'Regression'
                    # check if the dataset less than 100,000 samples
                    if self.clean_dataset.shape[0] > 100000:

                        self.estimation = SGDRegressor()  # Create a SGDRegressor object
                        self.estimation.fit(self.x, self.y)  # Fit the SGDRegressor object to the data
                        report = 'This is a regression problem, the machine learning algorithm to use would be SGDRegressor'
                        self.ML_report.append(report)
                    else:
                        # check if a few features are important
                        ask_features = input('Are a few features important? \n'
                                             'Type 1 for yes \n'
                                             'or'
                                             'Type 2 for no \n'
                                             'Enter your choice: ')

                        if ask_features == '1':
                            self.features_important = 'yes'
                        elif ask_features == '2':
                            self.features_important = 'no'
                        else:
                            print('Invalid choice, please try again')
                            self.ml_estimator()

                        if self.features_important == 'yes':
                            # try Lasso or ElasticNet
                            try:
                                self.estimation = ElasticNet()  # Create a ElasticNet object
                                self.estimation.fit(self.x, self.y)  # Fit the ElasticNet object to the data
                                report = 'This is a regression problem, the machine learning algorithm to use would be ElasticNet'
                                self.ML_report.append(report)
                            except:
                                # try Lasso
                                self.estimation = Lasso()  # Create a Lasso object
                                self.estimation.fit(self.x, self.y)  # Fit the Lasso object to the data
                                report = 'This is a regression problem, the machine learning algorithm to use would be Lasso'
                                self.ML_report.append(report)
                        else:
                            # try Ridge Regression
                            try:
                                self.estimation = Ridge()  # Create a Ridge object
                                self.estimation.fit(self.x, self.y)  # Fit the Ridge object to the data
                                report = 'This is a regression problem, the machine learning algorithm to use would be Ridge'
                                self.ML_report.append(report)
                            except:
                                # try SVR(kernel = 'linear')
                                self.estimation = SVR(kernel='linear')  # Create a SVR object
                                self.estimation.fit(self.x, self.y)  # Fit the SVR object to the data
                                report = 'This is a regression problem, the machine learning algorithm to use would be SVR'
                                self.ML_report.append(report)
                else:
                    print('Invalid choice, please try again')
                    self.ml_estimator()
                    
                    """

    def ml_predict(self):

        """
        With the machine learning algorithm chosen from the  ml_estimator function, this function will predict the output of the dataset.
        """
        pass
