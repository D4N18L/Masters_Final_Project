import numpy as np
import pandas as pd

import json
from pprint import pprint as nice_print

from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use('TkAgg')

from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, MeanShift
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.linear_model import SGDClassifier, SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC, SVC, SVR

from main_logger import logger


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

    def __init__(self, data: pd.DataFrame):
        self.data_numerical_columns = None
        self.data_numerical_IQR = None
        self.data_visualized = None
        self.data_numerical_outliers_count = None
        self.data_numerical = None
        self.normalized_data = None
        self.pca = None
        self.data_reduced = None
        self.data_duplicates_df = None
        self.data = data
        self.data_duplicates = None
        self.data_pca = None
        self.data_outliers = []
        self.data_missing = 0
        self.columns = self.data.columns
        self.data_type = type(data)
        self.data_shape = data.shape
        self.data_info = data.info()
        self.data_describe = data.describe()
        self.data_corr = data.corr()
        self.data_hist = data.hist(bins=50, figsize=(20, 15))
        self.data_boxplot = data.boxplot(figsize=(20, 15))
        self.data_kde = data.plot(kind='kde', figsize=(20, 15))
        self.DP_Report = list()

    def data_cleaning(self):
        """
        This function cleans the data by  handling missing values , smoothing the noisy data ,
        resolving inconsistencies in the data and removing outliers and duplicates

        :return: the cleaned dataframe
        """

        # self.DP_Report.append('\n ' + '-- Data Cleaning --' + '\n')

        self.data.head(10)
        self.DP_Report.append('Data Information: \n' + str(self.data_info))

        # Description of the dataset
        # self.data_describe = self.data.describe(include="all") if any else print('No Description')

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

        if self.data_missing == 0:
            self.DP_Report.append('Tip - There are no missing values in the dataframe')
            self.DP_Report.append('The next step is to check for duplicate rows: \n')

        elif self.data_missing <= (self.data.shape[0] * self.data.shape[1] * 0.1):
            self.DP_Report.append(
                'Amount of missing values is less than 10% of the total values : The best option is to drop the missing values')
            self.data.dropna(inplace=True)

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

        return self.data

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
                                                       axis=1) if any else None  # Drop the id column if it exists in the dataframe else do nothing
            self.data_reduced = self.data_reduced.drop(['date'],
                                                       axis=1) if any else None  # Drop the date column if it exists in the dataframe else do nothing

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
        # Visualize the data
        self.data_visualized = self.data.copy()
        if self.data_visualized.dtypes.isin(['int64', 'float64']).all():  # Check if the data is numeric
            self.DP_Report.append(
                'The entire dataset is numeric so we can visualize the data without extracting any features')

            # Perform one visualization of the entire dataset
            fig = matplotlib.pyplot.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.scatter(self.data_visualized[0], self.data_visualized[1])
            fig.add_subplot(111).set_title('Visualization of the entire dataset')

        else:

            self.data_visualized = self.data.copy()

            if self.data_visualized.dtypes.isin(['object']).any():
                self.data_visualized = self.data_visualized.select_dtypes(exclude=['object'])

                self.DP_Report.append('The entire dataset is not numeric so we can only visualize the numeric features')

                # Perform one visualization of the entire dataset
                self.data_visualized.plot(kind='scatter', x=self.data_visualized.columns[0],
                                          y=self.data_visualized.columns[1],
                                          figsize=(5, 5),
                                          title='Visualization of the entire dataset')

        return plt.gcf()

    def data_report(self):
        """
        This function is used to create a report of the data
        """
        # nice_print(self.DP_Report)
        # save report to file
        with open('data_report.txt', 'w') as f:
            for item in self.DP_Report:
                f.write("%s" % item)

        return self.DP_Report

    def export_clean_data(self):
        """
        This fucntion downloads the clean data as a csv file to the local machine after the data has been cleaned
        """
        self.data.to_csv('clean_data.csv')

        return self.data


# ignore warnings
import warnings

warnings.filterwarnings("ignore")


class Choose_ML:

    def __init__(self):
        self.prediction = 0
        self.accuracy_list = []
        self.accuracy = None
        self.dataset_length = 0
        self.estimator = ''
        self.depended = bool
        self.user_choice = ''
        self.report = []
        self.clean_dataset = pd.read_csv('clean_data.csv')
        self.made_estimation = False
        self.ML_Report = list()

    def ml_type(self):

        if len(self.clean_dataset) >= 50:
            self.ML_Report.append('Dataset length is large enough to use ML algorithms (More than 50 samples)')
            logger.info(f"Pointers: Dataset is large enough to use ML algorithms")

            ask_user = input('How would you like to predict the data? Category, Quantity or Dimensionality Reduction"')

            if ask_user == 'Category'.casefold():
                self.ML_Report.append('User chose to predict the data by category')
                logger.info(f"Pointers: Classification ML algorithms will be used")
                self.user_choice = 'Category'

            elif ask_user == 'Quantity'.casefold():
                logger.info(f"Pointers: Unsupervised ML algorithms will be used in this case")
                self.user_choice = 'Quantity'
                self.regresssion_ml()

            elif ask_user == 'DReduction'.casefold():
                logger.info(f"Pointers: Dimensionality Reduction ML algorithms will be used in this case")
                self.user_choice = 'Dimensionality Reduction'
                self.dimensionality_reduction()

            else:
                logger.error(f"Pointers: User did not enter a valid input, program will restart")
                self.ml_type()

        else:
            logger.info(f"Pointers: Dataset is less than 50 rows, The dataset has to be imputed")
            try:
                self.impute_data()
            except:
                logger.error(f"Pointers: Dataset could not be imputed, program will restart")
                self.ml_type()

    def classification_ml(self):

        self.ml_choice = 'Classification'
        print(self.ml_choice)

        self.depended = True
        self.estimator = 'Classification'
        self.ML_Report.append('Classification ML algorithms will be used')
        logger.info(f"Pointers: Classification ML algorithms will be used")

        if self.clean_dataset.select_dtypes(include=['object']).shape[1] > 0:
            self.ML_Report.append(
                f"Pointers: The dataset has {self.clean_dataset.select_dtypes(include=['object']).shape[1]} categorical columns")
            logger.info(f"Pointers: Dataset contains string values, they will extracted in order to use ML algorithms")
            self.clean_dataset_numeric = self.clean_dataset.select_dtypes(exclude=['object'])
            self.x = self.clean_dataset_numeric.iloc[:, :-1].values
            self.y = self.clean_dataset_numeric.iloc[:, -1].values

        if len(self.clean_dataset) < 100000:
            try:
                self.estimation = LinearSVC()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'LinearSVC'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

            except:
                if self.clean_dataset.select_dtypes(include=['object']):
                    self.estimation = GaussianNB()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'Naive Bayes'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

                else:
                    try:
                        self.estimation = KNeighborsClassifier()
                        self.estimation.fit(self.x, self.y)
                        self.made_estimation = True
                        report = 'KNeighborsClassifier'
                        self.ML_Report.append(
                            f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                        logger.info(
                            f"Decision: {report} : is recommended as one of the best ML algorithm for this dataset")
                        self.report = report
                    except:
                        self.estimation = SVC()
                        self.estimation.fit(self.x, self.y)
                        self.made_estimation = True
                        report = 'SVC'
                        self.ML_Report.append(
                            f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                        logger.info(
                            f"Decision: {report} : is recommended as one of the best ML algorithm for this dataset")
                        self.report.append(report)
            else:
                try:
                    self.estimation = SGDClassifier()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SGDClassifier'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as one of the best ML algorithm for this dataset")
                    self.report.append(report)
                except:
                    self.estimation = SVC()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SVC'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as one of the best ML algorithm for this dataset")
                    self.report.append(report)

    def clusteringML(self):

        self.ml_choice = 'Clustering'
        print(self.ml_choice)

        if self.clean_dataset.select_dtypes(include=['object']).shape[1] > 0:
            self.clean_dataset_numeric = self.clean_dataset.select_dtypes(exclude=['object'])
            self.x = self.clean_dataset_numeric.iloc[:, :-1].values
            self.y = self.clean_dataset_numeric.iloc[:, -1].values

        if len(self.clean_dataset) < 100000:
            try:
                # if estimation is KMEANS
                self.estimation = KMeans()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'KMeans'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)
            except:

                self.estimation = SpectralClustering()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'SpectralClustering'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

            finally:
                # GMM
                self.estimation = GaussianMixture()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'GaussianMixture'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

        else:
            try:
                # MiniBatchKMeans
                self.estimation = MiniBatchKMeans()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'MiniBatchKMeans'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

            except:
                # MeanShift
                self.estimation = MeanShift()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'MeanShift'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

            finally:
                self.estimation = BayesianGaussianMixture()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'BayesianGaussianMixture'
                self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

    def impute_data(self):
        logger.info(f"Pointers: Imputing data")

        self.dataset_length = len(self.clean_dataset)
        self.ML_Report.append(f"Pointers: Dataset length is {self.dataset_length}")
        logger.info(f"Pointers: Dataset length is {self.dataset_length}")

        # if the dataset is less than 50 samples, then use the mean to impute the data
        while self.dataset_length < 50:
            # add more data to the dataset with the mean of other data
            self.clean_dataset = self.clean_dataset.append(self.clean_dataset.mean(),
                                                           ignore_index=True)
            self.dataset_length = len(self.clean_dataset)

        if self.dataset_length == 50:
            self.ML_Report.append(f"Pointers: Dataset length is {self.dataset_length}")
            logger.info(f"Pointers: Dataset length is now {self.dataset_length}")
            self.ml_type()

    def feature_importance(self):

        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression
        from matplotlib import pyplot

        self.ML_Report.append(f"Pointers: Feature importance is being calculated")

        logger.info('Feature importance is being calculated')

        # define dataset
        # x and y of clean_dataset
        self.x, self.y = make_regression(n_samples=len(self.clean_dataset), n_features=len(self.clean_dataset.columns),
                                         n_informative=3, random_state=1)
        # define the model
        model = LinearRegression()

        # fit the model
        model.fit(self.x, self.y)

        # get importance
        importance = model.coef_ # this is the coefficient of the linear regression

        is_feature_importance = 0

        # summarize feature importance
        for i, v in enumerate(importance):
            self.ML_Report.append(f"Pointers: Feature: {self.clean_dataset.columns[i]}, Score: {v}")
            logger.info('Feature: %0d, Score: %.5f' % (i, v))

            # if a third of the features are important
            if v > 0.33:
                is_feature_importance += 1

        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()

        # if a third of the dataset is important
        if is_feature_importance > 3:
            return True
        else:
            return False

    def regresssion_ml(self):

        self.ml_choice = 'Regression'
        print(self.ml_choice)

        if len(self.clean_dataset) < 100000:  # if the dataset is less than 100000 samples

            # apply feature importance
            if self.feature_importance():  # perform feature importance to decide which features are important
                try:
                    self.estimation = Lasso()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'Lasso'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

                except:
                    self.estimation = ElasticNet()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'ElasticNet'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

            else:
                # Ridge regression
                try:
                    self.estimation = Ridge()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'Ridge'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

                except:
                    # svr(kernel='linear')
                    self.estimation = SVR(kernel='linear')
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SVR(kernel=linear)'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

                finally:
                    # svr(kernel='poly')
                    self.estimation = SVR(kernel='rbf')
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SVR(kernel=rbf)'
                    self.ML_Report.append(
                        f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)
        else:
            self.estimation = SGDRegressor()
            self.estimation.fit(self.x, self.y)
            report = 'SGDRegressor'
            self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
            logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
            self.report.append(report)

    def dimensionality_reduction(self):

        self.ml_choice = 'Dimensionality Reduction'
        print(self.ml_choice)

        if self.clean_dataset.select_dtypes(include=['object']).shape[1] > 0:
            self.ML_Report.append(
                f"Pointers: Dataset contain string values, dimensionality reduction is not possible unless the string values are extracted")
            logger.info(
                f"Pointers: Dataset contains string values, so dimensionality reduction is not possible unless the string values are extracted")
            self.clean_dataset_numeric = self.clean_dataset.select_dtypes(exclude=['object'])
            self.x = self.clean_dataset_numeric.iloc[:, :-1].values
            self.y = self.clean_dataset_numeric.iloc[:, -1].values

        try:

            self.estimation = PCA(svd_solver='randomized', n_components=4)
            self.estimation.fit(self.x)
            print(self.estimation.explained_variance_ratio_)
            print(self.estimation.singular_values_)

            self.made_estimation = True
            report = 'PCA(svd_solver=randomized)'
            self.ML_Report.append(f"Pointers: {report} is recommended as the best ML algorithm for this dataset")
            logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
            self.report.append(report)


        except:

            if len(self.clean_dataset) < 10000:
                try:
                    # Isomap
                    self.estimation = Isomap()
                    self.estimation.fit(self.x)
                    x_transformed = self.estimation.transform(self.x)
                    self.ML_Report.append(f"Pointers: Isomap is recommended as the best ML algorithm for this dataset")
                    self.ML_Report.append(f"Pointers: Isomap is reduced from {self.x.shape} to {x_transformed.shape}")
                    logger.info(f"Decision: Isomap : is reduced from {self.x.shape} to {x_transformed.shape}")
                    print(f"Isomap reduced the dataset from {self.x.shape} to {x_transformed.shape}")
                    self.made_estimation = True
                    report = 'Isomap'
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

                except:
                    self.estimation = SpectralEmbedding()
                    x_transformed = self.estimation.fit_transform(self.x)
                    self.ML_Report.append(
                        f"Pointers: SpectralEmbedding is recommended as the best ML algorithm for this dataset")
                    self.ML_Report.append(
                        f"Pointers: SpectralEmbedding is reduced from {self.x.shape} to {x_transformed.shape}")
                    print(f"SpectralEmbedding reduced the dataset from {self.x.shape} to {x_transformed.shape}")
                    self.made_estimation = True
                    report = 'SpectralEmbedding'
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)

                finally:
                    self.estimation = LocallyLinearEmbedding()
                    x_transformed = self.estimation.fit_transform(self.x)
                    self.ML_Report.append(
                        f"Pointers: LocallyLinearEmbedding is recommended as the best ML algorithm for this dataset")
                    self.ML_Report.append(
                        f"Pointers: LocallyLinearEmbedding is reduced from {self.x.shape} to {x_transformed.shape}")
                    print(f"LocallyLinearEmbedding reduced the dataset from {self.x.shape} to {x_transformed.shape}")

                    self.made_estimation = True
                    report = 'LocallyLinearEmbedding'
                    logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                    self.report.append(report)
            else:

                self.estimation = TruncatedSVD()
                self.estimation.fit(self.x)
                self.ML_Report.append(
                    f"Pointers: TruncatedSVD is recommended as the best ML algorithm for this dataset")
                self.ML_Report.append(
                    f"Pointers: TruncatedSVD has explained variance ratio of {self.estimation.explained_variance_ratio_}")
                self.ML_Report.append(
                    f"Pointers: TruncatedSVD has singular values of {self.estimation.singular_values_}")
                self.ML_Report.append(
                    f"TruncatedSVD has explained variance ratio sum of {self.estimation.explained_variance_ratio_.sum()}")
                print(f"Explained variance ratio: {self.estimation.explained_variance_ratio_}")
                print(f"Singular values: {self.estimation.singular_values_}")
                print(f"Explained variance sum: {sum(self.estimation.explained_variance_ratio_)}")
                self.made_estimation = True
                report = 'TruncatedSVD'
                logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
                self.report.append(report)

        finally:

            self.estimation = KernelPCA(n_components=4, kernel='linear')
            self.estimation.fit(self.x)
            self.ML_Report.append(f"Pointers: KernelPCA is recommended as the best ML algorithm for this dataset")
            self.ML_Report.append(f"Pointers: KernelPCA has explained variance ratio of {self.estimation.lambdas_}")
            self.ML_Report.append(f"Pointers: KernelPCA has singular values of {self.estimation.alphas_}")
            self.ML_Report.append(f"KernelPCA has explained variance ratio sum of {self.estimation.lambdas_.sum()}")
            self.ML_Report.append(
                f"KernelPCA has reduced the dataset from {self.x.shape} to {self.estimation.transform(self.x).shape}")

            print(f"Explained variance ratio: {self.estimation.lambdas_}")
            print(f"KernelPCA reduced the dataset from {self.x.shape} to {self.estimation.transform(self.x).shape}")
            self.made_estimation = True
            report = 'KernelPCA(kernel=linear)'
            logger.info(f"Decision: {report} : is recommended as the best ML algorithm for this dataset")
            self.report.append(report)

    def predict_ML(self):

        if self.made_estimation:
            print(f"The most recommended machine learning algorithm to use: {self.estimation}")
            self.ML_Report.append(f"The most recommended machine learning algorithm to use: {self.estimation}")
            logger.info(f"Predicting with: {self.estimation}")
        else:
            self.ML_Report.append(f"Pointers: No ML algorithm was recommended for this dataset")
            print('No prediction was made')

        if self.ml_choice == 'Dimensionality Reduction':
            print(f"The most recommended dimensionality reduction algorithm to use: {self.estimation}")
            logger.info(f"Predicting with: {self.estimation}")
            self.ML_Report.append(f"The most recommended dimensionality reduction algorithm to use: {self.estimation}")

        else:
            self.prediction = self.estimation.predict(self.x)
            self.ML_Report.append(f"The most recommended machine learning algorithm to use: {self.estimation}")

            try:
                self.accuracy = accuracy_score(self.y, self.prediction)
                self.accuracy_list.append(self.accuracy)
                logger.info(f"Accuracy score: {self.accuracy}")

            except:
                self.accuracy = r2_score(self.y, self.prediction)
                self.accuracy_list.append(self.accuracy)
                logger.info(f"Accuracy score: {self.accuracy}")

            finally:

                self.accuracy = explained_variance_score(self.y, self.prediction)
                self.accuracy_list.append(self.accuracy)
                logger.info(f"Accuracy score: {self.accuracy}")

                print(f"Accuracy list: {self.accuracy_list}")

                self.accuracy = max(self.accuracy_list)

                print(f"Accuracy score: {self.accuracy}")
                self.ML_Report.append(f"Pointers: The accuracy of the prediction is {self.accuracy}")

        return self.accuracy


import PySimpleGUI as sg


class SimpleGui:
    """
    This class is used to create a simple GUI for tool using PySimpleGUI
    """

    def __init__(self):
        """
        This is the constructor of the class
        """

        sg.theme('DarkAmber')  # Add a touch of color

        # Menu Definition
        self.menu_def = [['&Help', '&About...'], ]

        # Data Pre-processing tab layout

        self.file_browsing = [[sg.Text("Select a csv file")],
                              [sg.Input(key="--IN--"), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
                              [sg.Button("Start Data Cleaning"),
                               sg.Button("Download Cleaned Dataset"),
                               sg.Button("Exit")]]

        self.before_processing = [[sg.Text("Before Data Cleaning")],
                                  [sg.Canvas(size=(50, 50), key='--BCANVAS--')]]

        self.DP_steps = [[sg.Text("Data preprocessing steps")],
                         [sg.Listbox(values=[], size=(88, 50), key="-LIST-")],
                         [sg.Button("Write Steps To File")]]

        self.pre_processing_layout = \
            [[sg.Column(self.file_browsing),
              sg.VSeparator(),
              sg.Column(self.DP_steps)]]

        # ML tab layout
        self.ML_file_upload = [[sg.Text("What would you like to predict?")],
                               [sg.Radio('Category', "RADIO1", default=True, key='category'),
                                sg.Radio('Quantity', "RADIO1", key='quantity'),
                                sg.Radio('Dimensionality Reduction', "RADIO1", key='dim_reduction')],

                               # Add a line split
                               [sg.Text('_' * 100, size=(88, 1))],

                               [sg.Button("Submit Options")]]

        self.ML_steps = [[sg.Text("ML steps")],
                         [sg.Listbox(values=[], size=(88, 50), key="-ML-LIST-")],
                         [sg.Button("Write Steps To File")]]

        self.visual_layout = [[sg.Column(self.before_processing)]]

        self.ML_layout = \
            [[sg.Column(self.ML_file_upload),
              sg.VSeparator(),
              sg.Column(self.ML_steps)]]

        # ----- Full layout -----
        self.layout = [[sg.Menu(self.menu_def, tearoff=True)],
                       [sg.TabGroup([[sg.Tab('Data Pre-processing', self.pre_processing_layout),
                                      sg.Tab('Visualize', self.visual_layout),
                                      sg.Tab('Recommend Machine Learning Algorithm', self.ML_layout)]])]]

        # Create the window
        self.window = sg.Window('Data Science Tool', self.layout, default_button_element_size=(12, 1),
                                grab_anywhere=False, resizable=True, size=(1200, 600))

    def draw_figure(self, canvas, figure):
        """
        Draw a matplotlib figure onto a Tk canvas
        """
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def start(self):
        """
        Event loop to process "events" and get the "values" of the inputs
        """
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if event == "About...":
                self.window.disappear()
                sg.popup("DPPMLR stands for Data Pre-processing and Machine Learning Recommendations\n"
                         "This tool automatically recommends data preprocessing steps and machine learning algorithms for any data set.\n"
                         "It also allows you to save the data preprocessing steps to a file.\n"
                         "This tool was developed by Daniel Tiboah-Addo\n", grab_anywhere=True)
                self.window.reappear()

            if event == "Start Data Cleaning":
                if values["--IN--"] == "":  # if nothing is entered
                    sg.popup("No file selected", grab_anywhere=True)

                data_preprocessing = Handling_Data(pd.read_csv(values["--IN--"]))
                # Update the listbox with the data preprocessing steps
                data_preprocessing.data_cleaning()

                self.window["-LIST-"].update(data_preprocessing.DP_Report)
                self.window.refresh()

                if event == "Save Steps to a File":
                    if values["--IN--"] == "":
                        sg.popup("No file selected, Cannot save empty file", grab_anywhere=True)
                    else:
                        data_preprocessing = Handling_Data(pd.read_csv(values["--IN--"]))
                        download = sg.popup_yes_no("Do you want to save the data preprocessing steps to a file?",
                                                   grab_anywhere=True)
                        if download == "Yes":
                            data_preprocessing.data_report()
                            self.window.refresh()
                        else:
                            sg.popup("Data preprocessing steps not saved", grab_anywhere=True)
                            self.window.refresh()

                # Plot the before data cleaning figure
                fig = data_preprocessing.data_visualization()
                self.draw_figure(self.window['--BCANVAS--'].TKCanvas, fig)
                # Refresh the window
                self.window.refresh()

            if event == "Download Cleaned Dataset":
                if values["--IN--"] == "":
                    sg.popup("No file selected, Cannot save empty file", grab_anywhere=True)
                else:
                    data_preprocessing = Handling_Data(pd.read_csv(values["--IN--"]))
                    data_preprocessing.export_clean_data()
                    sg.popup("File saved successfully", grab_anywhere=True)
                    self.window.refresh()

            if event == "Submit Options":

                self.window.refresh()

                Recommend_Machine_Learning = Choose_ML()

                if values['category']:
                    # ask user a question with a checkbox
                    answer = sg.popup_yes_no("Is the dataset labelled?", grab_anywhere=True)
                    if answer == "Yes":
                        Recommend_Machine_Learning.classification_ml()

                    else:
                        Recommend_Machine_Learning.clusteringML()

                elif values['quantity']:
                    Recommend_Machine_Learning.regresssion_ml()

                elif values['dim_reduction']:
                    Recommend_Machine_Learning.dimensionality_reduction()

                self.window.refresh()

                # Predict the ML steps
                Recommend_Machine_Learning.predict_ML()

                self.window["-ML-LIST-"].update(Recommend_Machine_Learning.ML_Report)

        self.window.close()


if __name__ == '__main__':
    gui = SimpleGui()
    gui.start()
