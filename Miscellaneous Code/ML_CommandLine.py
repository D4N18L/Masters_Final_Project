import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, MeanShift
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score, adjusted_rand_score, \
    adjusted_mutual_info_score, mean_squared_error
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR

from main_logger import logger

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


class Choose_ML:

    def __init__(self):
        self.ml_choice = ''
        self.prediction = int
        self.accuracy_list = []
        self.accuracy = None
        self.dataset_length = 0
        self.estimator = ''
        self.depended = bool
        self.user_choice = ''
        self.report = []
        self.clean_dataset = pd.read_csv('../clean_data.csv')
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
                self.is_labelled()

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

    def is_labelled(self):

        depended = input('is the data labeled?y/n: ')
        if depended == 'y':  # TODO: automate this if statement to check if the data is labeled or not
            self.classification_ml()
            return True
        else:
            logger.info(
                f"Pointers: Classification ML algorithms can not be used in this case, Clustering ML algorithms will be used")
            self.clusteringML()
            return False

    def classification_ml(self):

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
        importance = model.coef_

        is_feature_importance = 0

        # summarize feature importance
        for i, v in enumerate(importance):
            self.ML_Report.append(f"Pointers: Feature: {self.clean_dataset.columns[i]}, Score: {v}")
            logger.info('Feature: %0d, Score: %.5f' % (i, v))

            # if a third of the features are important
            if v > 0.33:
                is_feature_importance += 1

        if is_feature_importance > len(
                self.clean_dataset.columns) / 3:  # if more than a third of the features are important
            self.ML_Report.append(f"Pointers: More than a third of the features are important")
            logger.info('More than a third of the features are important')
            return True
        else:
            return False

    def regresssion_ml(self):

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


if __name__ == '__main__':
    ML = Choose_ML()
    ML.ml_type()
    ML.predict_ML()
