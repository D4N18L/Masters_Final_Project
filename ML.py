import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, MeanShift
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR


class Choose_ML:

    def __init__(self):
        self.dataset_length = 0
        self.estimator = ''
        self.depended = bool
        self.report = []
        self.clean_dataset = pd.read_csv('clean_data.csv')
        self.made_estimation = False

    def ml_type(self):

        if len(self.clean_dataset) >= 50:
            print('Yes more than 50 samples')

            ask_user = input(' Would you like to predict a category?y/n: ')

            if ask_user == 'y'.casefold():
                self.user_choice = 'categorical'
                self.is_labelled()

            elif ask_user == 'n'.casefold():
                self.user_choice = 'quantitative'
                print('If you do not intend to predict a category, '
                      'you should use Unsupervised ML')
                self.pred_quantity()

            else:
                print('Invalid choice')
                self.ml_type()

        else:
            print('Dataset is too small , new data will be added')
            try:
                self.impute_data()
            except:
                print('Dataset could not be imputed')
                self.ml_type()

    def is_labelled(self):

        depended = input('is the data labeled?y/n: ')
        if depended == 'y':  # TODO: automate this if statement to check if the data is labeled or not
            self.classification_ml()
            return True
        else:
            print('Category prediction is not possible without labeled data,clustering will be performed')
            self.clusteringML()
            return False

    def classification_ml(self):
        print('Classification ML')

        self.depended = True
        self.estimator = 'Classification'
        print('Classification is the best choice')

        if self.clean_dataset.select_dtypes(include=['object']).shape[1] > 0:
            print('Dataset contains string values')
            self.clean_dataset_numeric = self.clean_dataset.select_dtypes(exclude=['object'])
            self.x = self.clean_dataset_numeric.iloc[:, :-1].values
            self.y = self.clean_dataset_numeric.iloc[:, -1].values

        if len(self.clean_dataset) < 100000:
            try:
                self.estimation = LinearSVC()
                print('LinearSVC is the best choice')
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'LinearSVC'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)
            except:
                if self.clean_dataset.select_dtypes(include=['object']):
                    self.estimation = GaussianNB()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'Naive Bayes'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

                else:
                    try:
                        self.estimation = KNeighborsClassifier()
                        self.estimation.fit(self.x, self.y)
                        self.made_estimation = True
                        report = 'KNeighborsClassifier'
                        print(f'The best algorithm to use of a dataset this size is {report}')
                        self.report = report
                    except:
                        self.estimation = SVC()
                        self.estimation.fit(self.x, self.y)
                        self.made_estimation = True
                        report = 'SVC'
                        print(f'The best algorithm to use of a dataset this size is {report}')
                        self.report.append(report)

            else:
                try:
                    self.estimation = SGDClassifier()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SGDClassifier'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)
                except:
                    self.estimation = SVC()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SVC'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

    def clusteringML(self):

        if len(self.clean_dataset) < 100000:
            try:
                # if estimation is KMEANS
                self.estimation = KMeans()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'KMeans'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)
            except:

                self.estimation = SpectralClustering()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'SpectralClustering'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)

            finally:
                # GMM
                self.estimation = GaussianMixture()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'GaussianMixture'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)

        else:
            try:
                # MiniBatchKMeans
                self.estimation = MiniBatchKMeans()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'MiniBatchKMeans'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)

            except:
                # MeanShift
                self.estimation = MeanShift()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'MeanShift'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)

            finally:
                self.estimation = BayesianGaussianMixture()
                self.estimation.fit(self.x, self.y)
                self.made_estimation = True
                report = 'BayesianGaussianMixture'
                print(f'The best algorithm to use of a dataset this size is {report}')
                self.report.append(report)

    def pred_quantity(self):
        pred_quantity = input('Would you like to predict a quantity?y/n: ')
        if pred_quantity == 'y':
            self.regresssion_ml()
        elif pred_quantity == 'n':
            self.dimensionality_reduction()

    def cluster_impute(self):
        pass

    def impute_data(self):
        print('Imputing data to make it usable')

        self.dataset_length = len(self.clean_dataset)
        print(f'The dataset is {self.dataset_length} rows long')

        # if the dataset is less than 50 samples, then use the mean to impute the data
        while self.dataset_length < 50:
            # add more data to the dataset with the mean of other data
            self.clean_dataset = self.clean_dataset.append(self.clean_dataset.mean(),
                                                           ignore_index=True)
            self.dataset_length = len(self.clean_dataset)

        if self.dataset_length == 50:
            print('The dataset is now 50 rows long')
            self.ml_type()

    def feature_importance(self):

        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression
        from matplotlib import pyplot

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
            print('Feature: %0d, Score: %.5f' % (i, v))

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

        if len(self.clean_dataset) < 100000:  # if the dataset is 100,000 samples

            # apply feature importance
            if self.feature_importance():
                try:
                    self.estimation = Lasso()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'Lasso'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

                except:
                    self.estimation = ElasticNet()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'ElasticNet'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

            else:
                # Ridge regression
                try:
                    self.estimation = Ridge()
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'Ridge'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

                except:
                    # svr(kernel='linear')
                    self.estimation = SVR(kernel='linear')
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SVR(kernel=linear)'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

                finally:
                    # svr(kernel='poly')
                    self.estimation = SVR(kernel='rbf')
                    self.estimation.fit(self.x, self.y)
                    self.made_estimation = True
                    report = 'SVR(kernel=rbf)'
                    print(f'The best algorithm to use of a dataset this size is {report}')
                    self.report.append(report)
        else:
            self.estimation = SGDRegressor()
            self.estimation.fit(self.x, self.y)
            report = 'SGDRegressor'
            print(f'The best algorithm to use of a dataset this size is {report}')
            self.report.append(report)

    def dimensionality_reduction(self):

        try:

            self.estimation = PCA(svd_solver='randomized')
            self.estimation.fit(self.x)
            self.made_estimation = True
            report = 'PCA(svd_solver=randomized)'
            print(f'The best dimensionality reduction algorithm to use of a dataset this size is {report}')
            self.report.append(report)


        except:

            if len(self.clean_dataset) < 10000:
                try:
                    # Isomap
                    self.estimation = Isomap()
                    self.estimation.fit(self.x)
                    self.made_estimation = True
                    report = 'Isomap'
                    print(f'The best dimensionality reduction algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

                except:
                    self.estimation = SpectralEmbedding()
                    self.estimation.fit(self.x)
                    self.made_estimation = True
                    report = 'SpectralEmbedding'
                    print(f'The best dimensionality reduction algorithm to use of a dataset this size is {report}')
                    self.report.append(report)

                finally:
                    self.estimation = LocallyLinearEmbedding()
                    self.estimation.fit(self.x)
                    self.made_estimation = True
                    report = 'LocallyLinearEmbedding'
                    print(f'The best dimensionality reduction algorithm to use of a dataset this size is {report}')
                    self.report.append(report)


            else:
                self.estimation = TruncatedSVD()
                self.estimation.fit(self.x)
                self.made_estimation = True
                report = 'TruncatedSVD'
                print(f'The best dimensionality reduction algorithm to use of a dataset this size is {report}')
                self.report.append(report)

        finally:
            self.estimation = KernelPCA()
            self.estimation.fit(self.x)
            self.made_estimation = True
            report = 'KernelPCA'
            print(f'The best dimensionality reduction algorithm to use of a dataset this size is {report}')
            self.report.append(report)

    def predict_ML(self):

        if self.made_estimation:
            print(f"Predicting with: {self.estimation}")
            self.prediction = self.estimation.predict(self.x)
            print(self.prediction)
        else:
            print('No prediction was made')


if __name__ == '__main__':
    ML = Choose_ML()
    ML.clean_dataset()
