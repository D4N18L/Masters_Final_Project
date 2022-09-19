import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


class Choose_ML:

    def __init__(self):
        self.report = []
        self.clean_dataset = pd.read_csv('clean_data.csv')

    def ml_type(self):

        self.ML_choice = input('Type of ML: '
                               '1 for Supervised ML, '
                               '2 for Unsupervised ML, '
                               '3 for Reinforcement ML: ')

        if self.ML_choice == '1':
            self.ML_choice = 'Supervised Learning'
            print(f"You have chosen {self.ML_choice}")
            self.supervised_ml()

        elif self.ML_choice == '2':
            self.ML_choice = 'Unsupervised Learning'
            print(f"You have chosen {self.ML_choice}")
            self.unsupervised_ml()

        elif self.ML_choice == '3':
            self.ML_choice = 'Reinforcement Learning'
            print(f"You have chosen {self.ML_choice}")
            self.reinforcement_ml()

        else:
            print('Invalid choice')
            self.ml_type()

        return self.ML_choice

    def supervised_ml(self):
        print('Supervised ML')

        if len(self.clean_dataset) > 50:
            print('Yes more than 50 samples')

            ask_user = input(' Would you like to predict a category'
                             ' Type Yes or No: ')

            if ask_user == 'Yes'.casefold():
                self.user_choice = 'categorical'
            elif ask_user == 'No'.casefold():
                self.user_choice = 'quantitative'
                print('If you do not intend to predict a category, '
                      'you should use Unsupervised ML')
                self.unsupervised_ml()

            else:
                print('Invalid choice')
                self.supervised_ml()

            if self.clean_dataset.shape[1] > 2:  # if more than 2 columns
                self.estimaor = 'Classification'
                print('Classification is the best choice')

                if self.clean_dataset.select_dtypes(include=['object']).shape[1] > 0:
                    print('Dataset contains string values')
                    self.clean_dataset_numeric = self.clean_dataset.select_dtypes(exclude=['object'])
                    self.x = self.clean_dataset_numeric.iloc[:, :-1].values
                    self.y = self.clean_dataset_numeric.iloc[:, -1].values

                if len(self.clean_dataset) > 100000:
                    try:
                        self.estimation = LinearSVC()
                        print('LinearSVC is the best choice')
                        self.estimation.fit(self.x, self.y)
                        report = 'LinearSVC'
                        print(f'The best algorithm to use of a dataset this size is {report}')
                        self.report.append(report)
                    except:
                        if self.clean_dataset.dtypes == 'object':  # if string
                            # use Naive Bayes
                            self.estimation = GaussianNB()
                            self.estimation.fit(self.x, self.y)
                            report = 'GaussianNB'
                            print(f'The best algorithm to use of a dataset this size is {report}')
                            self.report = report

                        else:
                            try:
                                self.estimation = KNeighborsClassifier()
                                self.estimation.fit(self.x, self.y)
                                report = 'KNeighborsClassifier'
                                print(f'The best algorithm to use of a dataset this size is {report}')
                                self.report = report

                            except:
                                self.estimation = SVC()
                                self.estimation.fit(self.x, self.y)
                                report = 'SVC'
                                print(f'The best algorithm to use of a dataset this size is {report}')
                                self.report.append(report)

                else:
                    try:
                        self.estimation = LinearSVC()
                        self.estimation.fit(self.x, self.y)
                        report = 'LinearSVC'
                        print(f'The best algorithm to use of a dataset this size is {report}')
                        self.report.append(report)
                    except:
                        # kernel approximation
                        self.estimation = SVC()
                        self.estimation.fit(self.x, self.y)
                        report = 'SVC'
                        print(f'The best algorithm to use of a dataset this size is {report}')
                        self.report.append(report)

        else:
            print('Dataset is too small')
            self.impute_data()

    def unsupervised_ml(self):
        print('Unsupervised ML')

    def reinforcement_ml(self):
        print('Reinforcement ML')

    def impute_data(self):
        pass


if __name__ == '__main__':
    ML = Choose_ML()
    ML.ml_type()
    print(ML.report)
