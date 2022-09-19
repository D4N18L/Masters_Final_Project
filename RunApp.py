
from preprocess import Handling_Data , Merge_Datasets, Choose_ML_Model
import pandas as pd

if __name__ == '__main__':

    if_merged = input('Have you merged your datasets? \n'
                      'Type yes \n'
                      'or'
                      'Type no \n'
                      'Enter your choice: ')

    if if_merged == 'yes':
        print('Great, lets get started. \n')
        print("Please enter the path to your merged dataset:")
        path = input()
        dataset = pd.read_csv(path)
        data_check = Handling_Data(dataset)
        data_check.data_cleaning()
        data_check.data_reduction()
        data_check.data_visualization()
        data_check.data_report()
        data_check.export_clean_data()
        exit()

        # Time to work on Choose Ml_Model
        ml_model = Choose_ML_Model()
        ml_model.split_data()
        ml_model.choose_ml_type()
        ml_model.choose_ml_learning()
        ml_model.ml_estimator()
        ml_model.ml_predict()

    elif if_merged == 'no':
        print('Please merge your datasets and try again')

        merge_data = Merge_Datasets(x_data=pd.read_csv('x_data.csv'), y_data=pd.read_csv('y_data.csv'))
        merge_data.merge_datasets()
        data_check = Handling_Data(data=pd.read_csv('merged_data.csv'))
        data_check.data_cleaning()
        data_check.data_transformation()
        data_check.data_reduction()
        data_check.data_visualization()

        # Time to work on Choose Ml_Model
        ml_model = Choose_ML_Model()
        ml_model.split_data()
        ml_model.choose_ml_type()
        ml_model.choose_ml_learning()
        ml_model.ml_estimator()
        ml_model.ml_predict()

    else:
        print('Invalid choice, please try again')
        sys.exit()