import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from exception import custom_exception
from logger import logging
from utils import save_obj

@dataclass
class datatransformationconfig:
    perprpcessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class datatransformtion:
    def __init__(self):
        self.data_transformation_config=datatransformationconfig

###the funtion below is responsible for the date transformation.
    def get_data_transfer_obj(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy ="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("The scaling of the numerical features is completed.")
            logging.info("The encoding of the categorical features is completed.")

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns),
            ])

            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
         try:
              train_df = pd.read_csv(train_path)
              test_df = pd.read_csv(test_path)

              logging.info("The training and test data has been read successfully.")
              logging.info("obtaining preprocessing object.")

              preprocessing_obj =self.get_data_transfer_obj()
              target_column_name = "math_score"
              numerical_columns = ["writing_score","reading_score"]
              
              input_features_train_df = train_df.drop(columns=[target_column_name])
              target_feature_train_df = train_df[target_column_name]

              input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
              target_feature_test_df = test_df[target_column_name]

              logging.info("Applying preprocessing to the train and test dataframes")
              
              input_feature_train_array = preprocessing_obj.fit_transform(input_features_train_df)
              input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

              train_arr = np.c_[
                   input_feature_train_array, np.array(target_feature_train_df)
              ]
              test_arr = np.c_[
                   input_feature_test_array , np.array(target_feature_test_df)
              ]
              
              logging.info("saved presprocessed objects")

              ### saving the preprocessor as .pkl file

              save_obj(
                   file_path = self.data_transformation_config.perprpcessor_obj_file_path,
                   obj = preprocessing_obj
              )

              return(
                   train_arr,
                   test_arr,
                   self.data_transformation_config.perprpcessor_obj_file_path
              )
         except:
                pass