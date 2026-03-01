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
                    ("imputer",SimpleImputer(strategy="median")),
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
        
        