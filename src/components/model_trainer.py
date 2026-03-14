import os
import sys
import dataclasses as dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import custom_exception
from logger import logging
from utils import evaluate_models, save_obj

@dataclass
class modeltrainerconfig:
    model_trainer_file_path = os.path.join("artifacts","model.pkl")

class modeltrainer:
    def __init__(self):
        self.model_trainer_config= modeltrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Model Training has been started.")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models={
                "randomforest":RandomForestRegressor(),
                "decisiontree":DecisionTreeRegressor(),
                "gradientboosting":GradientBoostingRegressor(),
                "linear":LinearRegression(),
                "kneighbors":KNeighborsRegressor(),
                "xgboost":XGBRegressor(),
                "catboost":CatBoostRegressor(verbose=False),
                "adaboost":AdaBoostRegressor(),
            }
            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Exception("no model is good enough")
            
            logging.info("Best model found.")
            
            save_obj(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            raise custom_exception(e,sys)