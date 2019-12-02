from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm

import pandas as pd
import pickle
import preprocessing
import utils

from typing import List

class Model2LateFusion:
    '''
    This class is used to code the second approach in the presentation
    of where we fuse data outputs together
    '''

    def __init__(self):
        pass

    def fetch_face_data(self):
        df_face, df_output = utils.load_data_from_csv(dtype="face")
        return df_face
    
    def fetch_text_data(self):
        df_text, df_output = utils.load_data_from_csv(dtype="text")
        return df_text
    
    def fetch_relation_data(self):
        df_relation, df_output = utils.load_data_from_csv(dtype="relation")

        # Getting sparse matrix based on page likes
        df_relation_matrix = utils.get_transformed_relation(df_relation, min_likes = 5)
        
        df_relation_matrix = pd.merge(df_relation_matrix, df_output, 
                                        left_on="userid", 
                                        right_on="userid", 
                                        how="outer")

        # Filling mean values for users with no page likes (among the pages selected) 
        df_relation_matrix.fillna(df_relation_matrix.mean(), inplace=True)
        return df_relation_matrix
    
    def fetch_node2vec_data(self):
        df_relation, df_output = utils.load_data_from_csv(dtype="relation")
        df_n2v = ""
        return df_n2v

    def combine_features(self, data: List):
        '''
        Combining features for early fusion.. 
        '''
        for i in data:
            df += pd.merge(i)
        return combined_df

    def combine_outputs():
        pass



def build_model_and_evaluate(data: List[str], target: str, classifier="XGB"):
    model = Model2LateFusion()
    
    df_X = combine_features(data)

    X, y = utils.extract_data(df_X, label=target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2)
    
    if classifier == "xgb":
        clf = XGBClassifier(n_estimators=200)
    elif classifier == "svm":
        clf = SGDClassifier()
    else:
        raise ValueError("Incorrect classifier")

    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return accuracy_score


def build_model_and_evaluate_rms(data: List[str], regressor="XGB"):
    model = Model2LateFusion()
    
    df_X = combine_features(data)

    X, y = utils.extract_data(df_X, label="personality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2)
    
    if regressor == "xgb":
        reg = MultiOutputRegressor(XGBRegressor(n_estimators=200,
                                                 max_depth=2, 
                                                objective="reg:squarederror"))
    elif regressor == "rf":
        reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))

    elif regressor == "lasso":
        reg = ""

    elif regressor == "lightgbm":
        reg = MultiOutputRegressor(lightgbm.LGBMRegressor(objective = "regression")) 
    else:
        raise ValueError("Incorrect classifier")

    reg = reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # Calculating RMSE for all personality
    rmse = []
    for i,value in enumerate(utils.regressor_labels):
        rmse.append(sqrt(mean_squared_error(y_pred[:,i], y_test[value])))

    return rmse



if __name__ == "__main__":

    ## Classification Tasks for combined datasets

    accuracy_face_text_age, clf = build_model_and_evaluate(
                                data = ["face","text"] ,
                                target = "age")
    pickle.dump(clf, open("face_text-age.pkl", 'wb'))

    accuracy_face_text_gender, clf = build_model_and_evaluate(
                                data = ["face","text"],
                                target = "gender")
    pickle.dump(clf, open("face_text-gender.pkl", 'wb'))

    accuracy_face_relation_age, clf = build_model_and_evaluate(
                                data = ["face", "relation"], 
                                target = "age")
    pickle.dump(clf, open("face_relation-age.pkl", 'wb'))

    accuracy_face_relation_gender, clf = build_model_and_evaluate(
                                data = ["face", "relation"], 
                                target = "gender")
    pickle.dump(clf, open("face_relation-gender.pkl", 'wb'))

    accuracy_text_relation_age, clf = build_model_and_evaluate(
                                data = ["text", "relation"], 
                                target = "age")
    pickle.dump(clf, open("text_relation-age.pkl", 'wb'))

    accuracy_text_relation_gender, clf = build_model_and_evaluate(
                                data = ["text", "relation"], 
                                target = "gender")
    pickle.dump(clf, open("text_relation-gender.pkl", 'wb'))


    # Combining all three data sources to predict age and gender

    accuracy_face_text_relation_age, clf = build_model_and_evaluate(
                                data = ["face", "text", "relation"], 
                                target = "age")
    pickle.dump(clf, open("face_text_relation-age.pkl", 'wb'))

    accuracy_face_text_relation_gender, clf = build_model_and_evaluate(
                                data = ["face", "text", "relation"], 
                                target = "gender")
    pickle.dump(clf, open("face_text_relation-gender.pkl", 'wb'))



    ## Regression Tasks

    rmse_face_text_personality, clf = build_model_and_evaluate_rms(
                                data = ["face", "text"]) 
    pickle.dump(clf, open("face_text_regression.pkl", 'wb'))

    rmse_face_relation_personality, clf = build_model_and_evaluate_rms(
                                data = ["face", "relation"]) 
    pickle.dump(clf, open("face_relation_regression.pkl", 'wb'))

    rmse_text_relation_personality, clf = build_model_and_evaluate_rms(
                                data = ["text", "relation"]) 
    pickle.dump(clf, open("face_text_regression.pkl", 'wb'))

    # Combining all three data sources to predict prediction

    rmse_face_text_relation_personality, clf = build_model_and_evaluate_rms(
                                data = ["face", "text", "relation"]) 
    pickle.dump(clf, open("face_text_relation_regression.pkl", 'wb'))
    

    






