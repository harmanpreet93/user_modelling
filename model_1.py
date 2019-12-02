
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklear.linear_model import SGDClassifier
from xgboost import XGBClassifier

import pandas as pd
import pickle
import preprocessing
import utils


class Model1:
    '''
    This class is used to code the first approach in the presentation
    of using each data source for every challenge.
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
        df_n2v = ""
        return df_n2v


def build_model_and_evaluate(data, target, classifier="XGB"):
    model = Model1()
    
    if data == "face":
        df_X = model.fetch_face_data()
    elif data == "text":
        df_X = model.fetch_text_data()
    elif data == "relation":
        df_X = model.fetch_relation_data()
    else:
        raise ValueError("Incorrect data format")

    X, y = utils.extract_data(df_X, label=target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2)
    
    if clf == "XGB":
        clf = XGBClassifier(n_estimators=200)
    elif clf == "SVM":
        clf = SGDClassifier()
    else:
        raise ValueError("Incorrect classifier")

    clf = clf.fit(X_train, y_train)
    score = accuracy_score(y_test, y_pred)

    return accuracy_score

if __name__ == "__main__":
    accuracy_face_age, clf = build_model_and_evaluate(
                                data = "face", 
                                target = "age")
    pickle.dump(clf, open("face_age.pkl", 'wb'))

    accuracy_face_gender, clf = build_model_and_evaluate(
                                data = "face", 
                                target = "gender")
    pickle.dump(clf, open("face_gender.pkl", 'wb'))

    accuracy_text_age, clf = build_model_and_evaluate(
                                data = "text", 
                                target = "age")
    pickle.dump(clf, open("text_age.pkl", 'wb'))

    accuracy_text_gender, clf = build_model_and_evaluate(
                                data = "text", 
                                target = "gender")
    pickle.dump(clf, open("text_gender.pkl", 'wb'))

    accuracy_relation_age, clf = build_model_and_evaluate(
                                data = "relation", 
                                target = "age")
    pickle.dump(clf, open("relation_age.pkl", 'wb'))

    accuracy_relation_gender, clf = build_model_and_evaluate(
                                data = "relation", 
                                target = "gender")
    pickle.dump(clf, open("relation_gender.pkl", 'wb'))

    

    






