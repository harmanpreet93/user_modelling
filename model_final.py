from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from xgboost import XGBClassifier, XGBRegressor
from preprocessing import preprocess_data as preprocess

import lightgbm

import pandas as pd
import pickle
import preprocessing
import utils

from typing import List


class Model3:
  '''
  This class is used to code our final approach for the submission where
  we fuse data sources together with predictions from previous task.
  '''

  def __init__(self):
    pass

  def get_face_data(self):
    df_face, _ = utils.load_data_from_csv(dtype="face")
    df_face = preprocess(df_face, dtype="face")
    return df_face

  def get_text_data(self):
    df_text, _ = utils.load_data_from_csv(dtype="text")
    df_text = preprocess(df_text, dtype="text")
    return df_text

  def get_relation_data(self):
    df_relation, df_output = utils.load_data_from_csv(dtype="relation")

    # Getting sparse matrix based on page likes
    df_relation_matrix = utils.get_transformed_relation(df_relation, min_likes=5)

    df_relation_matrix = pd.merge(df_relation_matrix, df_output,
                                  left_on="userid",
                                  right_on="userid",
                                  how="outer")

    # Filling mean values for users with no page likes (among the pages selected)
    df_relation_matrix.fillna(df_relation_matrix.mean(), inplace=True)
    return df_relation_matrix

  def get_node2vec_data(self):
    '''
    Need to work on it. Get page embeddings #Harman
    '''
    df_relation, df_output = utils.load_data_from_csv(dtype="relation")
    df_n2v = ""
    return df_n2v

  def combined_features(self):
    X_face = self.get_face_data()
    X_text = self.get_text_data()
    X_n2v = self.get_node2vec_data()
    X_combined = pd.concat([X_face, X_text, X_n2v])
    return X_combined


def build_model_and_evaluate(target: str, prev_pred=None):
  model = Model3()
  X_combined = model.combined_features()

  # combining the prediction of previous tasks to predict another task
  if prev_pred is not None:
    X_combined = pd.concat([X_combined, prev_pred])

  X, y = utils.extract_data(X_combined, label=target)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

  clf = XGBClassifier(n_estimators=200)
  clf = clf.fit(X_train, y_train)

  y_pred_test = clf.predict(X_test)
  y_pred_train = clf.predict(X_train)

  score = accuracy_score(y_test, y_pred_test)
  return accuracy_score, clf, y_pred_train


def build_model_and_evaluate_rms(prev_pred=None):
  model = Model3()
  X_combined = model.combined_features()

  # combining the prediction of previous tasks to predict another task
  if prev_pred is not None:
    X_combined = pd.concat([X_combined, prev_pred])

  X, y = utils.extract_data(X_combined, label="personality")
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

  reg = RegressorChain(XGBRegressor(n_estimators=200,
                                    max_depth=2,
                                    objective="reg:squarederror"),

                       order=[1, 2, 3, 4, 5])

  reg = reg.fit(X_train, y_train)
  y_pred = reg.predict(X_test)

  # Calculating RMSE for all personality
  rmse = []
  for i, value in enumerate(utils.regressor_labels):
    rmse.append(sqrt(mean_squared_error(y_pred[:, i], y_test[value])))

  return rmse, reg


if __name__ == "__main__":
  accuracy_gender, clf, y_gender = build_model_and_evaluate(target="gender")
  pickle.dump(clf, open("model_gender.pkl", 'wb'))

  accuracy_age, clf, y_age = build_model_and_evaluate(target="age", prev_pred=y_gender)
  pickle.dump(clf, open("model_age.pkl", 'wb'))

  rmse_personality, reg = build_model_and_evaluate_rms(prev_pred=pd.concat[y_gender, y_age])
  pickle.dump(reg, open("model_personality.pkl", 'wb'))

  # We can run this multiple times where the previous predictions are continously fed back to improve predictions but we didnt notice significant improvements.
