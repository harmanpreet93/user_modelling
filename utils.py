"""
__author__ = "Harmanpreet Singh, Akshay Singh Rana, Himanshu Arora"
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Class Project"
"""


import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.preprocessing import MinMaxScaler

project_path = "data"
regressor_labels = ['ope', 'con', 'ext', 'agr', 'neu']
df_output = pd.read_csv(project_path + "Train/Profile/Profile.csv")


def buckets(x):
    if x <= 24:
        return "xx-24"
    elif x > 24 and x <= 34:
        return "25-34"
    elif x > 34 and x <= 49:
        return "35-49"
    elif x > 49:
        return "50-xx"


def fetch_data_from_csv(path):
    global df_output
    df_face = pd.read_csv(path + "Image/oxford.csv")
    df_output = pd.read_csv(path + "Profile/Profile.csv")
    df_liwc = pd.read_csv(path + "Text/liwc.csv")
    df_nrc = pd.read_csv(path + "Text/nrc.csv")
    df_relation = pd.read_csv(path + "Relation/Relation.csv")

    # Merge with df_output to get labels on user_id..
    df_face = pd.merge(df_face, df_output, left_on="userId", right_on="userid")
    df_liwc = pd.merge(df_liwc, df_nrc, left_on="userId", right_on="userId")
    df_text = pd.merge(df_liwc, df_output, left_on="userId", right_on="userid")
    return df_text, df_face, df_relation, df_output


# clean irrelevant columns in data
def clean_dataframe(df):
	list_to_exclude = regressor_labels + ['age','gender'] + ['userId','faceID', 'userid','Unnamed', 'Unnamed: 0','headPose_pitch']
	return df[df.columns[~df.columns.isin(list_to_exclude)]]


# Transform relation data into sparse matrix
def get_transformed_relation(df_relation, min_likes):
    a = df_relation["like_id"].value_counts().reset_index()
    page_ids = a[a["like_id"] > min_likes]['index'].tolist()
    new_df = df_relation[df_relation['like_id'].isin(page_ids)]
    new_df_pivot = new_df.pivot(index="userid", columns="like_id",values="like_id")
    new_df_pivot[new_df_pivot.notna()] = 1
    new_df_pivot = new_df_pivot.fillna(0)
    new_df_pivot = new_df_pivot.reset_index()
    df_relation_matrix = pd.merge(new_df_pivot, df_output, left_on="userid", right_on="userid")
    return df_relation_matrix


# Extract outputs from merged values
def extract_data(df, label):
    if label == "age":
        return clean_dataframe(df), df['age'].apply(buckets)
    elif label == "gender":
        return clean_dataframe(df), df['gender']
    elif label == "personality":
        return clean_dataframe(df), df.loc[:, regressor_labels]


def MinMaxScaleDataframe(df):
    x = df.values   # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns)


def selectKFeatures(df, y, k, regression=False):
    if regression:
        kbest = SelectKBest(f_regression, k=k)
    else:
        kbest = SelectKBest(chi2, k=k)
    kbest.fit(df, y)
    # Get columns to keep
    mask = kbest.get_support()
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, df.columns):
        if bool:
            new_features.append(feature)
    return new_features