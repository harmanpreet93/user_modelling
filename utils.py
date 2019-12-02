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


def fetch_data_from_csv(path, dtype):
    global df_output
    df_output = pd.read_csv(path + "Profile/Profile.csv")
    
    if dtype == "face":
        df_face = pd.read_csv(path + "Image/oxford.csv")
        df_face = pd.merge(df_face, df_output, left_on="userId", right_on="userid", how="outer")

        # Dropping duplicate face_ids
        df_face.drop_duplicates(subset ="userId", keep = "first", inplace = True)

        # Filling mean values for users with no face
        df_face.fillna(df_face.mean(), inplace=True)

        return df_face, df_output

    elif dtype == "text"
        df_liwc = pd.read_csv(path + "Text/liwc.csv")
        df_nrc = pd.read_csv(path + "Text/nrc.csv")

        # Merging nrc with liwc, because individually nrc doesnt seem to perform well.
        df_liwc = pd.merge(df_liwc, df_nrc, left_on="userId", right_on="userId")

        df_text = pd.merge(df_liwc, df_output, left_on="userId", right_on="userid")
        return df_text, df_output
    
    elif dtype == "relation"
        df_relation = pd.read_csv(path + "Relation/Relation.csv")

        # Cant merge with outputs here because it is not tidy data.
        return df_relation, df_output

    else:
        raise ValueError("Invalid dtype - Should be one of (face, text, relation)")
    

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
    return new_df_pivot


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