import os
import pandas as pd
import subprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import utils
import preprocessing

n2v_install_dir = os.path.dirname(os.path.abspath(__file__))
project_path = "data/"

"""Read features"""
df_output = pd.read_csv(project_path + "Train/Profile/Profile.csv")
df_face = pd.read_csv(project_path + "Train/Image/oxford.csv")
df_liwc = pd.read_csv(project_path + "Train/Text/liwc.csv")
df_liwc.rename(columns={"anger": "nrc_anger"}, inplace=True)  # because anger comes both in nrc and liwc
df_nrc = pd.read_csv(project_path + "Train/Text/nrc.csv")
df_relation = pd.read_csv(project_path + "Train/Relation/Relation.csv")

df_text_ = pd.merge(df_liwc, df_nrc, left_on="userId", right_on="userId")
df_text = pd.merge(df_text_, df_output, left_on="userId", right_on="userid")
df_liwc = pd.merge(df_liwc, df_output, left_on="userId", right_on="userid")
df_nrc = pd.merge(df_nrc, df_output, left_on="userId", right_on="userid")

# drop users with multiple faces, keeping only the first face
df_face.drop_duplicates(subset="userId", keep="first", inplace=True)
df_face = pd.merge(df_face, df_output, left_on="userId", right_on="userid", how="outer")
del df_face["userId"]
df_face.rename(columns={"userid": "userId"}, inplace=True)
# since there were missing faces, fill mean face in place of no-faces
df_face.fillna(df_face.mean(), inplace=True)

X_age_face_train, y_age_face_train = utils.extract_data(df_face, label="age")

# Min Max scale features
X_age_face_train = preprocessing.MinMaxScaleDataframe(X_age_face_train)
X_age_text_train, y_age_text_train = utils.extract_data(df_text, label="age")
X_age_text_train = preprocessing.MinMaxScaleDataframe(X_age_text_train)

"""Code"""
print("Pre-processing data...\n")
# remove pages with count less than threshold (Note: this removes few users as well)
threshold = 5

page_like_count = df_relation.groupby(['like_id']).size()
df_relation['likes_count'] = df_relation['like_id'].apply(lambda x: page_like_count.get(x))
df_relation_filtered = df_relation[df_relation['likes_count'] > threshold]

x = df_relation_filtered.copy()
x.like_id = x.like_id.apply(str)
a = x["like_id"].value_counts().reset_index()
page_ids = {}
for page in a[a["like_id"] > threshold]['index'].tolist():
  page_ids[page] = 1


def filter_page(x):
  a = []
  for y in x.like_id:
    if y not in page_ids:
      continue
    else:
      a.append(y)
  return ' '.join(a)


df_tfidf = pd.DataFrame({"likes": x.groupby("userid").apply(filter_page)})

vectorizer = TfidfVectorizer(lowercase=False)
vectorizer.fit(df_tfidf["likes"])
tf_idf_X = vectorizer.transform(df_tfidf["likes"])

userid, like_id, weight = [], [], []

for index, (row, col) in enumerate(df_tfidf.iterrows()):
  a = tf_idf_X[index].toarray()
  for c in col["likes"].split():
    idx = vectorizer.vocabulary_[c]
    userid.append(row)
    like_id.append(int(c))
    weight.append(a[0][idx])

page_weights_df = pd.DataFrame({'userid': userid, 'like_id': like_id, 'page_weight': weight})

# important features were recognised by running various feature selection methods
# feel free to add/subtract features
imp_facial_features = ['facialHair_mustache', 'facialHair_beard', 'facialHair_sideburns', 'faceRectangle_width',
                       'faceRectangle_height']
imp_liwc_features = ['ipron', 'swear', 'social', 'negemo', 'feel', 'money']
imp_nrc_features = ['negative', 'anger', 'disgust', 'fear', 'joy', 'anticipation']

le = LabelEncoder()
le.fit(list(df_relation_filtered["userid"]) +
       list(df_relation_filtered["like_id"]) +
       imp_facial_features +  # oxford
       imp_liwc_features +  # liwc
       imp_nrc_features  # nrc
       )

merged = pd.merge(df_relation_filtered, df_output, left_on='userid', right_on='userid')
dummy_df = X_age_face_train.copy()
dummy_df['userid'] = df_face["userId"]
merged = pd.merge(merged, dummy_df[imp_facial_features + ['userid']], left_on='userid', right_on='userid')
dummy_df = X_age_text_train.copy()
dummy_df['userid'] = df_face["userId"]
merged = pd.merge(merged, dummy_df[imp_liwc_features + imp_nrc_features + ['userid']], left_on='userid',
                  right_on='userid')
merged.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y", "ope", "con", "ext", "agr", "neu"], inplace=True)
del dummy_df

# add age weights columns
merged = pd.merge(merged, page_weights_df, left_on=['userid', 'like_id'], right_on=['userid', 'like_id'])

merged['int_userid'] = le.transform(merged["userid"])
merged['int_like_id'] = le.transform(merged["like_id"])

all_imp_features = imp_facial_features + imp_liwc_features + imp_nrc_features
for feature in all_imp_features:
  merged['int_' + feature] = le.transform([feature])[0]

# formatize data as required by node2vec
relevant_id_cols = ['int_like_id']
relevant_id_cols_with_weights = ['int_' + feature for feature in all_imp_features]
all_relvant_cols = relevant_id_cols + relevant_id_cols_with_weights

melted_df = pd.DataFrame()
for relevant_id_col in all_relvant_cols:
  tmp_df = pd.melt(merged, id_vars=['int_userid'], value_vars=[relevant_id_col], value_name='int_node')
  tmp_df.drop(columns=['variable'], inplace=True)
  if relevant_id_col in relevant_id_cols_with_weights:
    tmp_df["weight"] = merged[relevant_id_col.split('int_')[1]]
  else:
    tmp_df["weight"] = merged['page_weight']
  melted_df = pd.concat([melted_df, tmp_df])

# drop duplicates here
melted_df.drop_duplicates(inplace=True)
melted_df = melted_df[melted_df['weight'] != 0.0]

directory = n2v_install_dir + '/graph'
if not os.path.exists(directory):
  os.makedirs(directory)
melted_df.to_csv(n2v_install_dir + "/graph/hetro_relations_weighted_pages.edgelist",
                 sep=" ",
                 header=False,
                 index=False)

print("Installing Node2vec...\n")
# Build Node2vec in data folder
clone = "git clone https://github.com/snap-stanford/snap.git"
subprocess.call(clone.split())
os.chdir("snap/examples/node2vec")
make_all = "make all"
subprocess.call(make_all.split())

directory = n2v_install_dir + '/emb'
if not os.path.exists(directory):
  os.makedirs(directory)

# Train Node2vec
print("Training Node2vec...\n")
train_command = './node2vec -i:"' + n2v_install_dir + '/graph/hetro_relations_weighted_pages.edgelist" -o:"' + n2v_install_dir + '/emb/hetro_relations_weighted_pages.emb" -l:4 -r:2 -k:3 -d:128 -p:0.8 -q:0.9 -v -e:1 -w'
subprocess.run(train_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
# ! ./node2vec -i:"/content/drive/My Drive/ML_data/DS_data/data/graph/hetro_relations_weighted_pages_16thDec.edgelist" -o:"/content/drive/My Drive/ML_data/DS_data/data/emb/hetro_relations_weighted_pages_16thDec.emb" -l:60 -r:15 -k:10 -d:128 -p:0.8 -q:0.9 -v -e:10 -w


print("Node2vec training done, processing embeddings...")
emb_path = n2v_install_dir + "/emb/hetro_relations_weighted_pages.emb"
user_emb_path = n2v_install_dir + "/emb/hetro_users_weighted_pages.emb"
pages_emb_path = n2v_install_dir + "/emb/hetro_pages_weighted_pages.emb"

# separate out users and page embeddings
int_userids = set(merged["int_userid"])
int_like_ids = set(merged["int_like_id"])

with open(emb_path, "r") as fin, open(user_emb_path, "w") as fout_user, open(pages_emb_path, "w") as fout_pages:
  for i, line in enumerate(fin.readlines()):
    if i == 0:
      pass
    embedding = line.split()
    int_id = int(embedding[0])
    embedding = " ".join([x for x in embedding[1:]])
    # embedding = [float(x) for x in embedding[1:]]
    if int_id in int_userids:
      user_id = le.inverse_transform([int_id])
      line_to_write = str(user_id[0]) + " " + embedding + "\n"
      fout_user.write(line_to_write)
    elif int_id in int_like_ids:
      like_id = le.inverse_transform([int_id])
      line_to_write = str(like_id[0]) + " " + embedding + "\n"
      fout_pages.write(line_to_write)

user_emb = pd.read_csv(user_emb_path, sep=" ", header=None, index_col=0)

# fill mean embedding for missing users
user_emb_ = pd.merge(user_emb, df_output, left_on=user_emb.index, right_on="userid", how="outer")
user_emb_.drop(columns=['Unnamed: 0', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'], inplace=True)
user_emb_.fillna(user_emb.mean(), inplace=True)
user_emb_.set_index('userid', inplace=True)

user_emb_.to_csv(n2v_install_dir + "/emb/all_user_hetro_emb__weighted_pages.csv", header=False)
