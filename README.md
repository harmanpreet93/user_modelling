## User Modelling

We tried multiple approaches and the final model is in the model_final.py where we have combined all three different modalities. 

All the below features were normalized using Min Max Scaling. We also used SelectKBest methods to select the top features for face and text data.  

1. Face Data (Imputed mean values for users with no face data and used the first face when a user had multiple face data)
2. Text Data (Concatenated the liwc and nrc data together after selecting the best features among them)
3. Relation Data (got node2vec embeddings for pages with more than 5 users, and averaged the page embeddings for each user), we could also use user embeddings but then we would have to retrain the  graph while testing which wasnt feasible.

`X = pd.concat[face, text, relation_n2v]`

### Gender Classification:
All the above three data sources were concatenated and trained using XGBoost to predict gender.

### Age Classification:
All the above three data sources were concatenated alongwith the prediction for gender and then trained using XGBoost to predict age.

### Personality Prediction:
All the above three data sources were concatenated alongwith the prediction for gender and age.
We used regression chaining where the prediction for one personality trait is again fed back and used for predicting the another personality trait using XGBRegressor. 
The below order for chaining was selected after multiple experiments   
[openness, ]
  

`python model_final.py`
