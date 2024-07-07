from catboost import CatBoostClassifier, Pool

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, matthews_corrcoef, log_loss, roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import warnings 
warnings.filterwarnings('ignore')

# DATA CLEANING & FILTERING ---------------------------------------------------------------------------------------
data_df = pd.read_csv("path/to/data.csv") 
pejaver19_filename = 'path/to/Pejaver_supMat_mmc2_clinvar2019.xlsx'
pejaver19 = pd.read_excel(pejaver19_filename)
pejaver20_filename = 'path/to/data/Pejaver_supMat_mmc4_clinvar2020.xlsx'
pejaver20 = pd.read_excel(pejaver20_filename)

pejaver19vars = pejaver19[['genename', 'aavar']].apply(lambda row: f'{row[0]} {row[1]}', axis=1) 
pejaver20vars = pejaver20[['genename', 'aavar']].apply(lambda row: f'{row[0]} {row[1]}', axis=1) 
vars = list(pejaver19vars) + list(set(pejaver20vars) - set(pejaver19vars))

cols_to_clean = ['ft7','ft8', 'ft9','ft10', 
              'ft11','ft12', 'ft13', 'ft14']
for col in cols_to_clean:
    data_df.loc[data_df.ft6 == 0, col] = 0.0 # if cm_index is 0, the value of features in cols_to_clean is 0

features = ['ft1', 'ft2','ft3','ft4', 'ft5',
            'ft6','ft7', 'ft8','ft9','ft10', 'ft11',
            'ft12', 'ft13','ft14']
topredict = 'clinsig_binary'
protein = 'uniprot'

data_df['variant_id'] = data_df[['genename', 'variant']].apply(lambda row: f'{row[0]} {row[1]}', axis=1)
data_df = data_df[[topredict] + [protein]  + features + ['variant_id']]
data_df['clinsig_binary'] = data_df['clinsig_binary'].replace({'LB/B': 0, 'LP/P': 1})

# TRAIN & TEST ---------------------------------------------------------------------------------------
SEED = 42

params = {'loss_function':'Logloss', # objective function
          'eval_metric':'AUC', # metric
          'early_stopping_rounds': 100, # stop if the metric does not improve
          'verbose': 100, # output to stdout
          'random_seed': SEED # for reproducibility
         }

unique_proteins = data_df[protein].unique() # get unique proteins

# store predictions and probabilities
all_predictions = []
all_pred_probs = []
all_test = []

# perform One-Protein-Out Cross-Validation
for test_protein in unique_proteins:
    # Filter train and test data based on trainingvars and testingvars
    train_df = data_df[data_df.variant_id.isin(vars) & (data_df[protein] != test_protein)].reset_index(drop=True)
    test_df = data_df[data_df.variant_id.isin(vars) & (data_df[protein] == test_protein)].reset_index(drop=True)

    # Separate features and target variables
    X_train = train_df[features]
    y_train = train_df[topredict]
    X_test = test_df[features]
    y_test = test_df[topredict]
    all_test.extend(y_test)

    # Train the model
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, plot=False)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    # Store predictions and predicted probabilities
    all_predictions.extend(y_pred)
    all_pred_probs.extend(y_pred_prob[:, 1])

# PERFORMANCE ESTIMATION ---------------------------------------------------------------------------------------
accuracy = accuracy_score(all_test, all_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

mcc = matthews_corrcoef(all_test, [1 if prob >= 0.5 else 0 for prob in all_pred_probs])
print("Matthews Correlation Coefficient: %.3f" % mcc)

roc_auc = roc_auc_score(all_test, all_pred_probs)
print("Area Under the Receiver Operating Characteristic Curve: %.2f" % roc_auc)

logloss = log_loss(all_test, all_pred_probs)
print("Log Loss: %.3f" % logloss)

# EXPORT MODEL DATA ---------------------------------------------------------------------------------------
model_df = pd.DataFrame()
model_df['actual_class'] = data_df['clinsig_binary']
model_df['actual_class'] = model_df['actual_class'].replace({0: -1})
model_df['pred'] = all_predictions
model_df['pred'] = model_df['pred'].replace({0: -1})
model_df['pred_prob'] = all_pred_probs

file_name = 'catboost_lopo.csv'
model_df.to_csv(file_name, index=False)

# DATA VISUALIZATION ---------------------------------------------------------------------------------------
plt.figure(figsize=(9, 8))
sns.set(font_scale=1.5, style='white')
my_pal = {"1": "firebrick", "-1": "forestgreen"}
sns.violinplot(data=model_df, x="actual_class", y="pred_prob", palette=my_pal)