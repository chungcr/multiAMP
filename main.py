from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, hamming_loss, confusion_matrix, multilabel_confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
import pickle

def PAAC_CTDD_rename(s):
    s = s.replace('Xc1', 'PAAC')
    s = s.replace('Xc2', 'PAAC')
    s = s.replace('_PRAM900101', '1')
    s = s.replace('_ARGP820101', '2')
    s = s.replace('_ZIMJ680101', '3')
    s = s.replace('_PONP930101', '4')
    s = s.replace('_CASG920101', '5')
    s = s.replace('_ENGD860101', '6')
    s = s.replace('_FASG890101', '7')
    s = s.replace('.1.', '_C1_')
    s = s.replace('.2.', '_C2_')
    s = s.replace('.3.', '_C3_')
    s = s.replace('residue0', '001')
    s = s.replace('residue25', '025')
    s = s.replace('residue50', '050')
    s = s.replace('residue75', '075')
    s = s.replace('residue100', '100')
    
    return s


df_train = pd.read_csv('data/multiAMP_train.csv')
df_test = pd.read_csv('data/multiAMP_test.csv')

AAC = pd.read_csv('data/features/AAC.tsv', sep='\t')
AAC_cols = AAC.columns[1:].tolist()
PAAC = pd.read_csv('data/features/PAAC.tsv', sep='\t')
PAAC.columns = [PAAC.columns[0]] + [PAAC_CTDD_rename(x) for x in PAAC.columns[1:].tolist()]
PAAC_cols = PAAC.columns[1:].tolist()
CTDD = pd.read_csv('data/features/CTDD.tsv', sep='\t')
CTDD.columns = [CTDD.columns[0]] + [PAAC_CTDD_rename(x) for x in CTDD.columns[1:].tolist()]
CTDD_cols = CTDD.columns[1:].tolist()

target_labels = ['Antibacterial', 'MammalianCells', 'Antifungal', 'Antiviral', 'Anticancer']

trainX = df_train[AAC_cols+PAAC_cols+CTDD_cols].values
trainY = df_train[target_labels].values

testX = df_test[AAC_cols+PAAC_cols+CTDD_cols].values
testY = df_test[target_labels].values


def get_single_label_metrics(testY, scoreY, label, threshold=0.5): # scoreY是一維的
    predY = (scoreY>threshold)*1
    cm = confusion_matrix(testY, predY)
    _df = pd.DataFrame(columns=['label', 'tn', 'fp', 'fn', 'tp'])

    _df.loc[len(_df)] = [label] + list(cm.ravel())

    _df['acc'] = ((_df['tn']+_df['tp'])/(_df['tn']+_df['fp']+_df['fn']+_df['tp']))
    _df['pre'] = (_df['tp']/(_df['tp']+_df['fp']).replace(0, np.nan))
    _df['recall'] = (_df['tp']/(_df['tp']+_df['fn']).replace(0, np.nan))
    _df['auc'] = roc_auc_score(testY, scoreY)
    _df['f1'] = ((2*_df['pre']*_df['recall'])/(_df['pre']+_df['recall']).replace(0, np.nan))

    mcc_upper = (_df['tp']*_df['tn']-_df['fp']*_df['fn'])
    mcc_lower = (((_df['tp']+_df['fp'])*(_df['tp']+_df['fn'])*(_df['tn']+_df['fp'])*(_df['tn']+_df['fn']))**(1/2)).replace(0, np.nan)
    _df['mcc'] = (mcc_upper/mcc_lower).astype(float).round(4)
    _df[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']] = _df[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']].astype(float).round(4)
    
    return _df


def get_multi_label_metrics(testY, scoreY, labels_list, threshold=0.5): #scoreY是2維的
    predY = (scoreY>threshold)*1
    cm = multilabel_confusion_matrix(testY, predY)
    _df = pd.DataFrame(columns=['label', 'tn', 'fp', 'fn', 'tp'])
    for i in range(len(labels_list)):
        _df.loc[len(_df)] = [labels_list[i]] + list(cm[i].ravel())

    _df['acc'] = ((_df['tn']+_df['tp'])/(_df['tn']+_df['fp']+_df['fn']+_df['tp']))
    _df['pre'] = (_df['tp']/(_df['tp']+_df['fp']).replace(0, np.nan))
    _df['recall'] = (_df['tp']/(_df['tp']+_df['fn']).replace(0, np.nan))
    _df['auc'] = 0
    for i in range(len(labels_list)):
        _df.loc[i, 'auc'] = roc_auc_score(testY[:, i], scoreY[:, i])
    _df['f1'] = ((2*_df['pre']*_df['recall'])/(_df['pre']+_df['recall']).replace(0, np.nan))

        
    mcc_upper = (_df['tp']*_df['tn']-_df['fp']*_df['fn'])
    mcc_lower = (((_df['tp']+_df['fp'])*(_df['tp']+_df['fn'])*(_df['tn']+_df['fp'])*(_df['tn']+_df['fn']))**(1/2)).replace(0, np.nan)
    _df['mcc'] = (mcc_upper/mcc_lower).astype(float).round(4)
    
    _df.loc[len(_df)] = ['macro', np.nan, np.nan, np.nan, np.nan, _df['acc'].mean(), _df['pre'].mean(), _df['recall'].mean(), _df['auc'].mean(), _df['f1'].mean(), _df['mcc'].mean()]
    _df[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']] = _df[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']].astype(float).round(4)
    _df[['tn', 'fp', 'fn', 'tp']] = _df[['tn', 'fp', 'fn', 'tp']].round()
    
    return _df


###### RF_single ######
with open('models/rf_single_10fold.pickle', 'rb') as f:
    rf_single_10fold = pickle.load(f)

rf_single_10fold.groupby('label').mean().round(4).iloc[[0,4,2,3,1]]


rf_single_models = []
for index, label in enumerate(target_labels):
    with open(f'models/rf_single_{label}_model.pickle', 'rb') as f:
        rf_single_models.append(pickle.load(f))



rf_single_importances_df = pd.DataFrame()

for index, label in enumerate(target_labels):
    model = rf_single_models[index]
    scoreY = model.predict_proba(testX)[:,1]
    
    rf_single_importances_df[f'{label}_f'] = pd.Series(model.feature_importances_, index=AAC_cols+PAAC_cols+CTDD_cols).sort_values(ascending=False).index
    rf_single_importances_df[f'{label}_v'] = pd.Series(model.feature_importances_, index=AAC_cols+PAAC_cols+CTDD_cols).sort_values(ascending=False).values
    
    if index==0:
        rf_single_metrics_df = get_single_label_metrics(testY[:, index], scoreY, label)
    else:
        rf_single_metrics_df.loc[len(rf_single_metrics_df)] = get_single_label_metrics(testY[:, index], scoreY, label).values[0]


rf_single_topN_10fold = []
for index, label in enumerate(target_labels):
    _df = pd.read_csv(f'performances/rf_single_topN_{label}_10fold.csv')
    rf_single_topN_10fold.append(_df)

rf_single_10fold.groupby('label').mean().round(4).reset_index().iloc[[0,4,2,3,1]]


rf_single_best_topN_10fold_metrics = pd.DataFrame()
for index, label in enumerate(target_labels):
    _df = rf_single_topN_10fold[index].groupby('label').mean().reset_index()
    __df = rf_single_10fold.groupby('label').mean().reset_index()
    _all_features_auc = round(__df[__df['label']==label]['auc'].values.reshape(-1)[0],4)

    rf_single_best_topN_10fold_metrics = pd.concat([rf_single_best_topN_10fold_metrics, pd.DataFrame(_df[_df['auc']>=_all_features_auc].iloc[0]).T])

rf_single_best_topN_10fold_metrics[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']] = rf_single_best_topN_10fold_metrics[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']].astype(float).round(4)
rf_single_best_topN_10fold_metrics = rf_single_best_topN_10fold_metrics.reset_index(drop=True)
rf_single_best_topN_10fold_metrics


rf_single_best_topN_testing_metrics = pd.DataFrame()
for index, label in enumerate(target_labels):
    topN = int(rf_single_best_topN_10fold_metrics.loc[index, 'label'].split('_')[-1])
    topN_trainX = pd.DataFrame(trainX, columns=AAC_cols+PAAC_cols+CTDD_cols)[rf_single_importances_df.loc[0:topN-1,f'{label}_f'].tolist()].values
    topN_testX = pd.DataFrame(testX, columns=AAC_cols+PAAC_cols+CTDD_cols)[rf_single_importances_df.loc[0:topN-1,f'{label}_f'].tolist()].values
    
    model = RandomForestClassifier(n_estimators=100, random_state=670, n_jobs=4)
    model.fit(topN_trainX, trainY[:, index])
    scoreY = model.predict_proba(topN_testX)[:,1]
    
    rf_single_best_topN_testing_metrics = pd.concat([rf_single_best_topN_testing_metrics, get_single_label_metrics(testY[:, index], scoreY, label)])


###### RF_multi ######
with open('models/rf_multi_10fold.pickle', 'rb') as f:
    rf_multi_10fold = pickle.load(f)

rf_multi_10fold.groupby('label').mean().round(4).iloc[[0,4,2,3,1,5]] 

with open('models/rf_multi_model.pickle', 'rb') as f:
    rf_multi_model = pickle.load(f)

scoreY = np.array(rf_multi_model.predict_proba(testX))[:,:,1].T
rf_multi_metrics_df = get_multi_label_metrics(testY, scoreY, target_labels)
rf_multi_metrics_df

rf_multi_topN_10fold = pd.read_csv(f'performances/rf_multi_topN_10fold.csv')
_df = rf_multi_10fold.groupby('label').mean().reset_index() 
_auc = _df[_df['label']=='macro']['auc'].values.reshape(-1)[0]

mask = rf_multi_topN_10fold.groupby('label').mean().reset_index()['label'].str.contains('macro')
__df = rf_multi_topN_10fold.groupby('label').mean().reset_index()[mask]
topN = __df[__df['auc']>_auc]['label'].values.reshape(-1)[0].split('_')[-1]
topN

_df = rf_multi_topN_10fold.groupby('label').mean().reset_index()
mask = rf_multi_topN_10fold.groupby('label').mean().reset_index()['label'].str.contains(topN)
rf_multi_best_topN_10fold_metrics = _df[mask].reset_index(drop=True)
rf_multi_best_topN_10fold_metrics.round(4).iloc[[0,4,2,3,1,5]] 


topN = int(rf_multi_best_topN_10fold_metrics.loc[0, 'label'].split('_')[-1])
topN_trainX = pd.DataFrame(trainX, columns=AAC_cols+PAAC_cols+CTDD_cols)[pd.Series(rf_multi_model.feature_importances_, AAC_cols+PAAC_cols+CTDD_cols).nlargest(topN).index].values
topN_testX = pd.DataFrame(testX, columns=AAC_cols+PAAC_cols+CTDD_cols)[pd.Series(rf_multi_model.feature_importances_, AAC_cols+PAAC_cols+CTDD_cols).nlargest(topN).index].values

with open('models/rf_multi_topN_model.pickle', 'rb') as f:
    model = pickle.load(f)
    
scoreY = np.array(model.predict_proba(topN_testX))[:,:,1].T

get_multi_label_metrics(testY, scoreY, target_labels)


###### cnn_single ######
with open('models/cnn_single_10fold.pickle', 'rb') as f:
    cnn_single_10fold = pickle.load(f)

cnn_single_10fold.groupby('label').mean().round(4).reset_index().iloc[[0,4,2,3,1]]


cnn_single_models = []
for index, label in enumerate(target_labels):
    cnn_single_models.append(load_model(f'models/cnn_single_{label}_model.h5'))


for index, label in enumerate(target_labels):
    model = cnn_single_models[index]
    scoreY = model.predict(testX.reshape((testX.shape[0], testX.shape[1], 1)))[:, 0]

    if index==0:
        cnn_single_metrics_df = get_single_label_metrics(testY[:, index], scoreY, label)
    else:
        cnn_single_metrics_df.loc[len(cnn_single_metrics_df)] = get_single_label_metrics(testY[:, index], scoreY, label).values[0]
cnn_single_metrics_df

###### cnn_multi ######
with open('models/cnn_multi_10fold.pickle', 'rb') as f:
    cnn_multi_10fold = pickle.load(f)

cnn_multi_10fold.groupby('label').mean().round(4).reset_index().iloc[[0,4,2,3,1,5]]

cnn_multi_model = load_model('models/cnn_multi_model.h5')
scoreY = cnn_multi_model.predict(testX.reshape((testX.shape[0], testX.shape[1], 1)))
cnn_multi_metrics_df = get_multi_label_metrics(testY, scoreY, target_labels)
cnn_multi_metrics_df

with open('cnn_multi_heatmaps.pickle', 'rb') as f:
    cnn_multi_heatmaps = pickle.load(f)

cnn_multi_topN_10fold = pd.read_csv(f'performances/cnn_multi_topN_10fold.csv')

with open('cnn_multi_10fold.pickle', 'rb') as f:
    cnn_multi_10fold = pickle.load(f)

cnn_multi_10fold.groupby('label').mean().round(4).reset_index().iloc[[0,4,2,3,1,5]]

_df = cnn_multi_10fold.groupby('label').mean().reset_index() 
_auc = _df[_df['label']=='macro']['auc'].values.reshape(-1)[0]

mask = cnn_multi_topN_10fold.groupby('label').mean().reset_index()['label'].str.contains('macro')
__df = cnn_multi_topN_10fold.groupby('label').mean().reset_index()[mask]
topN = __df[__df['auc']>_auc]['label'].values.reshape(-1)[0].split('_')[-1]
topN

_df = cnn_multi_topN_10fold.groupby('label').mean().reset_index()
mask = cnn_multi_topN_10fold.groupby('label').mean().reset_index()['label'].str.contains(topN)
cnn_multi_best_topN_10fold_metrics = _df[mask].reset_index(drop=True)
cnn_multi_best_topN_10fold_metrics.round(4).iloc[[0,4,2,3,1,5]] 
