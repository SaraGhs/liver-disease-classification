"""Educational ILPD comparison. See README and provenance; not for clinical use.
Starter workflow prepared with AI assistance for Sara's guided portfolio refresh.
"""

# 2. Import tools and record versions
from pathlib import Path
import io, json, hashlib, zipfile, platform, sys
from urllib.request import urlopen
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report)
import joblib

SEED = 42
ROOT = Path('liver_disease_project')
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
VERSIONS = {p: version(p) for p in
    ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib']}
print('Python:', platform.python_version())
print(json.dumps(VERSIONS, indent=2))
print('CHECKPOINT 1: environment ready')

# 3. Download and inspect the original UCI dataset
SOURCE_PAGE = 'https://archive.ics.uci.edu/dataset/225/ilpd'
ZIP_URL = ('https://archive.ics.uci.edu/static/public/225/'
           'ilpd+indian+liver+patient+dataset.zip')
archive_path = DATA / 'uci_ilpd.zip'
if not archive_path.exists():
    try:
        archive_path.write_bytes(urlopen(ZIP_URL, timeout=60).read())
    except Exception as exc:
        raise RuntimeError('Download failed. Stop and share this error; do not substitute another CSV silently.') from exc
raw = archive_path.read_bytes()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    csv_files = [n for n in z.namelist() if n.lower().endswith('.csv')]
    if len(csv_files) != 1:
        raise ValueError(f'Unexpected dataset archive contents: {csv_files}')
    csv_bytes = z.read(csv_files[0])

COLS = ['age','gender','total_bilirubin','direct_bilirubin',
        'alkaline_phosphotase','alanine_aminotransferase',
        'aspartate_aminotransferase','total_proteins','albumin',
        'albumin_globulin_ratio','selector']
df_raw = pd.read_csv(io.BytesIO(csv_bytes), header=None, names=COLS)
if df_raw.shape[1] != 11 or not (400 < len(df_raw) < 1000):
    raise ValueError(f'Unexpected shape: {df_raw.shape}. Stop and review the source.')
if not set(df_raw['selector'].dropna().unique()).issubset({1,2}):
    raise ValueError('Unexpected target encoding. Do not guess class meanings.')
df_raw.to_csv(DATA / 'ilpd_original.csv', index=False)
provenance = {
    'source_page': SOURCE_PAGE, 'download_url': ZIP_URL,
    'doi': '10.24432/C5D02C', 'license': 'CC BY 4.0',
    'archive_sha256': hashlib.sha256(raw).hexdigest(),
    'csv_sha256': hashlib.sha256(csv_bytes).hexdigest(),
    'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
    'raw_rows': int(len(df_raw)), 'raw_columns': int(df_raw.shape[1]),
    'target_mapping': {'1':'liver disease','2':'no liver disease'},
    'citation': 'Ramana, B. & Venkateswarlu, N. ILPD (Indian Liver Patient Dataset). UCI Machine Learning Repository.'
}
(RESULTS / 'data_provenance.json').write_text(json.dumps(provenance, indent=2))
print('Rows, columns:', df_raw.shape)
print('Original target counts:', df_raw['selector'].value_counts().to_dict())
print('Missing values:', df_raw.isna().sum().to_dict())
print('Exact duplicate rows:', int(df_raw.duplicated().sum()))
print('CHECKPOINT 2: data downloaded and inspected')

# 4. Deduplicate and reserve the test partition
df = df_raw.drop_duplicates().copy().reset_index(drop=True)
assert df['selector'].notna().all()
X = df.drop(columns='selector').copy()
y = df['selector'].map({1:1, 2:0}).astype(int)
assert 'selector' not in X.columns and X.shape[1] == 10
X['gender'] = X['gender'].astype(str).str.strip()
assert set(X['gender'].unique()).issubset({'Male','Female'})
NUMERIC = [c for c in X.columns if c != 'gender']
for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors='raise')
assert not np.isinf(X[NUMERIC].to_numpy()).any()
# Group identical feature rows so they cannot occur in both partitions.
# These groups are a duplicate safeguard, not verified patient identifiers.
groups = pd.util.hash_pandas_object(X, index=False).astype(str)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
assert set(g_train).isdisjoint(set(g_test))
assert y_train.nunique() == y_test.nunique() == 2
assert set(train_idx).isdisjoint(set(test_idx))
split_info = {
    'raw_rows': len(df_raw), 'deduplicated_rows': len(df),
    'exact_duplicate_rows_removed': len(df_raw)-len(df),
    'training_rows': len(X_train), 'test_rows': len(X_test),
    'training_target_counts': {str(k):int(v) for k,v in y_train.value_counts().items()},
    'test_target_counts': {str(k):int(v) for k,v in y_test.value_counts().items()},
    'feature_count': X.shape[1], 'seed': SEED,
    'split': 'first fold of shuffled 5-fold StratifiedGroupKFold',
    'grouping': 'exact feature-row hashes; no verified patient IDs',
    'positive_class': 'liver disease (original Selector=1)',
    'duplicate_feature_groups_across_partitions': 0
}
(RESULTS/'split_summary.json').write_text(json.dumps(split_info, indent=2))
pd.DataFrame({'clean_row':np.arange(len(df)), 'group':groups,
    'partition':np.where(np.isin(np.arange(len(df)),test_idx),'test','train')
}).to_csv(RESULTS/'split_assignments.csv', index=False)
print(json.dumps(split_info, indent=2))
print('CHECKPOINT 3: split checks passed; test data reserved')

# 5. Compare models using training cross-validation
def make_pipeline(model):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('numeric',numeric,NUMERIC),
                              ('categorical',categorical,['gender'])])
    return Pipeline([('preprocess',prep),('model',model)])

inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
cv_splits = list(inner.split(X_train, y_train, g_train))
for a,b in cv_splits:
    assert set(g_train.iloc[a]).isdisjoint(set(g_train.iloc[b]))
    assert y_train.iloc[a].nunique() == y_train.iloc[b].nunique() == 2

experiments = {
    'Majority baseline': (DummyClassifier(strategy='most_frequent'), {}),
    'Logistic regression': (LogisticRegression(max_iter=5000, random_state=SEED),
        {'model__C':[0.1,1,10], 'model__class_weight':[None,'balanced']}),
    'SVM': (SVC(kernel='rbf'),
        {'model__C':[0.1,1,10], 'model__gamma':['scale',0.01,0.1],
         'model__class_weight':[None,'balanced']})
}
searches = {}; cv_rows = []
for name,(model,grid) in experiments.items():
    search = GridSearchCV(make_pipeline(model), grid, scoring='balanced_accuracy',
        cv=cv_splits, n_jobs=1, refit=True, error_score='raise')
    search.fit(X_train, y_train)
    searches[name] = search
    cv_rows.append({'model':name, 'cv_balanced_accuracy':float(search.best_score_),
        'cv_std':float(search.cv_results_['std_test_score'][search.best_index_]),
        'settings':json.dumps(search.best_params_)})
    pd.DataFrame(search.cv_results_).to_csv(
        RESULTS/('cv_'+name.lower().replace(' ','_')+'.csv'),index=False)
cv_table = pd.DataFrame(cv_rows).sort_values('cv_balanced_accuracy',ascending=False)
selected_name = str(cv_table.iloc[0]['model'])
cv_table.to_csv(RESULTS/'cross_validation_results.csv', index=False)
print(cv_table.to_string(index=False))
print('Selected using training CV only:', selected_name)
print('CHECKPOINT 4: model selected before test evaluation')

# 6. Evaluate on the held-out test partition
test_rows = []; preds_by_model = {}; scores_by_model = {}
for name,search in searches.items():
    model = search.best_estimator_
    pred = model.predict(X_test)
    if hasattr(model,'decision_function'):
        score = model.decision_function(X_test)
    else:
        score = model.predict_proba(X_test)[:,list(model.classes_).index(1)]
    preds_by_model[name] = pred; scores_by_model[name] = score
    test_rows.append({'model':name, 'selected_by_training_cv':name==selected_name,
        'accuracy':accuracy_score(y_test,pred),
        'balanced_accuracy':balanced_accuracy_score(y_test,pred),
        'precision_disease':precision_score(y_test,pred,zero_division=0),
        'recall_disease':recall_score(y_test,pred,zero_division=0),
        'f1_disease':f1_score(y_test,pred,zero_division=0),
        'roc_auc':roc_auc_score(y_test,score),
        'average_precision':average_precision_score(y_test,score)})
test_table = pd.DataFrame(test_rows)
test_table.to_csv(RESULTS/'test_metrics.csv',index=False)
fig,axes = plt.subplots(1,3,figsize=(14,4))
for ax,(name,pred) in zip(axes,preds_by_model.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test,pred,labels=[0,1]),
        display_labels=['No disease','Disease']).plot(ax=ax,colorbar=False)
    ax.set_title(name)
fig.tight_layout(); fig.savefig(RESULTS/'confusion_matrices.png',dpi=160)
plt.show()
pd.DataFrame({'clean_row':test_idx, 'observed_disease':y_test.to_numpy(),
    'predicted_disease':preds_by_model[selected_name],
    'decision_score':scores_by_model[selected_name]}
).to_csv(RESULTS/'held_out_predictions.csv',index=False)
report = classification_report(y_test,preds_by_model[selected_name],
    labels=[0,1],target_names=['No disease','Disease'],zero_division=0)
(RESULTS/'classification_report.txt').write_text(report)
print(test_table.round(4).to_string(index=False))
print('Frozen selection:',selected_name)
print(report)
print('CHECKPOINT 5: held-out evaluation complete')

# 7. Save the model and reproducible reporting package
final_model = searches[selected_name].best_estimator_
joblib.dump(final_model, RESULTS/'selected_model.joblib')
reloaded = joblib.load(RESULTS/'selected_model.joblib')
assert np.array_equal(reloaded.predict(X_test),preds_by_model[selected_name])
(ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in VERSIONS.items())+'\n')
(RESULTS/'environment.json').write_text(json.dumps(
    {'python':platform.python_version(),'packages':VERSIONS,'seed':SEED},indent=2))
selected = test_table.loc[test_table['model']==selected_name].iloc[0]
metrics_md = test_table.round(4).to_csv(index=False)
readme = f'''# Liver Disease Classification

Reproducible educational reimplementation of an earlier ML for Bioinformatics
course project, comparing a majority baseline, logistic regression and RBF SVM.
The current workflow and guide were prepared with AI assistance; Sara's final
contribution and interpretation are documented in PERSONAL_CONTRIBUTION.md.
Do not describe this as independent clinical research or clinical deployment.

## Question
How do these predeclared classifiers compare on held-out ILPD records under
training-only preprocessing and cross-validation?

## Data and credit
UCI ILPD, DOI: 10.24432/C5D02C. Ramana, B. & Venkateswarlu, N.
Source: {SOURCE_PAGE}
Dataset licence: CC BY 4.0; dataset citation does not license third-party code.
The original course notebook is not redistributed in this package.
Downloaded {len(df_raw)} rows; retained {len(df)} after exact-row deduplication.
Target: original Selector=1 means liver disease and Selector=2 means no disease.
Data hash and retrieval details: results/data_provenance.json.

## Evaluation
Feature-identical rows are grouped across an approximately 80/20 split.
No patient identifiers are available; this does not guarantee patient-level independence.
Three-fold grouped stratified CV on training records tunes preprocessing/model
pipelines and selects by mean balanced accuracy. Missing values and scaling are
learned only within training folds. The target is excluded from all features.
No decision threshold is tuned against the test set.
Training rows: {len(X_train)}; held-out rows: {len(X_test)}.
Selected before test evaluation: **{selected_name}**.
CV standard deviations are fold dispersion, not confidence intervals.

## Held-out results
```csv
{metrics_md}```

![Confusion matrices](results/confusion_matrices.png)

## Limitations
Small historical single-region dataset; class imbalance; no external validation,
no verified patient grouping, no confidence intervals or clinical utility study.
Results describe one fixed split; no diagnostic or deployment claim is made.
Any previous exposure to this dataset limits how novel this holdout can be considered.

## Rerun
Python 3.12 was used for the starter validation; the actual runtime version
for your run is recorded in results/environment.json.
Install requirements.txt, then run liver_disease_classification.py in an empty
working folder or open the included notebook in Colab and run its cells in order.
Internet is required for the public UCI download. No GPU is needed.
Download outputs before a temporary Colab runtime ends. Only load joblib files
from trusted sources. The saved model is optional to publish.
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
contribution = '''# Personal contribution and interpretation

This is a new portfolio refresh of earlier coursework. The starter code and
guide were prepared with AI assistance; project execution and interpretation
must be documented below before a final public release.

Sara: replace the questions below with your own short factual answers.
1. What did you implement in the original coursework?
2. Which parts of this refresh did you inspect, modify and execute?
3. Explain why the target must not appear among features.
4. Explain why imputation and scaling belong inside training folds.
5. Which model was selected by CV, and what do its test results mean?
6. What error pattern and limitation matter most?
7. What would you improve next without tuning against this test set?

Keep assistance and source credits accurate. Do not claim sole authorship of
provided code. Remove these unanswered prompts only after writing your answers.
'''
contribution_path = ROOT/'PERSONAL_CONTRIBUTION.md'
if not contribution_path.exists():
    contribution_path.write_text(contribution,encoding='utf-8')
(ROOT/'.gitignore').write_text('data/\n__pycache__/\n.ipynb_checkpoints/\n*.zip\n')
checks = {'target_excluded': 'selector' not in X.columns,
    'no_feature_group_overlap':set(g_train).isdisjoint(set(g_test)),
    'model_reload_predictions_identical':True,
    'selected_by_training_cv':selected_name,
    'training_rows':len(X_train),'test_rows':len(X_test),
    'evaluation_run_utc':datetime.now(timezone.utc).isoformat()}
(RESULTS/'checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
print('CHECKPOINT 6: outputs saved; personal interpretation still required')
