import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = 'healthcare_dataset.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Download the dataset from Kaggle and place it in the working directory.")

df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
expected_cols = ['Name','Age','Gender','Blood Type','Medical Condition','Date of Admission','Doctor','Hospital','Insurance Provider','Billing Amount','Room Number','Admission Type','Discharge Date','Medication','Test Results']
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("Warning: expected columns missing:", missing)

for col in ['Age','Billing Amount','Room Number']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in ['Date of Admission','Discharge Date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Task 1 — EDA (alternative visualizations)
if 'Age' in df.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df['Age'].dropna())
    plt.title('Age distribution (Boxplot)')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'box_age.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(8,4))
    sns.kdeplot(df['Age'].dropna(), fill=True)
    plt.title('Age distribution (KDE)')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'kde_age.png'))
    plt.show()
    plt.close()

if 'Billing Amount' in df.columns:
    plt.figure(figsize=(8,4))
    sns.violinplot(x=df['Billing Amount'].dropna())
    plt.title('Billing Amount distribution (Violin)')
    plt.xlabel('Billing Amount')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'violin_billing_amount.png'))
    plt.show()
    plt.close()

    billing_nonan = df['Billing Amount'].dropna()
    plt.figure(figsize=(8,4))
    plt.hist(np.log1p(billing_nonan), bins=50)
    plt.title('Billing Amount (log1p histogram)')
    plt.xlabel('log1p(Billing Amount)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'hist_billing_amount_log.png'))
    plt.show()
    plt.close()

if 'Room Number' in df.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df['Room Number'].dropna())
    plt.title('Room Number distribution (Boxplot)')
    plt.xlabel('Room Number')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'box_room_number.png'))
    plt.show()
    plt.close()

for col in ['Medical Condition','Admission Type','Medication']:
    if col in df.columns:
        vc = df[col].fillna('Unknown').value_counts().head(40)
        plt.figure(figsize=(10,6))
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f'Frequency of {col} (top 40)')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,f'freq_{col.replace(" ","_")}_bar.png'))
        plt.show()
        plt.close()

eda_summary = {
    'n_rows': len(df),
    'Age_min': df['Age'].min() if 'Age' in df.columns else None,
    'Age_max': df['Age'].max() if 'Age' in df.columns else None,
    'Age_mean': df['Age'].mean() if 'Age' in df.columns else None,
    'Billing_min': df['Billing Amount'].min() if 'Billing Amount' in df.columns else None,
    'Billing_max': df['Billing Amount'].max() if 'Billing Amount' in df.columns else None,
    'Billing_mean': df['Billing Amount'].mean() if 'Billing Amount' in df.columns else None,
}
pd.Series(eda_summary).to_csv(os.path.join(OUT_DIR,'eda_summary.csv'))

# Task 2 — Supervised Learning (predict Test Results)
df_model = df.copy()

if 'Date of Admission' in df_model.columns and 'Discharge Date' in df_model.columns:
    df_model['stay_days'] = (df_model['Discharge Date'] - df_model['Date of Admission']).dt.days
else:
    df_model['stay_days'] = np.nan


if 'stay_days' in df_model.columns:
    df_model['stay_days'] = df_model['stay_days'].fillna(df_model['stay_days'].median())

if 'Medication' in df_model.columns:
    df_model['med_count'] = df_model['Medication'].astype(str).apply(lambda x: 0 if x.strip()=='nan' or x.strip()=='' else len([m for m in x.split(',') if m.strip()!='']))
else:
    df_model['med_count'] = 0

if 'Billing Amount' in df_model.columns:
    try:
        df_model['billing_bucket'] = pd.qcut(df_model['Billing Amount'].fillna(df_model['Billing Amount'].median()), q=4, duplicates='drop')
        df_model['billing_bucket'] = df_model['billing_bucket'].astype(str)
    except Exception:
        df_model['billing_bucket'] = 'unknown'
else:
    df_model['billing_bucket'] = 'unknown'

def group_top_k(series, k=30):
    series = series.fillna('Unknown').astype(str)
    top = series.value_counts().nlargest(k).index
    return series.apply(lambda x: x if x in top else 'Other')

for col in ['Doctor','Hospital']:
    if col in df_model.columns:
        df_model[col] = group_top_k(df_model[col], k=50)

features = [f for f in ['Age','Gender','Blood Type','Medical Condition','Doctor','Hospital','Insurance Provider','Billing Amount','Room Number','Admission Type','Medication','stay_days','med_count','billing_bucket'] if f in df_model.columns]

X = df_model[features].copy()
y = df_model['Test Results'].astype(str).fillna('Unknown')

numeric_cols = ['Age','Billing Amount','Room Number','stay_days','med_count']
for col in numeric_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include=['category']).columns:
    X[col] = X[col].astype(str)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Unknown').astype(str)

cat_cols = [c for c in X.columns if X[c].dtype == 'object']
print('Categorical columns detected for model:', cat_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

use_catboost = True
try:
    from catboost import CatBoostClassifier
except Exception as e:
    print('CatBoost not available, falling back to RandomForest. To use CatBoost install it with: pip install catboost')
    use_catboost = False

if use_catboost:
    catboost_params = {
        'iterations': 800,
        'depth': 8,
        'learning_rate': 0.05,
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': 100,
    }
    model_cb = CatBoostClassifier(**catboost_params)
    model_cb.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test), use_best_model=True)
    y_pred = model_cb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"CatBoost Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    try:
        model_cb.save_model(os.path.join(OUT_DIR,'catboost_model.cbm'))
    except Exception:
        pass
    try:
        fi = model_cb.get_feature_importance(prettified=True)
        pd.DataFrame(fi).to_csv(os.path.join(OUT_DIR,'catboost_feature_importance.csv'), index=False)
    except Exception:
        pass
    test_results = X_test.copy()
    test_results['actual'] = y_test.values
    try:
        y_pred_flat = np.asarray(y_pred).reshape(-1)
    except Exception:
        y_pred_flat = list(y_pred)
    if len(y_pred_flat) != len(test_results):
        y_pred_flat = np.asarray(y_pred).squeeze()
    test_results['predicted'] = pd.Series(y_pred_flat, index=test_results.index).astype(str)
    test_results.to_csv(os.path.join(OUT_DIR,'test_predictions_catboost.csv'))

    plt.figure(figsize=(8,6))
    pd.crosstab(test_results['actual'], test_results['predicted']).plot(kind='bar')
    plt.title('Actual vs Predicted (CatBoost)')
    plt.xlabel('Actual')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'actual_vs_predicted_catboost.png'))
    plt.show()
    plt.close()

else:
    print('Using RandomForest fallback pipeline with OneHot encoding...')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_cat = ohe.fit_transform(X_train[cat_cols]) if len(cat_cols) > 0 else np.zeros((len(X_train),0))
    X_test_cat = ohe.transform(X_test[cat_cols]) if len(cat_cols) > 0 else np.zeros((len(X_test),0))

    numeric_cols_present = [c for c in numeric_cols if c in X.columns]
    X_train_num = X_train[numeric_cols_present].values if len(numeric_cols_present)>0 else np.zeros((len(X_train),0))
    X_test_num = X_test[numeric_cols_present].values if len(numeric_cols_present)>0 else np.zeros((len(X_test),0))

    X_train_enc = np.hstack([X_train_num, X_train_cat])
    X_test_enc = np.hstack([X_test_num, X_test_cat])

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_enc, y_train)
    y_pred = rf.predict(X_test_enc)
    acc = accuracy_score(y_test, y_pred)
    print(f"RandomForest Test accuracy (fallback): {acc:.4f}")
    print(classification_report(y_test, y_pred))

    test_results = X_test.copy()
    test_results['actual'] = y_test.values
    test_results['predicted'] = y_pred
    test_results.to_csv(os.path.join(OUT_DIR,'test_predictions_randomforest.csv'))

    try:
        import joblib
        joblib.dump(ohe, os.path.join(OUT_DIR,'onehot_encoder.joblib'))
        joblib.dump(rf, os.path.join(OUT_DIR,'randomforest_model.joblib'))
    except Exception:
        pass

# Task 3 — Anomaly Detection in Billing Amounts
if 'Billing Amount' in df.columns:
    billing = df[['Billing Amount']].copy()
    billing_mean = billing['Billing Amount'].mean()
    billing_std = billing['Billing Amount'].std()
    billing['z_score'] = (billing['Billing Amount'] - billing_mean) / billing_std
    z_thresh = 3.0
    billing['anomaly_z'] = billing['z_score'].abs() > z_thresh

    iso_features = ['Billing Amount']
    if 'Age' in df.columns:
        iso_features.append('Age')
    iso_input = df[iso_features].fillna(df[iso_features].median())
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(iso_input)
    iso_preds = iso.predict(iso_input)
    billing['anomaly_iso'] = iso_preds == -1

    billing['anomaly_combined'] = billing['anomaly_z'] | billing['anomaly_iso']

    df_out = df.copy()
    df_out['billing_z_score'] = billing['z_score'].values
    df_out['anomaly_billing'] = billing['anomaly_combined'].values
    df_out.to_csv(os.path.join(OUT_DIR,'anomalies_marked.csv'), index=False)

    anoms = df_out[df_out['anomaly_billing']].sort_values('Billing Amount', ascending=False).head(50)
    if len(anoms) > 0:
        anoms[['Name','Age','Medical Condition','Billing Amount','Medication','anomaly_billing']].to_csv(os.path.join(OUT_DIR,'top_billing_anomalies.csv'), index=False)

    with open(os.path.join(OUT_DIR,'anomaly_interpretation.txt'),'w') as f:
        f.write('Anomaly detection on Billing Amounts')
        f.write(f'Total records: {len(df)}')
        f.write(f'Total anomalies flagged: {int(df_out["anomaly_billing"].sum())}')
        f.write('Interpretation examples:\n')
        f.write(' - Very high billing amounts often correspond to rare/expensive procedures, long ICU stays, or bundled charges.')
        f.write(' - Very low billing amounts may indicate short visits, billing errors, or fully-covered cases by insurance.')

# Task 4 — AI Task (LLM-Based) — Doctor Recommendation Generator
def generate_doctor_recommendation(predicted_result: str, age: int, medical_condition: str, medication: str) -> str:
    """Return a short doctor-style recommendation based on predicted test result and key attributes."""
    rec = []
    rec.append(f"Patient age: {age}. Primary condition: {medical_condition}. Current medication: {medication}.")

    pr = str(predicted_result).strip().lower()
    if 'normal' in pr:
        rec.append('Predicted test result: Normal. Continue current management and routine follow-up.')
        rec.append('Advice: Maintain medication adherence, lifestyle modifications, and return earlier if new concerning symptoms arise.')
    elif 'abnormal' in pr or 'positive' in pr:
        rec.append('Predicted test result: Abnormal. Recommend prompt clinical review and confirmatory testing.')
        rec.append('Advice: Consider escalation of care, medication review, and referral to a relevant specialist. Close follow-up within 48-72 hours is advised.')
    elif 'inconclusive' in pr or 'unknown' in pr:
        rec.append('Predicted test result: Inconclusive. Recommend repeat or additional diagnostic tests to clarify.')
        rec.append('Advice: Observe symptoms and schedule targeted investigations as needed.')
    else:
        rec.append(f'Predicted test result: {predicted_result}. Consider clinical correlation and further testing.')

    rec.append('General: Ensure medication reconciliation, vaccination status, and social support assessment where applicable.')
    return ' '.join(rec)

sample_recommendation = 'No sample available.'
try:
    if 'test_results' in locals():
        tr = test_results
    else:
        tr = locals().get('test_results', None)
    if tr is not None and len(tr) > 0:
        row = tr.reset_index().iloc[0]
        sample_recommendation = generate_doctor_recommendation(row['predicted'], int(row.get('Age',45) if not pd.isna(row.get('Age',45)) else 45), row.get('Medical Condition','Unknown'), row.get('Medication','None'))
    else:
        sample_recommendation = generate_doctor_recommendation('Abnormal', 67, 'Pneumonia', 'Amoxicillin')
except Exception:
    sample_recommendation = generate_doctor_recommendation('Abnormal', 67, 'Pneumonia', 'Amoxicillin')

with open(os.path.join(OUT_DIR,'sample_doctor_recommendation.txt'),'w') as f:
    f.write(sample_recommendation)

print('Sample AI Doctor Recommendation:')
print(sample_recommendation)