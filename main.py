import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

print("--- PHASE 1: Data Loading & Cleaning ---")

# 1. Load Data
file_names = [
    "FY10.csv", "FY11.csv",
    "FY12.csv", "FY13.csv",
    "FY14.csv", "FY15.csv",
    "FY16.csv", "FY17.csv",
    "FY18.csv", "FY19.csv"
]

# Standard column names expected
standard_columns = ['OPEID', 'SCHOOL_NAME', 'CITY', 'STATE', 'SCHOOL_TYPE', 'REFERRAL_REASON', 'FINE_AMOUNT', 'FINE_DATE']

dfs = []
for f in file_names:
    try:
        # Check if file exists in current directory, otherwise try finding it or skip
        simple_name = f.strip()
        if os.path.exists(simple_name):
            f = simple_name
        elif not os.path.exists(f):
            print(f"Skipping {f}: File not found.")
            continue

        if 'FY14' in f or 'FY 14' in f:
            # FY14 often has proper headers
            temp_df = pd.read_csv(f)
            temp_df.columns = [c.strip().replace('\n', ' ').upper() for c in temp_df.columns]
            col_map = {
                'SCH_NAME': 'SCHOOL_NAME',
                'CTY_NAME': 'CITY',
                'ST_CD': 'STATE',
                'SCH_TYPE': 'SCHOOL_TYPE',
                'REASON_REFER_DESC': 'REFERRAL_REASON',
                'IMPOSED_FINE_AMT': 'FINE_AMOUNT',
                'OUTCOME_DT': 'FINE_DATE'
            }
            temp_df = temp_df.rename(columns=col_map)
        else:
            # Other files often need skipping rows
            temp_df = pd.read_csv(f)
            if not any('OPE' in str(c).upper() for c in temp_df.columns):
                 temp_df = pd.read_csv(f, skiprows=2, header=None)
                 # Assign standard column names
                 num_cols = len(temp_df.columns)
                 # Ensure we don't assign more names than columns
                 current_cols = standard_columns[:num_cols]
                 temp_df.columns = current_cols + [f'EXTRA_{i}' for i in range(len(temp_df.columns) - len(current_cols))]

        # Keep only the standard columns we need
        for col in standard_columns:
            if col not in temp_df.columns:
                temp_df[col] = np.nan
        
        temp_df = temp_df[standard_columns]
        
        # Extract fiscal year from filename
        # Use regex to find "FY" followed by digits
        year_match = re.search(r'FY\s*(\d+)', f, re.IGNORECASE)
        if year_match:
            year_val = int(year_match.group(1))
            if year_val < 100: year_val += 2000
            temp_df['FISCAL_YEAR'] = year_val
        
        print(f"Loaded {f}: {len(temp_df)} rows")
        dfs.append(temp_df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not dfs:
    print("CRITICAL: No data loaded. Check file names.")
    exit()

df = pd.concat(dfs, ignore_index=True)

# 2. Date Conversion
def excel_date_to_datetime(serial):
    try:
        # Handle cases where it's already a string date or NaN
        if pd.isna(serial): return pd.NaT
        return datetime(1899, 12, 30) + timedelta(days=float(serial))
    except:
        return pd.NaT

df['FINE_DATE_OBJ'] = pd.to_datetime(df['FINE_DATE'].apply(excel_date_to_datetime), errors='coerce')

# 3. Clean Fine Amount
def clean_fine_amount(val):
    if pd.isna(val): return 0
    val_str = str(val).strip().replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(val_str)
    except:
        return 0

df['FINE_AMOUNT'] = df['FINE_AMOUNT'].apply(clean_fine_amount)
df['LOG_FINE'] = np.log1p(df['FINE_AMOUNT'])

# 4. Clean School Type
type_map = {'Prop': 'Proprietary', 'Priv': 'Private', 'Pub': 'Public', 'Foreign': 'Foreign'}
df['SCHOOL_TYPE'] = df['SCHOOL_TYPE'].map(type_map).fillna('Other')

# 5. Clean Referral Reason
def clean_reason(text):
    if pd.isna(text): return "Other"
    text = str(text).lower()
    if 'clery' in text or 'campus security' in text or 'part 86' in text: return 'Clery/Safety'
    if 'ipeds' in text: return 'IPEDS'
    if 'drug' in text: return 'Drug Prevention'
    if 'qui tam' in text: return 'Qui Tam (Fraud)'
    return 'Other'

df['REASON_GROUP'] = df['REFERRAL_REASON'].apply(clean_reason)
print(f"Total Records Cleaned: {len(df)}")

print("\n--- PHASE 2: Time Series & Stats ---")

yearly_stats = df.groupby('FISCAL_YEAR')['FINE_AMOUNT'].agg(['sum', 'count']).reset_index()

# 1. Visualization: Trend Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_stats, x='FISCAL_YEAR', y='sum', marker='o', label='Total Fines ($)')
plt.title('Total Fines Imposed by Year (Enforcement Trend)')
plt.ylabel('Total Fine Amount ($)')
plt.xlabel('Fiscal Year')
plt.grid(True)
plt.savefig('trend_analysis.png')
plt.close()
print("Saved plot: trend_analysis.png")

# 2. Visualization: Composition by Reason
plt.figure(figsize=(12, 6))
reason_time = df.groupby(['FISCAL_YEAR', 'REASON_GROUP'])['FINE_AMOUNT'].sum().unstack()
reason_time.plot(kind='bar', stacked=True, figsize=(12,6))
plt.title('Composition of Fines by Reason Over Time')
plt.ylabel('Total Amount ($)')
plt.tight_layout()
plt.savefig('reason_analysis.png')
plt.close()
print("Saved plot: reason_analysis.png")

# Trend Test
slope, intercept, r_val, p_val, std_err = stats.linregress(yearly_stats['FISCAL_YEAR'], yearly_stats['sum'])
print(f"Time Trend P-Value: {p_val:.4f} (Significant if < 0.05)")

# School Type Test
df_nonzero = df[df['FINE_AMOUNT'] > 0].copy()
groups = [df_nonzero[df_nonzero['SCHOOL_TYPE'] == t]['FINE_AMOUNT'].values 
          for t in df_nonzero['SCHOOL_TYPE'].unique() if len(df_nonzero[df_nonzero['SCHOOL_TYPE'] == t]) > 3]
try:
    s, p = stats.kruskal(*groups)
    print(f"School Type Difference P-Value: {p:.4e} (Significant if < 0.05)")
except:
    print("Could not run Kruskal-Wallis.")

print("\n--- PHASE 4: Predictive Modeling ---")

# We strictly need non-zero fines for these models
model_df = df[(df['FINE_AMOUNT'] > 0) & (df['FISCAL_YEAR'].notna())].copy()

X = model_df[['SCHOOL_TYPE', 'REASON_GROUP', 'FISCAL_YEAR']]
# Target 1: Log Fine (For Linear Reg & SVM)
y_log = model_df['LOG_FINE']
# Target 2: Raw Fine (For GLM)
y_raw = model_df['FINE_AMOUNT']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['FISCAL_YEAR']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['SCHOOL_TYPE', 'REASON_GROUP'])
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# 1. Linear Regression (Predicts Log)
lr_pipe = Pipeline([('prep', preprocessor), ('reg', LinearRegression())])
lr_scores = cross_val_score(lr_pipe, X, y_log, cv=kf, scoring='r2')
lr_rmse = -cross_val_score(lr_pipe, X, y_log, cv=kf, scoring='neg_root_mean_squared_error')

# 2. SVM (Predicts Log)
svm_pipe = Pipeline([('prep', preprocessor), ('reg', SVR(kernel='rbf', C=1.0, epsilon=0.1))])
svm_scores = cross_val_score(svm_pipe, X, y_log, cv=kf, scoring='r2')
svm_rmse = -cross_val_score(svm_pipe, X, y_log, cv=kf, scoring='neg_root_mean_squared_error')

# 3. GLM (Predicts Raw using Log-Link)
glm_pipe = Pipeline([
    ('prep', preprocessor),
    ('reg', TweedieRegressor(power=2, link='log', max_iter=2000, solver='newton-cholesky', alpha=1.0))
])

X_train, X_test, y_raw_train, y_raw_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
glm_pipe.fit(X_train, y_raw_train)
y_pred_glm_raw = glm_pipe.predict(X_test)

# Calculate R2 on Log Scale for fair comparison
r2_glm_log_scale = r2_score(np.log1p(y_raw_test), np.log1p(y_pred_glm_raw))
# Calculate RMSE on Log Scale
rmse_glm_log_scale = np.sqrt(mean_squared_error(np.log1p(y_raw_test), np.log1p(y_pred_glm_raw)))

print(f"Linear Regression | CV R2: {np.mean(lr_scores):.4f} | CV RMSE: {np.mean(lr_rmse):.4f}")
print(f"SVM Regressor     | CV R2: {np.mean(svm_scores):.4f} | CV RMSE: {np.mean(svm_rmse):.4f}")
print(f"GLM Gamma         | Log R2: {r2_glm_log_scale:.4f} | Log RMSE: {rmse_glm_log_scale:.4f}")

# Plotting the GLM vs Linear
plt.figure(figsize=(10,6))
X_train_log, X_test_log, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
lr_pipe.fit(X_train_log, y_log_train)
y_pred_lr = lr_pipe.predict(X_test_log)

plt.scatter(y_log_test, y_pred_lr, alpha=0.5, label='Linear Reg')
plt.scatter(np.log1p(y_raw_test), np.log1p(y_pred_glm_raw), alpha=0.5, color='orange', label='GLM')
plt.plot([0, 20], [0, 20], 'k--', label='Perfect Prediction')
plt.xlabel('Actual Log Fine')
plt.ylabel('Predicted Log Fine')
plt.legend()
plt.title('Model Comparison: Linear vs GLM')
plt.grid(True)
plt.savefig('model_comparison.png')
print("Model comparison plot saved.")

print("\n--- PHASE 5: Model Diagnostics ---")

# Fit model on whole dataset for diagnostics
lr_pipe.fit(X, y_log)
y_pred_full = lr_pipe.predict(X)
residuals = y_log - y_pred_full

# 1. Residuals vs Fitted Plot (Check Homoscedasticity)
plt.figure(figsize=(10,6))
plt.scatter(y_pred_full, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values (Log Fine)')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.grid(True)
plt.savefig('residuals_vs_fitted.png')
print("Saved plot: residuals_vs_fitted.png")

# 2. Q-Q Plot (Check Normality)
plt.figure(figsize=(10,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot of Residuals')
plt.grid(True)
plt.savefig('qq_plot.png')
print("Saved plot: qq_plot.png")

# --- PHASE 6: Classification Analysis (New Content) ---
print("\n--- PHASE 6: Logistic Regression Classification ---")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define "High Fine" as anything above the Median
# Use only non-zero fines for classification
class_df = df[(df['FINE_AMOUNT'] > 0) & (df['FISCAL_YEAR'].notna())].copy()
median_fine = class_df['FINE_AMOUNT'].median()
class_df['IS_HIGH_FINE'] = (class_df['FINE_AMOUNT'] > median_fine).astype(int)

X_class = class_df[['SCHOOL_TYPE', 'REASON_GROUP', 'FISCAL_YEAR']]
y_class = class_df['IS_HIGH_FINE']

# Logistic Regression Pipeline
log_pipe = Pipeline([
    ('prep', preprocessor), 
    ('clf', LogisticRegression(max_iter=1000))
])

# Cross Validation for Accuracy
cv_acc = cross_val_score(log_pipe, X_class, y_class, cv=kf, scoring='accuracy')
print(f"Logistic Regression CV Accuracy: {np.mean(cv_acc):.4f}")

# Train for classification report
X_train_cls, X_test_cls, y_cls_train, y_cls_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)
log_pipe.fit(X_train_cls, y_cls_train)
y_cls_pred = log_pipe.predict(X_test_cls)

print("\nClassification Report:")
print(classification_report(y_cls_test, y_cls_pred, target_names=['Low Fine', 'High Fine']))

# Confusion Matrix
cm = confusion_matrix(y_cls_test, y_cls_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Fine', 'High Fine'],
            yticklabels=['Low Fine', 'High Fine'])
plt.title('Confusion Matrix: High vs Low Fine Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("Saved plot: confusion_matrix.png")

print("\n--- DONE ---")