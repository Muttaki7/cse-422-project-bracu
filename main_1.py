print("===================== IMPORTS =====================")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

print("==================== OUTPUT DIRECTORY ====================")
output_dir = "report_figures"
os.makedirs(output_dir, exist_ok=True)

print("===================== LOAD DATA =====================")
data = pd.read_csv(r"C:\Users\Muttaki\Desktop\cse422\E-commerce Shipping Dataset - E-commerce Shipping Dataset.csv", header=None)  # No header

print("Raw columns (as numbers):", data.columns.tolist())

# Assign proper column names (standard for this dataset)
column_names = [
    "ID",
    "Warehouse_block",
    "Mode_of_Shipment",
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Product_importance",
    "Gender",
    "Discount_offered",
    "Weight_in_gms",
    "Reached_on_time_Y_N"
]

data.columns = column_names

# Drop the first row if it's the old header
data = data.iloc[1:].reset_index(drop=True)

print("Cleaned Columns:", data.columns.tolist())

print("===================== TARGET & FEATURES =====================")
target = "Reached_on_time_Y_N"
feature_cols = [c for c in data.columns if c not in ["ID", target]]
print(f"Target: {target}")
print(f"Number of Features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

print("==================== FEATURE TYPES ====================")
# Convert numerical columns to numeric
numerical_cols_expected = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product", "Prior_purchases", "Discount_offered", "Weight_in_gms"]
for col in numerical_cols_expected:
    data[col] = pd.to_numeric(data[col], errors='coerce')

categorical_cols = data[feature_cols].select_dtypes(include="object").columns.tolist()
numerical_cols = [c for c in feature_cols if c not in categorical_cols]
print("\nCategorical Features:", categorical_cols)
print("Numerical Features:", numerical_cols)

print("===================== ENCODING (ONCE ONLY) =====================")
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    print(f"Encoded {col}")

print("===================== SCALING =====================")
feature_cols = [c for c in data.columns if c not in ["ID", target]]
X = data[feature_cols].apply(pd.to_numeric, errors='coerce')  # Ensure all numeric
y = pd.to_numeric(data[target], errors='coerce')

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("===================== K-MEANS CLUSTERING =====================")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data["kmeans_cluster"] = clusters

print("\nCrosstab: Actual Label vs K-Means Cluster")
print(pd.crosstab(y, data["kmeans_cluster"], rownames=["Actual"], colnames=["Cluster"]))

print("===================== PCA VISUALIZATION =====================")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7, s=60)
plt.title("K-Means Clustering (PCA Projection)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_pca.png"), dpi=300)
plt.close()
print("   -> Saved: kmeans_pca.png")

print("===================== TRAIN-TEST SPLIT =====================")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

print("===================== MODEL TRAINING =====================")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=800, random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_proba = mlp.predict_proba(X_test)[:, 1]

results = {
    "Logistic Regression": {
        'pred': lr_pred, 'proba': lr_proba,
        'acc': accuracy_score(y_test, lr_pred),
        'prec': precision_score(y_test, lr_pred, average='weighted'),
        'rec': recall_score(y_test, lr_pred, average='weighted'),
        'auc': roc_auc_score(y_test, lr_proba)
    },
    "Decision Tree": {
        'pred': dt_pred, 'proba': dt_proba,
        'acc': accuracy_score(y_test, dt_pred),
        'prec': precision_score(y_test, dt_pred, average='weighted'),
        'rec': recall_score(y_test, dt_pred, average='weighted'),
        'auc': roc_auc_score(y_test, dt_proba)
    },
    "Neural Network (MLP)": {
        'pred': mlp_pred, 'proba': mlp_proba,
        'acc': accuracy_score(y_test, mlp_pred),
        'prec': precision_score(y_test, mlp_pred, average='weighted'),
        'rec': recall_score(y_test, mlp_pred, average='weighted'),
        'auc': roc_auc_score(y_test, mlp_proba)
    }
}

print("\nMODEL PERFORMANCE\n")
for name, r in results.items():
    print(f"{name}")
    print(f"   Accuracy : {r['acc']:.4f}")
    print(f"   Precision: {r['prec']:.4f}")
    print(f"   Recall   : {r['rec']:.4f}")
    print(f"   AUC      : {r['auc']:.4f}")
    print("-" * 40)

print("===================== ACCURACY COMPARISON PLOT =====================")
plt.figure(figsize=(9, 6))
plt.bar(results.keys(), [r['acc'] for r in results.values()], color=['#3498db', '#2ecc71', '#9b59b6'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate([r['acc'] for r in results.values()]):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
plt.close()
print("   -> Saved: accuracy_comparison.png")

print("===================== CONFUSION MATRICES =====================")
for name, r in results.items():
    cm = confusion_matrix(y_test, r['pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"), dpi=300)
    plt.close()
    print(f"   -> Saved: confusion_matrix_{safe_name}.png")

print("===================== ROC CURVE COMPARISON =====================")
plt.figure(figsize=(10, 8))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {r['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_comparison.png"), dpi=300)
plt.close()
print("   -> Saved: roc_curve_comparison.png")
