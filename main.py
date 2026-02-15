import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report






df_train=pd.read_csv(r"kdd_train.csv")
df_test=pd.read_csv(r"kdd_test.csv")
# print(df.head())
# print(df.columns)
# print(df['labels'].unique())
# print("\nLabel counts:")
# print(df['labels'].value_counts())
attack_map = {
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'normal': 'Normal'
}
# Remove trailing dot if present
df_train['labels'] = df_train['labels'].str.strip('.')
df_test['labels'] = df_test['labels'].str.strip('.')

# Apply mapping
df_train['attack_class'] = df_train['labels'].map(attack_map)
df_test['attack_class'] = df_test['labels'].map(attack_map)

# Drop any unmapped rows (safety)
df_train = df_train.dropna(subset=['attack_class'])
df_test = df_test.dropna(subset=['attack_class'])


#print(df['attack_class'].value_counts())
y_train = df_train['attack_class']
X_train = df_train.drop(['labels', 'attack_class'], axis=1)

y_test = df_test['attack_class']
X_test = df_test.drop(['labels', 'attack_class'], axis=1)

# print("\nFeature matrix shape:", X.shape)
# print("Target vector shape:", y.shape)


categorical_cols = ['protocol_type', 'service', 'flag']

X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# print("Before encoding shape:", X.shape)
# print("After encoding shape:", X_encoded.shape)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("Scaled train shape:", X_train_scaled.shape)
# print("Scaled test shape:", X_test_scaled.shape)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
print("\nTraining SVM model...")
svm_model.fit(X_train_scaled, y_train)
print("SVM training completed.")
y_pred_svm = svm_model.predict(X_test_scaled)
print("\nTesting SVM model...")


svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:", svm_accuracy)

cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\nSVM Confusion Matrix:\n", cm_svm)

   
plt.figure(figsize=(8,6))
plt.imshow(cm_svm, interpolation='nearest')
plt.title("SVM Confusion Matrix")
plt.colorbar()

classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.close()


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("Random Forest training completed.")
print("\nTesting Random Forest model...")
y_pred_rf = rf_model.predict(X_test)



rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:", rf_accuracy)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:\n", cm_rf)
print("\nModel Comparison")
print("SVM Accuracy:", svm_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
plt.figure(figsize=(8,6))
plt.imshow(cm_rf, interpolation='nearest')
plt.title("Random forest Confusion Matrix")
plt.colorbar()

classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()

report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)



svm_recalls = [report_svm[c]['recall'] for c in classes]
rf_recalls = [report_rf[c]['recall'] for c in classes]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, svm_recalls, width, label='SVM')
plt.bar(x + width/2, rf_recalls, width, label='Random Forest')

plt.xticks(x, classes)
plt.ylabel("Recall")
plt.title("Class-wise Recall Comparison")
plt.legend()
plt.tight_layout()

plt.savefig("class_recall_comparison.png")
plt.close()
# Accuracy Comparison Plot
models = ['SVM', 'Random Forest']
accuracies = [svm_accuracy, rf_accuracy]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies, color=['steelblue', 'orange'])

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.95,0.97)  # zoomed since values are high
plt.tight_layout()

plt.savefig("accuracy_comparison.png")
plt.close()



    
