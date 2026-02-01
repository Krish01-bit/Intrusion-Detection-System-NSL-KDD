import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


df=pd.read_csv(r"kdd_train.csv")
# print(df.head())
# print(df.columns)
# print(df['labels'].unique())
# print("\nLabel counts:")
# print(df['labels'].value_counts())
attack_map = {
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS',

    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',

    # R2L
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L',

    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',

    # Normal
    'normal': 'Normal'
}
df['attack_class'] = df['labels'].map(attack_map)
#print(df['attack_class'].value_counts())
y = df['attack_class']
X = df.drop(['labels', 'attack_class'], axis=1)
# print("\nFeature matrix shape:", X.shape)
# print("Target vector shape:", y.shape)


categorical_cols = ['protocol_type', 'service', 'flag']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
# print("Before encoding shape:", X.shape)
# print("After encoding shape:", X_encoded.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("Scaled train shape:", X_train_scaled.shape)
# print("Scaled test shape:", X_test_scaled.shape)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
print("\nTraining SVM model...")
svm_model.fit(X_train_scaled, y_train)
print("SVM training completed.")
print("\nTesting SVM model...")
y_pred_svm = svm_model.predict(X_test_scaled)

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
plt.show()

    
