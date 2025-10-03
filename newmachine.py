

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import traceback

# helper to display DataFrame nicely in notebook UI (ace_tools is available in this environment)
try:
    from ace_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None

paths = ["weather_data.csv", "weather_forecast_data.csv"]
dfs = []
for p in paths:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            df['_source_file'] = os.path.basename(p)
            dfs.append(df)
            print(f"Loaded {p} shape={df.shape}")
        except Exception as e:
            print(f"Failed to read {p}: {e}")
    else:
        print(f"File not found: {p}")

if len(dfs) == 0:
    raise SystemExit("No input CSVs found at the expected paths. Please upload files to /mnt/data and retry.")

# Attempt to combine datasets intelligently
def combine_datasets(dfs):
    # If columns sets match (or one is subset), do vertical concat on shared columns
    cols_sets = [set(df.columns) for df in dfs]
    common_cols = set.intersection(*cols_sets)
    if len(common_cols) >= max(len(cols_sets[0]), len(cols_sets[-1])) * 0.5:
        # keep common columns (plus source tag)
        common = sorted(list(common_cols))
        combined = pd.concat([df[common + ['_source_file']] if '_source_file' in df.columns else df[common] for df in dfs], ignore_index=True, sort=False)
        return combined
    # else if there is a datetime-like column common, try merging on it
    datetime_like = None
    for col in set.intersection(*cols_sets):
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_like = col
            break
    if datetime_like:
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=datetime_like, how='outer', suffixes=('','_r'))
        return merged
    # fallback: horizontal join (align by index) with suffixes
    merged = dfs[0].copy()
    for i, df in enumerate(dfs[1:], start=1):
        merged = pd.concat([merged.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    return merged

combined = combine_datasets(dfs)
print("Combined dataset shape:", combined.shape)
if display_dataframe_to_user:
    display_dataframe_to_user("Combined dataset preview", combined.head(200))
else:
    print(combined.head())

# Choose target column heuristically: common names or last column (excluding _source_file)
possible_targets = ['target','label','y','outcome','rain','raintomorrow','rain_tomorrow','RainTomorrow','RainToday','Rain']
target = None
cols = [c for c in combined.columns if c != '_source_file']
lowercols = {c.lower(): c for c in cols}
for cand in possible_targets:
    if cand.lower() in lowercols:
        target = lowercols[cand.lower()]
        break
if target is None:
    # choose last column (not source file)
    if len(cols) >= 2:
        target = cols[-1]
    else:
        raise SystemExit("Couldn't determine a target column automatically. Please ensure your CSVs contain a label column.")

print("Selected target column:", target)

# Preprocessing
data = combined.copy()

# Drop rows where target is missing
data = data.dropna(subset=[target])
print(f"After dropping missing target rows: {data.shape}")


# Separate features and label
X = data.drop(columns=[target, '_source_file'] if '_source_file' in data.columns else [target])
y = data[target].copy()

# Encode y if categorical
if y.dtype == object or y.dtype.name == 'category' or y.nunique() <= 10:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    classes = le.classes_
else:
    # if numeric but continuous, try to binarize if many unique values
    if y.nunique() > 10:
        # regression-like target -> convert to binary by median split
        med = y.median()
        y_enc = (y > med).astype(int).values
        classes = np.array(['<=median','>median'])
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        classes = le.classes_

print("Classes:", classes)

# Simple feature processing: drop columns with >50% missing, fill numeric with median, categorical with mode
X_proc = X.copy()
missing_frac = X_proc.isna().mean()
drop_cols = missing_frac[missing_frac > 0.5].index.tolist()
if drop_cols:
    print("Dropping columns with >50% missing:", drop_cols)
    X_proc = X_proc.drop(columns=drop_cols)
for col in X_proc.columns:
    if X_proc[col].dtype == object or X_proc[col].dtype.name == 'category':
        X_proc[col] = X_proc[col].fillna(X_proc[col].mode().iloc[0] if not X_proc[col].mode().empty else 'missing')
    else:
        X_proc[col] = X_proc[col].fillna(X_proc[col].median())

# One-hot encode categorical features
cat_cols = X_proc.select_dtypes(include=['object','category']).columns.tolist()
if len(cat_cols) > 0:
    X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)
print("Processed features shape:", X_proc.shape)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_proc.values)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(np.unique(y_enc))>1 else None)
print("Train/test shapes:", X_train.shape, X_test.shape)

# Train classical baseline - RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_clf = accuracy_score(y_test, y_pred)
print(f"RandomForest accuracy: {acc_clf:.4f}")
print("Classification report for RandomForest:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))

# Confusion matrix plot for classical model
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion matrix — RandomForest')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(range(len(classes)), [str(c) for c in classes], rotation=45)
plt.yticks(range(len(classes)), [str(c) for c in classes])
for (i,j),val in np.ndenumerate(cm):
    plt.text(j, i, val, ha='center', va='center')
plt.tight_layout()
plt.show()

# Attempt quantum model with PennyLane if available
quantum_results = {}
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    print("PennyLane version:", qml.__version__)
    # Use a small subset if dataset is large
    max_samples = 200
    Xq = X_scaled.copy()
    yq = np.array(y_enc).copy()
    if Xq.shape[0] > max_samples:
        # stratified subsample
        _, Xq, _, yq = train_test_split(Xq, yq, train_size=max_samples, stratify=yq, random_state=1)
    # reduce features to at most 6 by PCA-like via SVD
    from sklearn.decomposition import PCA
    n_features_q = min(6, Xq.shape[1])
    pca = PCA(n_components=n_features_q)
    Xq_red = pca.fit_transform(Xq)
    print("Quantum model using reduced features:", Xq_red.shape[1])

    # Normalize to [0, pi] for angle embedding
    Xq_norm = (Xq_red - Xq_red.min())/(Xq_red.max()-Xq_red.min()+1e-9) * np.pi

    n_qubits = n_features_q
    dev = qml.device("default.qubit", wires=n_qubits)

    def variational_circuit(x, weights):
        # AngleEmbedding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        # variational layers
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # measure expectation for first wire in PauliZ basis as logits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # shape of weights: (n_layers, n_qubits, 3)
    n_layers = 2
    weight_shape = (n_layers, n_qubits, 3)
    qnode = qml.QNode(variational_circuit, dev, interface='autograd')

    # simple model: linear mapping from n_qubits outputs to class logits
    rng = np.random.RandomState(42)
    weights = 0.01 * rng.randn(*weight_shape)
    linear_w = 0.01 * rng.randn(n_qubits, len(np.unique(yq)))

    # convert to pnp arrays for autograd
    weights = pnp.array(weights, requires_grad=True)
    linear_w = pnp.array(linear_w, requires_grad=True)

    # one-hot labels
    num_classes = len(np.unique(yq))
    Y_onehot = np.eye(num_classes)[yq]

    def model_forward(x, weights, linear_w):
        outs = qnode(x, weights)  # length n_qubits
        outs = pnp.array(outs)
        logits = pnp.dot(outs, linear_w)
        # softmax
        exp = pnp.exp(logits - pnp.max(logits))
        probs = exp / pnp.sum(exp)
        return probs

    # loss over dataset
    def loss(weights, linear_w, Xb, Yb):
        L = 0.0
        for xi, yi in zip(Xb, Yb):
            probs = model_forward(xi, weights, linear_w)
            L = L - pnp.sum(yi * pnp.log(probs + 1e-9))
        return L / len(Xb)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    epochs = 8
    batch_size = 16

    # training loop (small number of epochs to keep runtime low)
    for epoch in range(epochs):
        # mini-batch SGD
        perm = rng.permutation(len(Xq_norm))
        for i in range(0, len(Xq_norm), batch_size):
            batch_idx = perm[i:i+batch_size]
            Xb = Xq_norm[batch_idx]
            Yb = Y_onehot[batch_idx]
            weights, linear_w, _ = opt.step_and_cost(lambda w, lw: loss(w, lw, Xb, Yb), weights, linear_w)
        if epoch % 1 == 0:
            curr_loss = loss(weights, linear_w, Xq_norm, Y_onehot)
            print(f"Epoch {epoch+1}/{epochs} - loss: {curr_loss:.4f}")

    # evaluation on held-out classical test set (project through same PCA and normalization)
    X_test_red = pca.transform(X_test)
    X_test_norm = (X_test_red - Xq_red.min(axis=0))/(Xq_red.max(axis=0)-Xq_red.min(axis=0)+1e-9) * np.pi
    y_preds = []
    for xi in X_test_norm:
        probs = model_forward(xi, weights, linear_w)
        y_preds.append(int(pnp.argmax(probs)))
    y_preds = np.array(y_preds)
    acc_q = accuracy_score(y_test[:len(y_preds)], y_preds)
    quantum_results['accuracy'] = float(acc_q)
    print(f"Quantum model accuracy (approx): {acc_q:.4f}")
    print("Quantum model classification report:")
    print(classification_report(y_test[:len(y_preds)], y_preds, target_names=[str(c) for c in classes]))
except Exception as e:
    print("Could not run PennyLane quantum model in this environment. Reason:")
    traceback.print_exc()
    print("Proceeding without a quantum model. The classical RandomForest results above are valid.")

# Summary outputs
print("\\nSUMMARY:")
print(f"Classical RandomForest accuracy: {acc_clf:.4f}")
if 'accuracy' in quantum_results:
    print(f"Quantum model accuracy: {quantum_results['accuracy']:.4f}")
else:
    print("Quantum model: not available in this environment.")

# If we created any dataframes, display small summaries
summary_df = pd.DataFrame({
    'model': ['RandomForest'] + (['QuantumVariational'] if 'accuracy' in quantum_results else []),
    'accuracy': [acc_clf] + ([quantum_results.get('accuracy', None)] if 'accuracy' in quantum_results else [])
})
if display_dataframe_to_user:
    display_dataframe_to_user("Model comparison", summary_df)
else:
    print(summary_df)

    
# Save a short report to /mnt/data/ml_report.txt
report_path = "ml_report.txt"
with open(report_path, "w") as f:
    f.write("Model comparison report\\n")
    f.write(summary_df.to_string(index=False))
    f.write("\\n\\nRandomForest classification report:\\n")
    f.write(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))
    if 'accuracy' in quantum_results:
        f.write("\\n\\nQuantum model accuracy:\\n")
        f.write(str(quantum_results['accuracy']))

print(f"Report saved to: {report_path}")
