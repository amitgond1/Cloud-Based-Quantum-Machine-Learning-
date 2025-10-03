
import os
import argparse
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.decomposition import PCA

def safe_onehot_encoder(*args, **kwargs):
    # wrapper to handle sklearn changes between versions (sparse -> sparse_output)
    if sklearn.__version__ >= "1.2":
        # newer sklearn uses sparse_output
        if 'sparse' in kwargs:
            kwargs.pop('sparse')
        kwargs['sparse_output'] = False
    else:
        # older sklearn expects 'sparse'
        kwargs['sparse'] = False
    return OneHotEncoder(*args, **kwargs)

def read_and_combine(path1, path2):
    dfs = []
    for p in (path1, path2):
        if p and os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"Warning: file not found: {p}")
    if not dfs:
        raise FileNotFoundError("No input CSVs found.")
    # simple intelligent combine: if shared columns available, vertical concat on shared cols else concat horizontally
    cols_sets = [set(df.columns) for df in dfs]
    common = set.intersection(*cols_sets)
    if len(common) >= min(map(len, cols_sets)) * 0.5:
        common = sorted(list(common))
        combined = pd.concat([df[common].assign(_source_file=os.path.basename(p)) for df, p in zip(dfs, (path1, path2))], ignore_index=True, sort=False)
    else:
        # fallback: concat by columns aligning indices
        combined = pd.concat(dfs, axis=1)
    return combined

def preprocess_features(df, target_col):
    # drop rows with missing target
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # encode target if categorical
    if y.dtype == object or y.nunique() <= 10:
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        classes = le.classes_
    else:
        # regression target (too many unique values)
        # for this script we convert to regression handling separately
        y_enc = y.values
        classes = None

    # simple handling: drop columns with >50% missing
    missing_frac = X.isna().mean()
    drop_cols = missing_frac[missing_frac > 0.5].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # fill numeric and categorical
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == 'category':
            if X[col].mode().empty:
                X[col] = X[col].fillna('missing')
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    # one-hot encode categorical columns (using wrapper)
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    if cat_cols:
        ohe = safe_onehot_encoder(handle_unknown='ignore')
        ohe_arr = ohe.fit_transform(X[cat_cols])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), ohe_df], axis=1)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, y_enc, classes, scaler

def train_classical_rf(X_train, y_train, problem='classification', max_samples=200000, n_estimators=50, max_depth=16, n_jobs=1):
    # Subsample if dataset is huge
    if X_train.shape[0] > max_samples:
        print(f"Subsampling training set from {X_train.shape[0]} to {max_samples} for faster training")
        idx = np.random.RandomState(42).choice(np.arange(X_train.shape[0]), size=max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    if problem == 'classification':
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=42)
    else:
        clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main(args):
    # paths
    data1 = args.data
    data2 = args.data2
    outdir = args.outdir or "./output"
    os.makedirs(outdir, exist_ok=True)
    out_report = os.path.join(outdir, "ml_report.txt")

    df = read_and_combine(data1, data2)
    print("Combined shape:", df.shape)
    # try to auto-detect target
    possible_targets = ['target','label','y','outcome','rain','raintomorrow','rain_tomorrow','RainTomorrow','RainToday','Rain','Wind_Speed_kmh']
    target = None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in possible_targets:
        if cand.lower() in cols_lower:
            target = cols_lower[cand.lower()]
            break
    if target is None:
        # fallback to last column
        target = df.columns[-1]
    print("Using target:", target)

    X_scaled, y_enc, classes, scaler = preprocess_features(df, target)

    # detect problem type
    problem = 'classification' if classes is not None else 'regression'
    print("Detected problem type:", problem)

    # split
    if problem == 'classification' and len(np.unique(y_enc)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)

    # Train classical
    clf = train_classical_rf(X_train, y_train, problem=problem, max_samples=200000, n_estimators=50, max_depth=16, n_jobs=1)

    # Evaluate classical
    if problem == 'classification':
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"RandomForest accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes] if classes is not None else None))
    else:
        # regression metrics
        y_pred = clf.predict(X_test)
        # print some regression metrics
        from sklearn.metrics import mean_squared_error, r2_score
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("R2:", r2_score(y_test, y_pred))

    # Attempt PennyLane quantum model (only if pennylane available)
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        print("PennyLane version:", qml.__version__)

        # small subsample for quantum training
        max_q_samples = 200
        Xq = X_scaled.copy()
        yq = np.array(y_enc).copy()
        if Xq.shape[0] > max_q_samples:
            _, Xq, _, yq = train_test_split(Xq, yq, train_size=max_q_samples, stratify=yq if problem=='classification' and len(np.unique(yq))>1 else None, random_state=1)

        # PCA reduce to <= 6 features
        n_features_q = min(6, Xq.shape[1])
        pca = PCA(n_components=n_features_q)
        Xq_red = pca.fit_transform(Xq)
        # normalize to [0, pi]
        Xq_norm = (Xq_red - Xq_red.min())/(Xq_red.max()-Xq_red.min()+1e-9) * np.pi

        n_qubits = n_features_q
        dev = qml.device("default.qubit", wires=n_qubits)

        def variational_circuit(x, weights):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            qml.templates.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        n_layers = 2
        weight_shape = (n_layers, n_qubits, 3)
        qnode = qml.QNode(variational_circuit, dev, interface='autograd')

        rng = np.random.RandomState(42)
        weights = 0.01 * rng.randn(*weight_shape)
        linear_w = 0.01 * rng.randn(n_qubits, len(np.unique(yq)) if problem=='classification' else 1)

        weights = pnp.array(weights, requires_grad=True)
        linear_w = pnp.array(linear_w, requires_grad=True)

        num_classes = len(np.unique(yq)) if problem=='classification' else 1
        if problem=='classification':
            Y_onehot = np.eye(num_classes)[yq]

        def model_forward(x, weights, linear_w):
            outs = qnode(x, weights)
            outs = pnp.array(outs)
            logits = pnp.dot(outs, linear_w)
            if problem == 'classification':
                exp = pnp.exp(logits - pnp.max(logits))
                probs = exp / pnp.sum(exp)
                return probs
            else:
                return logits

        def loss(weights, linear_w, Xb, Yb):
            L = 0.0
            if problem == 'classification':
                for xi, yi in zip(Xb, Yb):
                    probs = model_forward(xi, weights, linear_w)
                    L = L - pnp.sum(yi * pnp.log(probs + 1e-9))
                return L / len(Xb)
            else:
                for xi, yi in zip(Xb, Yb):
                    pred = model_forward(xi, weights, linear_w)
                    L = L + (pred - yi)**2
                return L / len(Xb)

        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        epochs = 8
        batch_size = 16

        for epoch in range(epochs):
            perm = rng.permutation(len(Xq_norm))
            for i in range(0, len(Xq_norm), batch_size):
                batch_idx = perm[i:i+batch_size]
                Xb = Xq_norm[batch_idx]
                if problem == 'classification':
                    Yb = Y_onehot[batch_idx]
                else:
                    Yb = yq[batch_idx]
                # update parameters with opt.step
                # opt.step accepts the function and the parameters as separate args
                weights = opt.step(lambda w: loss(w, linear_w, Xb, Yb), weights)
                # update linear_w using same optimizer by treating it similarly
                linear_w = opt.step(lambda lw: loss(weights, lw, Xb, Yb), linear_w)

            curr_loss = loss(weights, linear_w, Xq_norm, Y_onehot if problem=='classification' else yq)
            print(f"Epoch {epoch+1}/{epochs} - loss: {float(curr_loss):.4f}")

        # Evaluate on the held-out test set (apply same PCA+norm)
        X_test_red = pca.transform(X_test)
        X_test_norm = (X_test_red - Xq_red.min(axis=0))/(Xq_red.max(axis=0)-Xq_red.min(axis=0)+1e-9) * np.pi
        preds = []
        for xi in X_test_norm:
            out = model_forward(xi, weights, linear_w)
            if problem == 'classification':
                preds.append(int(pnp.argmax(out)))
            else:
                preds.append(float(out))
        # compute metrics
        if problem == 'classification':
            from sklearn.metrics import accuracy_score
            acc_q = accuracy_score(y_test[:len(preds)], preds)
            print(f"Quantum model accuracy (approx): {acc_q:.4f}")
        else:
            from sklearn.metrics import mean_squared_error
            print("Quantum proxy MSE:", mean_squared_error(y_test[:len(preds)], preds))
    except Exception as ex:
        print("Could not run full PennyLane quantum model in this environment. Reason:")
        traceback.print_exc()

    # Save report
    ensure_dir(out_report)
    with open(out_report, "w") as f:
        f.write("Model run report\n")
        f.write(f"Combined shape: {df.shape}\n")
        f.write(f"Target: {target}\n")
        if problem == 'classification':
            f.write(f"RandomForest accuracy: {acc:.4f}\n")
    print("Report written to:", out_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="weather_data.csv")
    parser.add_argument("--data2", type=str, default="weather_forecast_data.csv")
    parser.add_argument("--outdir", type=str, default="output")
    args = parser.parse_args()
    main(args)
