import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATHS = {
    "cm1": "cm1.csv",
    "kc1": "kc1.csv",
    "jm1": "jm1.csv",
    "pc1": "pc1.csv",
}

RANDOM_SEED = 42
N_FOLDS = 10
TEST_SIZE = 0.2
CORR_THRESHOLD = 0.9

USE_IMBLEARN = False  # set True if imbalanced-learn is installed, otherwise use my own SMOTE below

LABEL_ALIASES = ["defects", "defect", "class", "bug", "label", "target"]
POSITIVE_STRINGS = {"true", "yes", "y", "1", "t", "buggy", "defective"}
NEGATIVE_STRINGS = {"false", "no", "n", "0", "f", "clean", "non-defective"}


def load_raw_file(path: Path) -> pd.DataFrame: # Reads either .csv or .arff since the NASA/PROMISE datasets come in both formats
    """Load CSV or ARFF into a DataFrame."""
    if path.suffix.lower() == ".arff":
        from scipy.io import arff
        data, _ = arff.loadarff(str(path))
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        return df
    return pd.read_csv(path)


def find_label_column(df: pd.DataFrame) -> str:  # Every dataset names its defect column differently, so search for common names first
    # and just grab the last column if nothing matches
    """Identify the defect-label column by name match, fallback to last column."""
    for alias in LABEL_ALIASES:
        for col in df.columns:
            if alias in col.lower():
                return col
    return df.columns[-1]

# The label column can be True/False, yes/no, 1/0, etc depending on the dataset
    # this turns all of that into a plain 0/1 integer
def standardise_label(series: pd.Series) -> pd.Series:
    """Convert defect label of any encoding (bool/str/int) into 0/1 ints."""
    def convert(v):
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (int, np.integer, float, np.floating)):
            return int(v > 0)
        s = str(v).strip().lower()
        if s in POSITIVE_STRINGS:
            return 1
        if s in NEGATIVE_STRINGS:
            return 0
        try:
            return int(float(s) > 0)
        except ValueError:
            raise ValueError(f"Unrecognised label value: {v!r}")
    return series.apply(convert).astype(int)



 # Basic cleaning: drop ID-like columns (they don't help prediction), remove duplicate
def clean_dataframe(df: pd.DataFrame, label_col: str): # rows, and fill any missing values with the column median
    """Drop identifier columns and duplicates; impute missing numeric values with the median."""
    df = df.copy()

    id_like = [c for c in df.columns if re.search(r"(^id$|module|file|name)", c, re.IGNORECASE)]
    id_like = [c for c in id_like if c != label_col]
    if id_like:
        df = df.drop(columns=id_like)

    before = len(df)
    df = df.drop_duplicates()
    dup_removed = before - len(df)

    feature_cols = [c for c in df.columns if c != label_col]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    df = df.dropna(subset=[label_col])
    return df, dup_removed


def remove_correlated_features(df: pd.DataFrame, label_col: str, threshold: float = CORR_THRESHOLD):
    """Drop one of each pair of features correlated above `threshold`."""
    feature_cols = [c for c in df.columns if c != label_col]
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def simple_smote(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED, k_neighbors: int = 5):
    """
    My own basic version of SMOTE (Chawla et al., 2002) in case imbalanced-learn
    isn't available. It finds the k nearest neighbours of each minority-class point
    and creates a new synthetic point somewhere between them, until the minority
    class matches the majority class count.
    """
    rng = np.random.RandomState(random_state)
    counts = y.value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    n_to_generate = counts[majority_class] - counts[minority_class]

    if n_to_generate <= 0:
        return X.copy(), y.copy()

    X_min = X[y == minority_class].reset_index(drop=True)
    k = min(k_neighbors, len(X_min) - 1)
    if k < 1:
        return X.copy(), y.copy()

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min.values)
    _, neighbours = nn.kneighbors(X_min.values)

    synthetic = []
    for _ in range(n_to_generate):
        i = rng.randint(0, len(X_min))
        neighbour_choices = neighbours[i][1:]
        j = neighbour_choices[rng.randint(0, len(neighbour_choices))]
        gap = rng.random()
        new_point = X_min.iloc[i].values + gap * (X_min.iloc[j].values - X_min.iloc[i].values)
        synthetic.append(new_point)

    X_synth = pd.DataFrame(synthetic, columns=X.columns)
    y_synth = pd.Series([minority_class] * len(X_synth))

    X_res = pd.concat([X.reset_index(drop=True), X_synth], ignore_index=True)
    y_res = pd.concat([y.reset_index(drop=True), y_synth], ignore_index=True)
    return X_res, y_res


 
def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]: 
    if USE_IMBLEARN:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_SEED)
        X_res, y_res = sm.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    return simple_smote(X, y)

 # This runs the full pipeline for one dataset: load, clean, split, scale, SMOTE, save
def process_dataset(name: str, filename: str, summary_rows: list):
    path = RAW_DIR / filename
    if not path.exists():
        print(f"[SKIP] {name}: file not found at {path}. Download it first, then re-run.")
        return

    raw_df = load_raw_file(path)
    label_col = find_label_column(raw_df)

    df = raw_df.copy()
    df[label_col] = standardise_label(df[label_col])
    df, dup_removed = clean_dataframe(df, label_col)
    df, dropped_corr_cols = remove_correlated_features(df, label_col)

    X = df.drop(columns=[label_col])
    y = df[label_col]
    defect_rate = y.mean()

# Split before SMOTE so the test set stays untouched
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

 # Scaled version needed separately for SVM and the Neural Network  
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train_res, y_train_res = apply_smote(X_train, y_train)
    X_train_res_scaled, y_train_res_scaled = apply_smote(X_train_scaled, y_train)

    def save(X_part, y_part, suffix):
        out = X_part.copy()
        out[label_col] = y_part.values if hasattr(y_part, "values") else y_part
        out.to_csv(PROC_DIR / f"{name}_{suffix}.csv", index=False)

    save(X_train, y_train, "train")
    save(X_test, y_test, "test")
    save(X_train_res, y_train_res, "train_smote")
    save(X_train_scaled, y_train, "train_scaled")
    save(X_test_scaled, y_test, "test_scaled")
    save(X_train_res_scaled, y_train_res_scaled, "train_smote_scaled")

  # modelling script for WPDP cross-validation; not persisted here to avoid duplication.

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    n_folds_check = sum(1 for _ in skf.split(X, y))

    summary_rows.append({
        "dataset": name,
        "label_column": label_col,
        "n_instances_raw": len(raw_df),
        "n_instances_after_cleaning": len(df),
        "duplicates_removed": dup_removed,
        "correlated_features_dropped": len(dropped_corr_cols),
        "n_features_final": X.shape[1],
        "defect_rate_pct": round(defect_rate * 100, 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_size_after_smote": len(X_train_res),
        "cv_folds_verified": n_folds_check,
    })
    print(f"[OK] {name}: {len(df)} rows -> {X.shape[1]} features, defect rate {defect_rate:.2%}, "
          f"train {len(X_train)} -> {len(X_train_res)} after SMOTE")

def main(): # Loops through all four datasets and writes one combined summary CSV at the end
    summary_rows = []
    for name, filename in DATASET_PATHS.items():
        process_dataset(name, filename, summary_rows)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(PROC_DIR / "summary_report.csv", index=False)
        print("\nSummary written to " + str(PROC_DIR / "summary_report.csv"))
    else:
        print("No datasets processed. Check that files exist in data/raw/.")


if __name__ == "__main__":
    main()