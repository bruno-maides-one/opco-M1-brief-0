from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocessing_old_dataset(df):
    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """
    numerical_cols = ["age", "anciennete_contrat", "nombre_sinistres", ]
    categorical_cols = ["region", "type_contrat"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Prétraitement
    if "id_client" in df.columns:
        X = df.drop(columns=["id_client"])
    else:
        X = df
    if "montant_total_sinistres" in df.columns:
        y = df["montant_total_sinistres"]
    else:
        y = None

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

def preprocessing_new_dataset(df):
    """
    Fonction pour effectuer le prétraitement des données :
    - Imputation des valeurs manquantes.
    - Standardisation des variables numériques.
    - Encodage des variables catégorielles.
    """
    numerical_cols = ["age", "revenu_annuel", "anciennete_contrat", "nombre_sinistres", "franchise", "contacts_service_client", "probabilite_sinistre", "montant_sinistre_estime"]
    categorical_cols = ["region", "type_contrat", "smoker", "participation_prevention"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Prétraitement
    X = df.drop(columns=["id_client"])
    y = df["montant_total_sinistres"]

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor