from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# numerical_pipeline
def create_numerical_pipeline():

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return numerical_pipeline


#categorical_pipeline
def create_categorical_pipeline():

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return categorical_pipeline


# ColumnTransformer
def create_preprocessor(numerical_features, categorical_features):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", create_numerical_pipeline(), numerical_features),
            ("cat", create_categorical_pipeline(), categorical_features)
        ]
    )

    return preprocessor


# Pipeline
def create_dense_preprocessor(numerical_features, categorical_features):

    numerical_pipeline = create_numerical_pipeline()

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    dense_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        sparse_threshold=0
    )

    return dense_preprocessor