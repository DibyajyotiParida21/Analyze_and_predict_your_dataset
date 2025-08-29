import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🤖 ML Modeling + Hyperparameter Tuning")

# ====== 0) DATA CHECK ======
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No dataset available. Please upload & preprocess your data first.")
    st.stop()

df = st.session_state.df.copy()
st.write("### Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ====== 1) TARGET & FEATURES ======
target_col = st.selectbox("🎯 Select Target Column", options=[""] + list(df.columns))
if target_col == "":
    st.info("👉 Please select a target column to continue.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Detect column types for convenience
all_cols = X.columns.tolist()
numeric_cols_all = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols_default = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# User picks categorical columns explicitly
categorical_cols = st.multiselect(
    "🔠 Select Categorical Columns (will be encoded)",
    options=all_cols,
    default=categorical_cols_default
)
numeric_cols = [c for c in all_cols if c not in categorical_cols]

# ====== 2) TASK TYPE ======
if y.dtype in ["int64", "float64"] and y.nunique() > 15:
    task_type = "Regression"
elif y.nunique() == 2:
    task_type = "Binary Classification"
else:
    task_type = "Multi-class Classification"
st.write(f"📌 Detected Task: **{task_type}** (unique target classes: {y.nunique()})")

# ====== 3) MODEL CHOICE ======
if task_type == "Regression":
    model_choice = st.selectbox("🧠 Select Model", ["Linear Regression", "Random Forest Regressor", "SVM Regressor"])
else:
    model_choice = st.selectbox("🧠 Select Model", ["Logistic Regression", "Random Forest Classifier", "SVM Classifier"])

# ====== 4) ENCODING / SCALING STRATEGY ======
# For RF & SVM → Ordinal (label-like) encoding; For Logistic/Linear → OneHot
use_ordinal = ("Random Forest" in model_choice) or ("SVM" in model_choice)
needs_scaling = ("SVM" in model_choice) or (model_choice == "Logistic Regression")

enc_step = None
if use_ordinal:
    enc_step = ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
else:
    enc_step = ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)

num_step = ("num", (StandardScaler() if needs_scaling else "passthrough"), numeric_cols)

preprocessor = ColumnTransformer(
    transformers=[num_step, enc_step],
    remainder="drop"
)

# ====== 5) BASE MODEL ======
if model_choice == "Linear Regression":
    base_model = LinearRegression()
elif model_choice == "Random Forest Regressor":
    base_model = RandomForestRegressor(random_state=42, n_estimators=200)
elif model_choice == "SVM Regressor":
    base_model = SVR()
elif model_choice == "Logistic Regression":
    base_model = LogisticRegression(max_iter=3000)
elif model_choice == "Random Forest Classifier":
    base_model = RandomForestClassifier(random_state=42, n_estimators=300)
else:  # SVM Classifier
    base_model = SVC(probability=True, random_state=42)

pipe = Pipeline([
    ("pre", preprocessor),
    ("model", base_model)
])

# ====== 6) PARAM GRIDS ======
param_grid = {}

if task_type != "Regression":
    # Classification
    if model_choice == "Logistic Regression":
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": [None, "balanced"]
        }
    elif model_choice == "Random Forest Classifier":
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None]
        }
    else:  # SVM Classifier
        param_grid = {
            "model__C": [0.1, 1, 10, 50],
            "model__kernel": ["rbf", "linear", "poly"],
            "model__gamma": ["scale", "auto"]
        }
else:
    # Regression
    if model_choice == "Linear Regression":
        param_grid = {
            "model__fit_intercept": [True, False],
            # positive param exists in newer sklearn; if error, remove below line.
            "model__positive": [False, True]
        }
    elif model_choice == "Random Forest Regressor":
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None]
        }
    else:  # SVR
        param_grid = {
            "model__C": [0.1, 1, 10, 50],
            "model__kernel": ["rbf", "linear", "poly"],
            "model__gamma": ["scale", "auto"],
            "model__epsilon": [0.01, 0.1, 0.2]
        }

# ====== 7) SPLIT & TUNING CONTROLS ======
test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
cv_folds = st.slider("CV Folds (for tuning)", 3, 10, 5, 1)

# Task-aware scoring
if task_type == "Regression":
    scoring_choice = st.selectbox("Scoring", ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"])
else:
    scoring_choice = st.selectbox("Scoring", ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"])

tune_mode = st.selectbox("Hyperparameter Tuning", ["None", "Grid Search", "Randomized Search"])
n_iter = 20
if tune_mode == "Randomized Search":
    n_iter = st.slider("Randomized Search: n_iter", 5, 100, 20, 5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y if "Classification" in task_type else None
)

# ====== 8) TRAIN ======
if st.button("🚀 Train (with selected tuning)"):
    with st.spinner("Training..."):
        model_to_fit = pipe

        if tune_mode == "None":
            model_to_fit.fit(X_train, y_train)
            best_model = model_to_fit
            best_params = None
            best_cv_score = None
        else:
            if tune_mode == "Grid Search":
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring=scoring_choice,
                    cv=cv_folds,
                    n_jobs=-1,
                    refit=True
                )
            else:
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    scoring=scoring_choice,
                    cv=cv_folds,
                    n_jobs=-1,
                    refit=True,
                    random_state=42
                )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_

    st.success("✅ Training complete!")
    if best_params is not None:
        st.write("**Best Params:**", best_params)
        st.write(f"**Best CV Score ({scoring_choice}):** {best_cv_score:.4f}")

    # ====== 9) EVALUATE ======
    y_pred = best_model.predict(X_test)

    if task_type == "Regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📈 Regression Metrics")
        st.write(f"R²: {r2:.4f}")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAE: {mae:.4f}")

        # Actual vs Predicted
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.subheader("📊 Classification Metrics")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision (weighted): {prec:.4f}")
        st.write(f"Recall (weighted): {rec:.4f}")
        st.write(f"F1 (weighted): {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # store model & schema for prediction
    st.session_state.trained_model = best_model
    st.session_state.predict_cols = list(X.columns)
    st.session_state.predict_num_cols = numeric_cols
    st.session_state.predict_cat_cols = categorical_cols

# ====== 10) PREDICT NEW RECORD ======
if "trained_model" in st.session_state:
    st.subheader("🔮 Predict on New Record")

    new_data = {}
    cols_for_input = st.session_state.predict_cols

    # Simple inputs—string for categorical, number for numeric
    for col in cols_for_input:
        if col in st.session_state.predict_cat_cols:
            new_data[col] = st.text_input(f"{col} (categorical)", "")
        elif col in st.session_state.predict_num_cols:
            new_data[col] = st.text_input(f"{col} (numeric)", "")

    if st.button("Predict"):
        new_df = pd.DataFrame([new_data])

        # Coerce numeric
        for col in st.session_state.predict_num_cols:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

        # Let Pipeline handle encoding/scaling
        pred = st.session_state.trained_model.predict(new_df)

        if "Classification" in task_type:
            st.success(f"📌 Predicted Class: {pred[0]}")
            # Optional: show probabilities if classifier supports it
            try:
                probs = st.session_state.trained_model.predict_proba(new_df)
                proba_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in st.session_state.trained_model.classes_])
                st.write("Class Probabilities:")
                st.dataframe(proba_df)
            except Exception:
                pass
        else:
            st.success(f"📌 Predicted Value: {pred[0]}")
