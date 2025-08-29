import streamlit as st
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Platform", layout="wide")

st.title("Preprocess your data")
# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")

# Store uploaded file in session so it persists across pages
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
elif "uploaded_file" in st.session_state:
    uploaded_file = st.session_state.uploaded_file

# Initialize session state for dataframe
if "df" not in st.session_state:
    st.session_state.df = None

# Validate file
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        st.success("✅ File accepted successfully!")

        # Load dataset only once
        if st.session_state.df is None:
            st.session_state.df = pd.read_csv(uploaded_file)

        # 🔹 Ask user what percentage of dataset to keep
        st.write("### 📌 Dataset Sampling")
        perc = st.slider("Select percentage of records to keep:", 10, 100, 100, step=10)
        if perc < 100:
            n_rows = int(len(st.session_state.df) * (perc / 100))
            st.session_state.df = st.session_state.df.sample(n=n_rows, random_state=42).reset_index(drop=True)
            st.info(f"Using {perc}% of dataset ({n_rows} rows).")

        df = st.session_state.df  # work with session copy

        st.write("### Preview of Dataset:")
        st.dataframe(df, use_container_width=True, height=400)  # always updated

        st.write(f"size = {st.session_state.df.shape}")

        col_dtype, col_change = st.columns([1,2])

        with col_dtype:
            # Show column names with data types
            st.write("### Column Data Types")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str)})
            st.dataframe(dtype_df["Data Type"], use_container_width=True, height=250) 

        with col_change:
            st.write("### 🔹 Changes in columns")
            columns = ["Not selected"] + list(df.columns)
            
            # Drop column
            drop_col = st.selectbox("Drop a column:", columns, index=0, key="drop")
            
            # Rename column
            rename = st.selectbox("Rename a column:", columns, index=0, key="rename")
            new_name = None
            if rename != "Not selected":
                new_name = st.text_input(f"Enter new name for column '{rename}':", key="rename_text")
            
            # Encoding of column
            encode = st.selectbox("Encode any categorical column:", columns, index=0, key="encode")

            # Apply button
            if st.button("Apply Changes"):
                if drop_col != "Not selected":
                    st.session_state.df = st.session_state.df.drop(columns=[drop_col])
                    st.success(f"🗑️ Column '{drop_col}' dropped successfully!")

                if rename != "Not selected" and new_name:
                    st.session_state.df = st.session_state.df.rename(columns={rename: new_name})
                    st.success(f"✅ Column '{rename}' renamed to '{new_name}'")

                if encode != "Not selected" and df[encode].dtype == "object":
                    st.session_state.df = pd.get_dummies(st.session_state.df, columns=[encode], dtype=int)
                    st.success(f"✅ Column '{encode}' has been encoded")

                st.rerun()  # refresh page to update preview with changes
            
        st.write("### Dataset Description")
        st.write(st.session_state.df.describe(include="all"))

        # show all the null records
        if df.isnull().sum().sum()>0:
            null_records = st.session_state.df[st.session_state.df.isnull().any(axis=1)]
            st.write("### Rows with Null Values")
            st.dataframe(null_records, use_container_width=True, height=200)
            st.write(f"There are total of {df.isnull().sum().sum()} null values")
            y_or_n = st.selectbox("Do you want to remove all the null values", ["NO", "YES", "FILL THE RECORDS"], index = 0)
            if y_or_n == "YES":
                st.session_state.df = st.session_state.df.dropna()
                st.rerun()
            if y_or_n == "NO":
                pass
            if y_or_n == "FILL THE RECORDS":
                fill_method = st.selectbox("How to handle null values?", 
                           ["Not selected", "Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill"])
                col = st.selectbox("Select column to fill:", st.session_state.df.columns)
                if st.session_state.df[col].dtype == "object":
                    st.write("Select an integer or decimal datatype")
                elif st.button("Apply Fill"):
                    if fill_method == "Mean":
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mean())
                    elif fill_method == "Median":
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].median())
                    elif fill_method == "Mode":
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mode()[0])
                    elif fill_method == "Zero":
                        st.session_state.df[col] = st.session_state.df[col].fillna(0)
                    elif fill_method == "Forward Fill":
                        st.session_state.df[col] = st.session_state.df[col].fillna(method="ffill")
                    elif fill_method == "Backward Fill":
                        st.session_state.df[col] = st.session_state.df[col].fillna(method="bfill")
                    st.rerun()
        
        # Feature binning 
        st.write("### 🔹 Feature Binning")

        #  Select numeric column
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for binning.")
        else:
            col_to_bin = st.selectbox("Select column for binning:", numeric_cols, key="bin_col")
            n_bins = st.slider("Number of bins:", 2, 10, 4, key="bin_slider")
            strategy = st.radio("Choose Binning Strategy:", 
                                ["Equal Width", "Equal Frequency", "Custom"], key="bin_strategy")

            new_col_name = st.text_input("Enter name for new binned column:", f"{col_to_bin}_binned", key="bin_name")

            if strategy == "Custom":
                custom_edges = st.text_input("Enter custom bin edges (comma separated, e.g. 0,18,35,50,100)", key="bin_edges")
            
            if st.button("Apply Binning"):
                if strategy == "Equal Width":
                    st.session_state.df[new_col_name] = pd.cut(st.session_state.df[col_to_bin], bins=n_bins)
                elif strategy == "Equal Frequency":
                    st.session_state.df[new_col_name] = pd.qcut(st.session_state.df[col_to_bin], q=n_bins)
                elif strategy == "Custom" and custom_edges:
                    try:
                        edges = [float(x) for x in custom_edges.split(",")]
                        st.session_state.df[new_col_name] = pd.cut(st.session_state.df[col_to_bin], bins=edges)
                    except:
                        st.error("Invalid bin edges. Please enter numeric values separated by commas.") 
                st.rerun()

        # ================= FEATURE SELECTION ===================
        st.write("### 🔹 Feature Selection")

        target_col = st.selectbox("Select target column:", df.columns, key="target_col")

        # Only allow numeric target for regression stats, categorical for classification
        method = st.selectbox("Choose Feature Selection Method:", 
                              ["Not selected", "Correlation (Pearson)", "Chi-Square", "ANOVA (f_classif)", 
                               "Mutual Information", "Wrapper (RFE)", "Embedded (Tree Based)"], key="fs_method")

        threshold = st.slider("Select importance threshold (for display):", 0.0, 1.0, 0.0, 0.01, key="fs_threshold")

        if st.button("Run Feature Selection"):

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Convert categorical vars to dummy
            X = pd.get_dummies(X, drop_first=True, dtype=int)

            feature_scores = pd.Series(dtype="float64")

            try:
                if method == "Correlation (Pearson)":
                    corr = X.corrwith(y, method="pearson").abs()
                    feature_scores = corr

                elif method == "Chi-Square":
                    from sklearn.preprocessing import MinMaxScaler
                    X_norm = MinMaxScaler().fit_transform(X)
                    chi_vals, _ = chi2(X_norm, y)
                    feature_scores = pd.Series(chi_vals, index=X.columns)

                elif method == "ANOVA (f_classif)":
                    f_vals, _ = f_classif(X, y)
                    feature_scores = pd.Series(f_vals, index=X.columns)

                elif method == "Mutual Information":
                    mi = mutual_info_classif(X, y)
                    feature_scores = pd.Series(mi, index=X.columns)

                elif method == "Wrapper (RFE)":
                    if y.dtype == "object":
                        model = LogisticRegression(max_iter=200)
                    else:
                        model = LinearRegression()
                    selector = RFE(model, n_features_to_select=5)
                    selector = selector.fit(X, y)
                    feature_scores = pd.Series(selector.ranking_, index=X.columns)
                    feature_scores = 1 / feature_scores  # smaller rank = higher importance

                elif method == "Embedded (Tree Based)":
                    if y.dtype == "object":
                        model = RandomForestClassifier()
                    else:
                        model = RandomForestRegressor()
                    model.fit(X, y)
                    feature_scores = pd.Series(model.feature_importances_, index=X.columns)

                if not feature_scores.empty:
                    feature_scores = feature_scores / feature_scores.max()  # normalize to [0,1]
                    feature_scores = feature_scores.sort_values(ascending=False)

                    selected = feature_scores[feature_scores >= threshold]

                    st.write("### 📊 Feature Importance Ranking")
                    st.dataframe(selected)

                    # Horizontal bar plot
                    fig, ax = plt.subplots(figsize=(8, max(3, len(selected) * 0.3)))
                    selected.sort_values().plot(kind="barh", ax=ax)
                    ax.set_xlabel("Importance Score")
                    ax.set_ylabel("Features")
                    ax.set_title(f"Feature Selection using {method}")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

        # Download modified dataset
        st.write("### 📥 Download Modified Dataset")

        if st.session_state.df is not None:
            if st.button("Generate Download Link"):
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                label="Download CSV",
                data=csv,
                file_name="modified_dataset.csv",
                mime="text/csv"
                )



    else:
        st.error("❌ Please kindly enter a CSV file")
else:
    st.warning("⚠️ No file uploaded yet")

