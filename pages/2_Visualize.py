import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.title("📊 Visualization")

# Check if dataframe exists
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No dataset available. Please upload and preprocess your data in the Preprocess page.")
else:
    df = st.session_state.df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    all_cols = df.columns.tolist()

    # ---------------- Graph Section ----------------
    st.write("### 📈 Graphs")

    graph_type = st.selectbox(
        "Choose a graph type:",
        ["Scatter Plot", "Bar Plot", "Pie Chart", "Line Graph", "KDE Plot"]
    )

    # Scatter Plot
    if graph_type == "Scatter Plot":
        x_axis = st.selectbox("Select X-axis:", numeric_cols)
        y_axis = st.selectbox("Select Y-axis:", numeric_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        st.pyplot(fig)

    # Bar Plot
    elif graph_type == "Bar Plot":
        col = st.selectbox("Select a column for Bar Plot:", all_cols)
        orientation = st.radio("Orientation:", ["Vertical", "Horizontal"])
        value_counts = df[col].value_counts()
        fig, ax = plt.subplots()
        if orientation == "Vertical":
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        else:
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
        st.pyplot(fig)

    # Pie Chart
    elif graph_type == "Pie Chart":
        col = st.selectbox("Select a column for Pie Chart:", all_cols)
        value_counts = df[col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    # Line Graph
    elif graph_type == "Line Graph":
        x_axis = st.selectbox("Select X-axis (index if None):", [None] + numeric_cols.tolist())
        y_axis = st.selectbox("Select Y-axis:", numeric_cols)
        fig, ax = plt.subplots()
        if x_axis:
            sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
        else:
            sns.lineplot(y=df[y_axis], ax=ax)
        st.pyplot(fig)

    # KDE Plot
    elif graph_type == "KDE Plot":
        col = st.selectbox("Select a numeric column for KDE Plot:", numeric_cols)
        fig, ax = plt.subplots()
        sns.kdeplot(df[col], fill=True, ax=ax)
        st.pyplot(fig)

    # ---------------- Existing Code ----------------
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

    # Boxplot
    st.write("#### Boxplot of a selected column")
    selected_col = st.selectbox("Select a column for Boxplot:", numeric_cols)
    if selected_col:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[selected_col], ax=ax)
        st.pyplot(fig)

    # Outlier detection (IQR)
    st.write("#### Outlier count for all numeric columns (IQR based)")
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = outlier_count

    outlier_df = (
        pd.DataFrame.from_dict(outlier_summary, orient="index", columns=["Outlier Count"])
        .sort_values(by="Outlier Count", ascending=False)
    )
    st.dataframe(outlier_df)

    # Outlier Handling
    st.write("### Handle Outliers(Use wisely, as once deleted can't be recovered)")
    option = st.selectbox(
        "Choose how you want to handle outliers:",
        [
            "Do Nothing",
            "Delete Outliers Manually",
            "Delete Outliers Automatically (Z-Score)",
            "Delete Outliers Automatically (Isolation Forest)"
        ]
    )

    if option == "Delete Outliers Manually":
        st.write("Select row indices to delete from the dataset:")
        selected_rows = st.multiselect("Choose row indices:", df.index.tolist())
        if selected_rows:
            st.write("Rows you selected to delete:", selected_rows)
        if st.button("Delete Selected Rows"):
            st.session_state.df = df.drop(index=selected_rows)
            st.success(f"Deleted {len(selected_rows)} rows successfully.")
            st.rerun()

    elif option == "Delete Outliers Automatically (Z-Score)":
        threshold = st.slider("Select Z-score threshold:", 2.0, 5.0, 3.0, 0.1)
        numeric_df = df[numeric_cols].dropna()
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        outlier_indices = numeric_df[(z_scores > threshold).any(axis=1)].index.tolist()
        st.write(f"Total rows detected as outliers (Z-Score): {len(outlier_indices)}")
        st.dataframe(df.loc[outlier_indices])
        if st.button("Delete All Z-Score Outliers"):
            st.session_state.df = df.drop(index=outlier_indices)
            st.success(f"Deleted {len(outlier_indices)} outlier rows successfully.")
            st.rerun()

    elif option == "Delete Outliers Automatically (Isolation Forest)":
        contamination = st.slider("Select contamination (proportion of outliers):", 0.01, 0.2, 0.05, 0.01)
        numeric_df = df[numeric_cols].dropna()
        if not numeric_df.empty:
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(numeric_df)
            outlier_indices = numeric_df.index[preds == -1].tolist()
            st.write(f"Total rows detected as outliers (Isolation Forest): {len(outlier_indices)}")
            st.dataframe(df.loc[outlier_indices])
            if st.button("Delete All Isolation Forest Outliers"):
                st.session_state.df = df.drop(index=outlier_indices)
                st.success(f"Deleted {len(outlier_indices)} outlier rows successfully.")
                st.rerun()
