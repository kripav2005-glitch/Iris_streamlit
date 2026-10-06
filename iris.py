import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import streamlit as st

# Load pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load Iris dataset for visualizations
@st.cache_data
def get_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df

df = get_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Prediction", "Data Visualization","Flower Details"])

# ================= PAGE 1: PREDICTION =================
if page == "Prediction":
    st.title("Iris Flower Prediction App")

    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)
    predict = st.button("Predict Species")

    if predict:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Species: **{prediction}**")

# ================= PAGE 2: VISUALIZATION =================
elif page == "Data Visualization":
    st.title("Iris Dataset Visualizations")

    # Dataset display toggle
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

    # Feature distribution chart
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select Feature to view distribution:", df.columns[:-1])
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=feature, hue="species", kde=True, ax=ax1, palette="Set2")
    st.pyplot(fig1)

    # Bivariate scatter plot
    st.subheader("Feature Comparison (Scatter Plot)")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis:", df.columns[:-1], index=2)
    with col2:
        y_col = st.selectbox("Y-axis:", df.columns[:-1], index=3)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="species", style="species", s=70, ax=ax2, palette="Set1")
    st.pyplot(fig2)


    # ================= PAGE 3: FLOWER DETAILS =================
elif page == "Flower Details":
    st.title("🌸 Details of Iris Flower Species")
    st.write(
        "The Iris flower dataset, introduced by British statistician Ronald Fisher in 1936, "
        "consists of three related species. Below are their specific physical characteristics and measurements."
    )

    tab1, tab2, tab3 = st.tabs(["Iris Setosa", "Iris Versicolor", "Iris Virginica"])

    with tab1:
        st.header("Iris Setosa")
        col_img, col_info = st.columns([1, 1.2])
        with col_img:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                caption="Iris setosa",
                use_container_width=True
            )
        with col_info:
            st.markdown("""
            * **Key Trait:** Notably smaller petals compared to other species, but relatively wide sepals.
            * **Separability:** Easily linearly separable from Versicolor and Virginica based purely on petal measurements.
            * **Typical Petal Length:** ~1.0 cm – 1.9 cm
            * **Typical Petal Width:** ~0.1 cm – 0.6 cm
            * **Habitat:** Arctic, subarctic coasts, and rocky terrain.
            """)
            
    with tab2:
        st.header("Iris Versicolor")
        col_img, col_info = st.columns([1, 1.2])
        with col_img:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                caption="Iris versicolor (Blue Flag)",
                use_container_width=True
            )
        with col_info:
            st.markdown("""
            * **Key Trait:** Also known as the *Blue Flag*. Falls in the intermediate size range for all petal and sepal measurements.
            * **Separability:** Moderately overlaps with Virginica, making boundary classification more challenging.
            * **Typical Petal Length:** ~3.0 cm – 5.1 cm
            * **Typical Petal Width:** ~1.0 cm – 1.8 cm
            * **Habitat:** Wetlands, marshes, and stream banks across North America.
            """)


    with tab3:
        st.header("Iris Virginica")
        col_img, col_info = st.columns([1, 1.2])
        with col_img:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
                caption="Iris virginica",
                use_container_width=True
            )
    with col_info:    
             st.markdown("""
             * **Distinguishing Features:** Also known as the *Virginia Iris*. Generally the largest of the three species, especially in petal length and width.
             * **Separability:** Overlaps slightly with Versicolor, but distinctly larger than Setosa.
             * **Typical Petal Length:** ~4.5 cm – 6.9 cm
             * **Typical Petal Width:** ~1.4 cm – 2.5 cm
            * **Habitat:** Coastal plains, swamps, and freshwater marshes.
            """)

    st.divider()

    # Species Summary Table
    st.subheader("Statistical Summary by Species")
    st.write("Average measurements (in cm) for each species across all samples:")
    species_summary = df.groupby('species').mean().round(2)
    st.table(species_summary)