import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import seaborn as sns

st.title("Online Retail Dataset Menggunakan Algoritma K-Means")

df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')

X = df.drop(["CustomerID", "InvoiceNo","StockCode","Description","InvoiceDate","Country"], axis=1)

st.header("isi dataset")
st.write(X)

k_values = list(range(1, 11))
inertia_values = []

for best_k in k_values:
    kmeans = KMeans(
        n_clusters=best_k,
        init="k-means++",
        tol=0.0001,
        random_state=45,
    )
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve to find the optimal k value
plt.plot(k_values, inertia_values)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")


st.set_option("deprecation.showPyplotGlobalUse", False)
elbo_plot = st.pyplot()

st.header("Nilai jumlah K")
clust = st.slider("Pilih jumlah cluster :", 2, 10, 3, 1)


def k_means(best_k, data):
    # Pisahkan kolom yang akan digunakan untuk clustering
    features = ['Quantity', 'UnitPrice']
    X = data[features]

    # Melakukan K-means clustering
    kmeans = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=45,
    ).fit(X)

    # Menambahkan kolom Cluster ke data
    data["Cluster"] = kmeans.labels_

    # Membuat plot 2D menggunakan plotly express
    fig = px.scatter(data, x='Quantity', y='UnitPrice', color='Cluster')

    # Mengatur layout plot
    fig.update_layout(
        title="K-means Clustering",
        xaxis_title="Quantity",
        yaxis_title="UnitPrice",
    )

    # Menampilkan plot dan data
    st.header("Cluster Plot")
    st.plotly_chart(fig)
    st.write(data)

# Memanggil fungsi k_means dengan jumlah cluster yang dipilih
k_means(clust, X)

