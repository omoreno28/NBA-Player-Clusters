import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load Data
data = pd.read_csv('https://raw.githubusercontent.com/omoreno28/NBA-Player-Clusters/main/NBA_PLAYERS.csv')
# Filter relevant numeric columns for clustering
features = ['PTS', 'TRB', 'AST', 'FG%', 'FG3%', 'FT%', 'eFG%', 'PER', 'WS']
data_numeric = data[features]

# Impute missing values with column means
data_numeric.fillna(data_numeric.mean(), inplace=True)

# Standardize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# KMeans Clustering
k = 25  # number of clusters (can be parameterized)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Dimensionality Reduction with PCA for Visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data_scaled)
data['PCA1'] = reduced_data[:, 0]
data['PCA2'] = reduced_data[:, 1]

# Streamlit App
st.title("NBA Player Similarity Finder using K-Means")

# Player selection
player_name = st.selectbox("Select a player:", data['Name'].sort_values())

if player_name:
    player_cluster = data.loc[data['Name'] == player_name, 'Cluster'].values[0]
    st.subheader(f"Players similar to {player_name} (Cluster {player_cluster})")

    # Show similar players in the same cluster
    similar_players = data[data['Cluster'] == player_cluster]
    st.dataframe(similar_players[['Name', 'PTS', 'TRB', 'AST', 'FG%', 'FG3%', 'FT%', 'eFG%', 'PER', 'WS']].sort_values(by='PTS', ascending=False))
    # Visualize clusters in 2D using PCA
    st.markdown("---")
    st.subheader("Cluster Visualization (PCA-reduced)")

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        data['PCA1'], data['PCA2'], 
        c=data['Cluster'], cmap='tab10', alpha=0.6, edgecolor='k'
    )
    selected = data[data['Name'] == player_name]
    ax.scatter(selected['PCA1'], selected['PCA2'], color='red', s=100, label=player_name, edgecolor='black')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-means Cluster Visualization (2D PCA)")
    ax.legend()
    st.pyplot(fig)

    # Optionally show cluster center stats
    st.markdown("---")
    st.subheader("Cluster Center Averages")
    center_scaled = kmeans.cluster_centers_[player_cluster]
    center_unscaled = scaler.inverse_transform(center_scaled.reshape(1, -1))
    center_df = pd.DataFrame(center_unscaled, columns=features)
    st.write(center_df.round(2))
