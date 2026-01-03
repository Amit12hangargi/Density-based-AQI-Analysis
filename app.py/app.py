import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
import hdbscan


# --------------------------------------------------
# Page Configuration & Header
# --------------------------------------------------
st.set_page_config(
    page_title="AQI Density-Based Clustering",
    layout="wide"
)

st.markdown("""
# Density-Based Clustering of AQI Patterns
### Respiratory Health-Oriented Pollution Analysis  
**Major Project – SRM Institute of Science and Technology**
""")

st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "data", "city_day.csv")
    return pd.read_csv(file_path)

aqi_df = load_data()

# -------- AQI CAPPING FOR VISUALIZATION ONLY --------
aqi_vis_df = aqi_df.copy()
aqi_vis_df["AQI"] = aqi_vis_df["AQI"].clip(upper=500)

st.success("Dataset loaded successfully")

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
st.markdown("## Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(aqi_df))
col2.metric("Total Cities", aqi_df["City"].nunique())
col3.metric(
    "AQI Range",
    f"{int(aqi_vis_df['AQI'].min())} – {int(aqi_vis_df['AQI'].max())}"
)

st.dataframe(aqi_df.head())

st.markdown("---")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.markdown("## Analysis Controls")
st.sidebar.markdown("Explore AQI patterns city-wise")

cities = sorted(aqi_df["City"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", cities)

city_df = aqi_df[aqi_df["City"] == selected_city]
city_vis_df = aqi_vis_df[aqi_vis_df["City"] == selected_city]

# --------------------------------------------------
# City-wise AQI Analysis
# --------------------------------------------------
st.markdown(f"## AQI Analysis for {selected_city}")
st.dataframe(city_df.head(20))

st.markdown("### AQI Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(city_vis_df["AQI"].dropna(), bins=30, kde=True, ax=ax)
ax.set_xlabel("AQI")
ax.set_ylabel("Frequency")
ax.set_title(f"AQI Distribution – {selected_city}")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# --------------------------------------------------
# Top Polluted Cities
# --------------------------------------------------
st.markdown("## Top 10 Most Polluted Cities (Average AQI)")

top_cities = (
    aqi_vis_df.groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(9, 4))
top_cities.plot(kind="bar", ax=ax)
ax.set_xlabel("City")
ax.set_ylabel("Average AQI")
ax.set_title("Top 10 Polluted Cities in India")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# --------------------------------------------------
# Feature Preparation for Clustering (RAW DATA)
# --------------------------------------------------
features = aqi_df[["PM2.5", "PM10", "NO2", "SO2", "O3", "AQI"]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# DBSCAN
# --------------------------------------------------
st.markdown("## DBSCAN Clustering")

dbscan = DBSCAN(eps=0.8, min_samples=10)
clusters_db = dbscan.fit_predict(X_scaled)

st.write("Cluster Distribution:")
st.write(pd.Series(clusters_db).value_counts())

pca_df_db = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df_db["Cluster"] = clusters_db

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=pca_df_db, x="PC1", y="PC2",
    hue="Cluster", palette="Set2", s=40, ax=ax
)
ax.set_title("DBSCAN Clusters (PCA Projection)")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Observation:** DBSCAN identifies core pollution regimes using fixed density thresholds and marks extreme or irregular AQI patterns as noise.
""")

st.markdown("---")

# --------------------------------------------------
# Cluster-wise Pollution Profile
# --------------------------------------------------
st.markdown("## Cluster-wise Pollution Profile (DBSCAN)")

clustered_df = features.copy()
clustered_df["Cluster"] = clusters_db

profile = clustered_df.groupby("Cluster").mean().reset_index()
st.dataframe(profile)

selected_cluster = st.selectbox("Select Cluster", profile["Cluster"])

cluster_data = profile[
    profile["Cluster"] == selected_cluster
].drop(columns=["Cluster"])

fig, ax = plt.subplots(figsize=(9, 4))
cluster_data.T.plot(kind="bar", legend=False, ax=ax)
ax.set_title(f"Pollution Profile – Cluster {selected_cluster}")
ax.set_ylabel("Average Value")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# --------------------------------------------------
# HDBSCAN
# --------------------------------------------------
st.markdown("## HDBSCAN Clustering")

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=10
)

clusters_hdb = hdbscan_model.fit_predict(X_scaled)

st.write("Cluster Distribution:")
st.write(pd.Series(clusters_hdb).value_counts())

pca_df_hdb = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df_hdb["Cluster"] = clusters_hdb

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=pca_df_hdb,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    s=40,
    ax=ax
)
ax.set_title("HDBSCAN Clusters (PCA Projection)")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Observation:** HDBSCAN adapts to variable-density AQI patterns
and separates dense pollution zones from transitional regions.
""")

st.markdown("---")

# --------------------------------------------------
# Cluster-wise Pollution Profile (HDBSCAN)
# --------------------------------------------------
st.markdown("## Cluster-wise Pollution Profile (HDBSCAN)")

hdb_profile_df = features.copy()
hdb_profile_df["Cluster"] = clusters_hdb   # ✅ FIXED

hdb_profile = hdb_profile_df.groupby("Cluster").mean().reset_index()
st.dataframe(hdb_profile)

selected_hdb_cluster = st.selectbox(
    "Select HDBSCAN Cluster",
    hdb_profile["Cluster"],
    key="hdb_cluster_select"
)

hdb_cluster_data = hdb_profile[
    hdb_profile["Cluster"] == selected_hdb_cluster
].drop(columns=["Cluster"])

fig, ax = plt.subplots(figsize=(9, 4))
hdb_cluster_data.T.plot(kind="bar", legend=False, ax=ax)
ax.set_title(f"HDBSCAN Pollution Profile – Cluster {selected_hdb_cluster}")
ax.set_ylabel("Average Value")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Observation:** HDBSCAN cluster profiles reveal adaptive pollution regimes,
capturing both dense hotspots and transitional AQI behavior.
""")

st.markdown("---")

# --------------------------------------------------
# OPTICS
# --------------------------------------------------
st.markdown("## OPTICS Clustering")

optics_model = OPTICS(
    min_samples=10,
    xi=0.05,
    min_cluster_size=0.05
)

clusters_opt = optics_model.fit_predict(X_scaled)

st.write("Cluster Distribution:")
st.write(pd.Series(clusters_opt).value_counts())

pca_df_opt = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df_opt["Cluster"] = clusters_opt

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=pca_df_opt,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set3",
    s=40,
    ax=ax
)
ax.set_title("OPTICS Clusters (PCA Projection)")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Observation:** OPTICS captures hierarchical density variations
and identifies gradual transitions between AQI regimes.
""")

st.markdown("---")

# --------------------------------------------------
# Cluster-wise Pollution Profile (OPTICS)
# --------------------------------------------------
st.markdown("## Cluster-wise Pollution Profile (OPTICS)")

opt_profile_df = features.copy()
opt_profile_df["Cluster"] = clusters_opt   # ✅ CORRECT VARIABLE

opt_profile = opt_profile_df.groupby("Cluster").mean().reset_index()
st.dataframe(opt_profile)

selected_opt_cluster = st.selectbox(
    "Select OPTICS Cluster",
    opt_profile["Cluster"],
    key="opt_cluster_select"
)

opt_cluster_data = opt_profile[
    opt_profile["Cluster"] == selected_opt_cluster
].drop(columns=["Cluster"])

fig, ax = plt.subplots(figsize=(9, 4))
opt_cluster_data.T.plot(kind="bar", legend=False, ax=ax)
ax.set_title(f"OPTICS Pollution Profile – Cluster {selected_opt_cluster}")
ax.set_ylabel("Average Value")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Observation:** OPTICS cluster profiles expose gradual pollution
intensity changes and transitional AQI behavior across clusters.
""")

st.markdown("---")
# --------------------------------------------------
# Algorithm Comparison
# --------------------------------------------------
st.markdown("## Algorithm Comparison")

def summary(name, labels):
    s = pd.Series(labels)
    return {
        "Algorithm": name,
        "Clusters Found": len(set(labels)) - (1 if -1 in labels else 0),
        "Noise Points": (s == -1).sum()
    }

comparison_df = pd.DataFrame([
    summary("DBSCAN", clusters_db),
    summary("HDBSCAN", clusters_hdb),
    summary("OPTICS", clusters_opt)
])

st.dataframe(comparison_df)

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(comparison_df["Algorithm"], comparison_df["Noise Points"])
ax.set_title("Noise Detection Comparison")
ax.set_ylabel("Noise Points")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Key Insight:** HDBSCAN and OPTICS handle variable-density AQI patterns more robustly compared to DBSCAN.
""")

# --------------------------------------------------
# Respiratory Health Risk Drivers (Visual & Insightful)
# --------------------------------------------------
st.markdown("## 🫁 Respiratory Health Risk Drivers")

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3"]

def respiratory_risk_section(title, cluster_labels, key_prefix):
    st.markdown(f"### {title}")

    rhi_df = features[pollutants].copy()
    rhi_df["Cluster"] = cluster_labels

    selected_cluster = st.selectbox(
        f"Select {title} Cluster",
        sorted(rhi_df["Cluster"].unique()),
        key=key_prefix
    )

    cluster_means = (
        rhi_df[rhi_df["Cluster"] == selected_cluster]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Relative Concentration")
    ax.set_title(f"Respiratory Risk Drivers – {title} Cluster {selected_cluster}")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    dominant_pollutant = cluster_means.idxmax()
    st.success(
        f"🔎 **Key Insight:** Respiratory risk in this cluster is primarily driven by **{dominant_pollutant}**."
    )

    st.markdown("---")


respiratory_risk_section(
    title="DBSCAN",
    cluster_labels=clusters_db,
    key_prefix="dbscan_rhi"
)

respiratory_risk_section(
    title="HDBSCAN",
    cluster_labels=clusters_hdb,
    key_prefix="hdbscan_rhi"
)

respiratory_risk_section(
    title="OPTICS",
    cluster_labels=clusters_opt,
    key_prefix="optics_rhi"
)
