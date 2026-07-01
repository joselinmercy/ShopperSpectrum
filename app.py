import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="ShopperSpectrum",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------
# Load Files
# ----------------------------
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
rfm = pd.read_csv("data/rfm_customers.csv")
df = pd.read_csv("data/online_retail.csv")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🛒 ShopperSpectrum")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📋 Customer Dataset",
        "📊 Cluster Analysis",
        "🔍 Search Customer",
        "📈 Charts",
        "🎯 Customer Segmentation",
        "🎁 Product Recommendation"
    ]
)

# ----------------------------
# Dashboard
# ----------------------------
if page == "🏠 Dashboard":

    st.title("🛒 ShopperSpectrum")

    st.subheader(
        "Customer Segmentation using RFM Analysis and K-Means Clustering"
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", len(rfm))
    c2.metric("Clusters", rfm["Cluster"].nunique())
    c3.metric("Average Recency", round(rfm["Recency"].mean(),2))
    c4.metric("Average Monetary", round(rfm["Monetary"].mean(),2))

    st.markdown("---")

    left,right = st.columns(2)

    with left:

        fig = px.pie(
            rfm,
            names="Cluster",
            title="Customer Segment Distribution",
            hole=0.4
        )

        st.plotly_chart(fig,use_container_width=True)

    with right:

        cluster_count = (
            rfm["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_count.columns=["Cluster","Customers"]

        fig2 = px.bar(
            cluster_count,
            x="Cluster",
            y="Customers",
            color="Cluster",
            title="Customers in Each Cluster"
        )

        st.plotly_chart(fig2,use_container_width=True)

# ----------------------------
# Dataset
# ----------------------------
elif page == "📋 Customer Dataset":

    st.title("📋 Customer Dataset")

    st.dataframe(rfm,use_container_width=True)

    csv = rfm.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "rfm_customers.csv",
        "text/csv"
    )

# ----------------------------
# Cluster Analysis
# ----------------------------
elif page == "📊 Cluster Analysis":

    st.title("📊 Cluster Analysis")

    cluster = st.selectbox(
        "Select Cluster",
        sorted(rfm["Cluster"].unique())
    )

    data = rfm[rfm["Cluster"]==cluster]

    st.write("Customers :",len(data))

    st.dataframe(data)

# ----------------------------
# Search Customer
# ----------------------------
elif page == "🔍 Search Customer":

    st.title("🔍 Search Customer")

    customer = st.number_input(
        "Enter Customer ID",
        min_value=int(rfm.CustomerID.min()),
        max_value=int(rfm.CustomerID.max()),
        step=1
    )

    result = rfm[rfm.CustomerID==customer]

    if len(result)>0:

        st.success("Customer Found")

        st.dataframe(result)

    else:

        st.error("Customer Not Found")

# ----------------------------
# Charts
# ----------------------------
elif page == "📈 Charts":

    st.title("📈 Customer Analysis")

    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Cluster",
        size="Frequency",
        hover_data=["CustomerID"],
        title="Customer Segmentation"
    )

    st.plotly_chart(fig,use_container_width=True)

    fig2 = px.histogram(
        rfm,
        x="Frequency",
        color="Cluster",
        title="Purchase Frequency Distribution"
    )

    st.plotly_chart(fig2,use_container_width=True)

    fig3 = px.box(
        rfm,
        x="Cluster",
        y="Monetary",
        color="Cluster",
        title="Monetary Value by Cluster"
    )

    st.plotly_chart(fig3,use_container_width=True)

# ----------------------------
# Customer Segmentation
# ----------------------------

elif page == "🎯 Customer Segmentation":

    st.title("🎯 Customer Segmentation")

    st.write("Enter Recency, Frequency and Monetary values.")

    recency = st.number_input(
        "Recency (Days since last purchase)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency (Number of purchases)",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Monetary (Total Spend)",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Segment"):

        sample = pd.DataFrame({
            "Recency":[recency],
            "Frequency":[frequency],
            "Monetary":[monetary]
        })

        sample_scaled = scaler.transform(sample)

        cluster = kmeans.predict(sample_scaled)[0]

        segment_name = {
            0:"🛍️ Regular Customer",
            1:"⚠️ At Risk Customer",
            2:"🌟 High Value Customer",
            3:"🎯 Occasional Shopper"
        }

        st.success(
            f"Customer belongs to : **{segment_name[cluster]}**"
        )
    
# ----------------------------
# Product Recommendation
# ----------------------------

elif page == "🎁 Product Recommendation":

    st.title("🎁 Product Recommendation")

    st.write("Search a product and view similar products.")

    products = sorted(
        df["Description"].dropna().unique()
    )

    product = st.selectbox(
        "Select Product",
        products
    )

    if st.button("Recommend"):

        st.success(f"Selected Product : {product}")

        recommendations = (
            df[
                df["Description"] != product
            ]["Description"]
            .dropna()
            .unique()[:5]
        )

        st.subheader("Recommended Products")

        for item in recommendations:

            st.write("✅", item)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.caption("""
🛒 ShopperSpectrum

Customer Segmentation using RFM Analysis and K-Means Clustering

Developed by Joselin"""
)
