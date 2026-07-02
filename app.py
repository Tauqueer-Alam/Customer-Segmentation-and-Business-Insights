import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

# Set Page Config
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    /* Main Layout Styling */
    .main {
        background-color: #0f1116;
        color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #2d313e;
    }
    
    /* Header/Footer Styling */
    .stAppHeader {
        background-color: transparent;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2230 0%, #151821 100%);
        border: 1px solid #3b4257;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #8c96a8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00ffd0;
        margin-top: 5px;
    }
    
    /* Custom Cluster Segment Cards */
    .cluster-card {
        background-color: #171b26;
        border-left: 5px solid #00ffd0;
        border-top: 1px solid #2d313e;
        border-right: 1px solid #2d313e;
        border-bottom: 1px solid #2d313e;
        border-radius: 4px 12px 12px 4px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s ease-in-out;
    }
    .cluster-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
    .cluster-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .cluster-tag {
        background-color: #1d2b3a;
        color: #00ffd0;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
        border: 1px solid rgba(0, 255, 208, 0.3);
    }
    .cluster-stat {
        font-size: 14px;
        color: #d1d5db;
        margin-bottom: 5px;
    }
    .cluster-recommendation {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        border: 1px dashed #3b4257;
    }
    
    /* Form input styling */
    .stButton>button {
        background: linear-gradient(135deg, #00ffd0 0%, #0088cc 100%) !important;
        color: #0f1116 !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 15px rgba(0, 255, 208, 0.5) !important;
    }
    
    /* Header decoration */
    .header-decor {
        height: 4px;
        background: linear-gradient(90deg, #00ffd0 0%, #0088cc 50%, #ff007f 100%);
        border-radius: 2px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load serialized scaler
@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler.pkl: {e}. Please ensure train_model.py has been run.")
        return None

# Helper function to load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Mall_Customers.csv')
        return df
    except Exception as e:
        st.error(f"Error loading Mall_Customers.csv: {e}")
        return None

# Load resources
scaler = load_scaler()
raw_df = load_data()

# Dynamic Clustering & Profiling Engine
def run_clustering(raw_df, algorithm, params, scaler):
    if raw_df is None or scaler is None:
        return None, None, None
    
    # 1. Preprocess raw data for clustering
    df_clean = raw_df.drop('CustomerID', axis=1, errors='ignore').copy()
    df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
    
    # Scale data
    scaled_data = scaler.transform(df_clean)
    
    # 2. Fit selected model
    if algorithm == "K-Means":
        model = KMeans(n_clusters=params['k_clusters'], random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)
    elif algorithm == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=params['h_clusters'], linkage=params['linkage'])
        labels = model.fit_predict(scaled_data)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = model.fit_predict(scaled_data)
        
    # 3. Fit a KNN classifier wrapper to predict cluster labels for unseen data
    knn = KNeighborsClassifier(n_neighbors=min(3, len(scaled_data)))
    knn.fit(scaled_data, labels)
    
    return labels, model, knn

def generate_cluster_info(raw_df, labels):
    df_temp = raw_df.copy()
    df_temp['Cluster'] = labels
    
    cluster_info = {}
    unique_labels = sorted(list(set(labels)))
    
    # Color palette for clusters (excluding outliers)
    colors_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    
    for idx, label in enumerate(unique_labels):
        if label == -1:
            # DBSCAN Outliers
            cluster_info[-1] = {
                "name": "Outliers / Noise",
                "color": "#7f8c8d",
                "description": "Customers who do not fit into any dense cluster. They exhibit unique, atypical purchasing behaviors.",
                "gender": "N/A",
                "age": "N/A",
                "income": "N/A",
                "spending": "N/A",
                "strategy": "Engage individually. Monitor for anomalous transactions or highly customized, bespoke promotional outreach."
            }
            continue
            
        subset = df_temp[df_temp['Cluster'] == label]
        size = len(subset)
        
        mean_age = subset['Age'].mean()
        min_age = subset['Age'].min()
        max_age = subset['Age'].max()
        
        mean_inc = subset['Annual Income (k$)'].mean()
        min_inc = subset['Annual Income (k$)'].min()
        max_inc = subset['Annual Income (k$)'].max()
        
        mean_spend = subset['Spending Score (1-100)'].mean()
        min_spend = subset['Spending Score (1-100)'].min()
        max_spend = subset['Spending Score (1-100)'].max()
        
        # Gender composition
        gender_counts = subset['Gender'].value_counts()
        male_pct = (gender_counts.get('Male', 0) / size) * 100
        female_pct = (gender_counts.get('Female', 0) / size) * 100
        
        if male_pct > 80:
            gender_desc = "Predominantly Male"
        elif female_pct > 80:
            gender_desc = "Predominantly Female"
        else:
            gender_desc = f"Mixed ({male_pct:.0f}% Male, {female_pct:.0f}% Female)"
            
        # Dynamic naming based on mean income and mean spending score
        if mean_inc > 70 and mean_spend > 60:
            name = "Affluent Spenders"
            desc = "High-income customers with high retail spending. They represent the highest lifetime value segment."
            strategy = "Provide VIP access, premium loyalty rewards, exclusive brand previews, and direct white-glove communications."
        elif mean_inc > 70 and mean_spend <= 35:
            name = "Conservative Affluent"
            desc = "High-income customers who are highly conservative in their spending. They are value-focused and pragmatic."
            strategy = "Promote investment-grade products, high durability items, and premium bulk deals. Avoid low-quality discounts."
        elif mean_inc <= 45 and mean_spend > 60:
            name = "Active Budget Spenders"
            desc = "Low-income customers who spend highly. They are likely young, fashion-conscious, and trend-driven."
            strategy = "Target with student discounts, flash sales, trendy social-media driven promotions, and mobile-first deals."
        elif mean_inc <= 45 and mean_spend <= 35:
            name = "Frugal Shoppers"
            desc = "Low-income and low-spending customers. They are highly price-sensitive and focus on essentials."
            strategy = "Offer deep discounts, value bundle packs, and budget-friendly essentials. Highlight cost savings."
        else:
            if mean_age < 35:
                name = "Youthful Balanced Shoppers"
                desc = "Younger customers with moderate incomes and spending patterns. They balance budget and trends."
                strategy = "Target with buy-now-pay-later (BNPL) schemes, reward points, and interactive social media campaigns."
            else:
                name = "Mature Balanced Shoppers"
                desc = "Older customers with moderate incomes and spending patterns. They value stability and practical sales."
                strategy = "Target with loyalty benefits, utility-driven messaging, direct email marketing, and family-oriented bundle offers."
        
        color = colors_palette[label % len(colors_palette)]
        
        cluster_info[label] = {
            "name": name,
            "color": color,
            "description": desc,
            "gender": gender_desc,
            "age": f"{'Young' if mean_age < 35 else 'Older'} (Mean: {mean_age:.1f} yrs, Range: {min_age} - {max_age})",
            "income": f"{'High' if mean_inc > 70 else ('Low' if mean_inc < 45 else 'Moderate')} (Mean: ${mean_inc:.1f}k, Range: ${min_inc}k - ${max_inc}k)",
            "spending": f"{'High' if mean_spend > 60 else ('Low' if mean_spend < 35 else 'Moderate')} (Mean: {mean_spend:.1f}, Range: {min_spend} - {max_spend})",
            "strategy": strategy
        }
        
    return cluster_info

# Sidebar Navigation & Algorithm Control Panel
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3225/3225196.png", width=80)
    st.title("Navigation")
    page = st.radio(
        "Select a Workspace Section:",
        ["📈 Dashboard & Insights", "📊 Exploratory Data Analysis", "🎯 Single Predictor", "📁 Batch Segmentation"]
    )
    
    st.markdown("---")
    st.markdown("### Clustering Control Panel")
    algorithm = st.selectbox(
        "Clustering Algorithm:",
        ["K-Means", "Hierarchical", "DBSCAN"]
    )
    
    params = {}
    if algorithm == "K-Means":
        params['k_clusters'] = st.slider("Number of Clusters (K):", min_value=2, max_value=10, value=5)
        optimal_str = f"{params['k_clusters']}"
    elif algorithm == "Hierarchical":
        params['h_clusters'] = st.slider("Number of Clusters (K):", min_value=2, max_value=10, value=5)
        params['linkage'] = st.selectbox("Linkage Method:", ["ward", "complete", "average", "single"])
        optimal_str = f"{params['h_clusters']} (Linkage: {params['linkage']})"
    elif algorithm == "DBSCAN":
        params['eps'] = st.slider("Epsilon (Radius):", min_value=0.1, max_value=2.0, value=0.5, step=0.05)
        params['min_samples'] = st.slider("Min Samples:", min_value=2, max_value=15, value=5)
        optimal_str = "Auto (Density-based)"
        
    st.markdown("---")
    st.markdown("### Active Model Info")
    st.markdown(f"- **Algorithm**: {algorithm}")
    if algorithm != "DBSCAN":
        st.markdown(f"- **Clusters**: {optimal_str}")
    else:
        st.markdown(f"- **Eps / MinSamples**: {params['eps']} / {params['min_samples']}")
    st.markdown("- **Evaluated Features**: Gender, Age, Income, Spending")

# Execute Clustering on the fly
labels, fitted_model, predictor_knn = run_clustering(raw_df, algorithm, params, scaler)

# Generate Dynamic Segment Metadata
if labels is not None:
    CLUSTER_INFO = generate_cluster_info(raw_df, labels)
    raw_df['Cluster'] = labels
    raw_df['Cluster Name'] = raw_df['Cluster'].map(lambda x: CLUSTER_INFO[x]['name'])
else:
    CLUSTER_INFO = {}

# ================= PAGE 1: DASHBOARD & INSIGHTS =================
if page == "📈 Dashboard & Insights":
    st.title("📊 Mall Customer Segmentation Dashboard")
    st.markdown("<div class='header-decor'></div>", unsafe_allow_html=True)
    st.markdown("This dashboard presents business segment definitions and targeting strategies derived from unsupervised customer learning models. Customers are segmented based on their demographics, annual earnings, and retail spending behaviors.")
    
    # KPI metrics row
    if raw_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Total Customers</div>
                <div class='metric-value'>{len(raw_df)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Average Customer Age</div>
                <div class='metric-value'>{raw_df['Age'].mean():.1f} yrs</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Avg. Annual Income</div>
                <div class='metric-value'>${raw_df['Annual Income (k$)'].mean():.1f}k</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Avg. Spending Score</div>
                <div class='metric-value'>{raw_df['Spending Score (1-100)'].mean():.1f}/100</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 👥 Identified Business Segments")
    
    # Display Customer Segment cards in two columns
    col_left, col_right = st.columns(2)
    
    for idx, (c_id, c_data) in enumerate(CLUSTER_INFO.items()):
        # Distribute cards alternatingly
        target_col = col_left if idx % 2 == 0 else col_right
        with target_col:
            st.markdown(f"""
<div class='cluster-card' style='border-left-color: {c_data['color']}'>
    <div class='cluster-title'>Cluster {c_id}: {c_data['name']}</div>
    <div class='cluster-tag' style='color: {c_data['color']}; border-color: {c_data['color']}'>Profile Overview</div>
    <p><strong>Description:</strong> {c_data['description']}</p>
    <div class='cluster-stat'>• <strong>Gender composition:</strong> {c_data['gender']}</div>
    <div class='cluster-stat'>• <strong>Age:</strong> {c_data['age']}</div>
    <div class='cluster-stat'>• <strong>Annual Income:</strong> {c_data['income']}</div>
    <div class='cluster-stat'>• <strong>Spending Score:</strong> {c_data['spending']}</div>
    <div class='cluster-recommendation'>
        <strong>🎯 Business Strategy:</strong><br>
        {c_data['strategy']}
    </div>
</div>
""", unsafe_allow_html=True)

# ================= PAGE 2: EXPLORATORY DATA ANALYSIS =================
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Customer Behavior Data Explorer")
    st.markdown("<div class='header-decor'></div>", unsafe_allow_html=True)
    st.markdown("Interactively explore demographics and behavior distributions of different clusters.")
    
    if raw_df is None:
        st.warning("No data found to show EDA.")
    else:
        tab1, tab2, tab3 = st.tabs(["✨ Inter-Cluster Scatter Visualizer", "📊 Variable Distributions", "🍰 Segment Breakdown"])
        
        with tab1:
            st.markdown("### Customer Clusters in 2D & 3D space")
            # 2D Scatter plot
            fig_2d = px.scatter(
                raw_df,
                x="Annual Income (k$)",
                y="Spending Score (1-100)",
                color="Cluster Name",
                size="Age",
                hover_data=["Gender", "Age"],
                title="2D Cluster Scatter: Annual Income vs. Spending Score (Bubble size represents Customer Age)",
                color_discrete_map={info['name']: info['color'] for info in CLUSTER_INFO.values()},
                template="plotly_dark"
            )
            fig_2d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title="Customer Segment"
            )
            st.plotly_chart(fig_2d, use_container_width=True, theme=None)
            
            st.markdown("---")
            
            # 3D Scatter plot
            fig_3d = px.scatter_3d(
                raw_df,
                x="Age",
                y="Annual Income (k$)",
                z="Spending Score (1-100)",
                color="Cluster Name",
                hover_data=["Gender"],
                title="3D Cluster Scatter: Age vs. Income vs. Spending Score",
                color_discrete_map={info['name']: info['color'] for info in CLUSTER_INFO.values()},
                template="plotly_dark"
            )
            fig_3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title="Customer Segment"
            )
            st.plotly_chart(fig_3d, use_container_width=True, theme=None)
            
        with tab2:
            st.markdown("### Demographic Feature Distributions")
            feature = st.selectbox("Select variable to show distribution:", ["Age", "Annual Income (k$)", "Spending Score (1-100)"])
            
            fig_dist = px.histogram(
                raw_df,
                x=feature,
                color="Cluster Name",
                marginal="box",
                barmode="overlay",
                title=f"Distribution of {feature} by Cluster",
                color_discrete_map={info['name']: info['color'] for info in CLUSTER_INFO.values()},
                template="plotly_dark"
            )
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title="Customer Segment"
            )
            st.plotly_chart(fig_dist, use_container_width=True, theme=None)
            
        with tab3:
            st.markdown("### Proportion Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Segment Pie Chart
                fig_pie = px.pie(
                    raw_df,
                    names="Cluster Name",
                    title="Segment Proportions in Overall Base",
                    color="Cluster Name",
                    color_discrete_map={info['name']: info['color'] for info in CLUSTER_INFO.values()},
                    hole=0.4,
                    template="plotly_dark"
                )
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True, theme=None)
                
            with col2:
                # Gender count inside clusters
                fig_bar = px.histogram(
                    raw_df,
                    x="Cluster Name",
                    color="Gender",
                    barmode="group",
                    title="Gender Count Across Segment Clusters",
                    color_discrete_sequence=["#00ffd0", "#ff007f"],
                    template="plotly_dark"
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Segment Cluster"
                )
                st.plotly_chart(fig_bar, use_container_width=True, theme=None)

# ================= PAGE 3: SINGLE PREDICTOR =================
elif page == "🎯 Single Predictor":
    st.title("🎯 Real-Time Customer Segment Predictor")
    st.markdown("<div class='header-decor'></div>", unsafe_allow_html=True)
    st.markdown("Use this predictor tool to classify new customers into one of the 5 segments and view the corresponding targeting strategy.")
    
    if scaler is None or predictor_knn is None:
        st.warning("Model and Preprocessor components are missing. Run train_model.py first.")
    else:
        st.markdown("### Enter Customer Profile Details")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age (years)", 18, 75, 30)
        with col2:
            income = st.slider("Annual Income ($k)", 10, 150, 50)
            spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prediction triggers
        if st.button("Predict Target Segment"):
            # Map input to numeric model format
            gender_num = 1 if gender == "Male" else 0
            
            # Build input array
            # The columns MUST match training: Gender, Age, Annual Income (k$), Spending Score (1-100)
            features = np.array([[gender_num, age, income, spending_score]])
            
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Predict cluster
            pred_cluster = int(predictor_knn.predict(scaled_features)[0])
            cluster_details = CLUSTER_INFO[pred_cluster]
            
            # Output result
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            # Display Prediction Result Card
            st.markdown(f"""
<div class='cluster-card' style='border-left-color: {cluster_details['color']}; background-color: #1a1e2d;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div class='cluster-title' style='color: {cluster_details['color']}; margin-bottom: 0;'>
            Predicted Target: {cluster_details['name']}
        </div>
        <span class='cluster-tag' style='margin-bottom: 0; background-color: #2d3b4e; border-color: {cluster_details['color']}; color: {cluster_details['color']}'>
            Cluster #{pred_cluster}
        </span>
    </div>
    <hr style='border-color: #3b4257; margin-top: 15px; margin-bottom: 15px;'>
    <p><strong>Segment Profile:</strong> {cluster_details['description']}</p>
    <div class='cluster-stat'>• <strong>Ideal Gender Demographics:</strong> {cluster_details['gender']}</div>
    <div class='cluster-stat'>• <strong>Typical Age Range:</strong> {cluster_details['age']}</div>
    <div class='cluster-stat'>• <strong>Typical Annual Earnings:</strong> {cluster_details['income']}</div>
    <div class='cluster-stat'>• <strong>Typical Retail Spending Score:</strong> {cluster_details['spending']}</div>
    <div class='cluster-recommendation' style='background-color: rgba(0, 255, 208, 0.05); border: 1px solid {cluster_details['color']};'>
        <strong style='color: {cluster_details['color']};'>🎯 Tailored Marketing Strategy:</strong><br>
        {cluster_details['strategy']}
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Gauge charts for inputs relative to cluster defaults
            st.markdown("#### Input comparison vs Average of the Segment")
            g_cols = st.columns(3)
            
            # Extract cluster averages from CLUSTER_INFO string (using regex/parsing would be hard, so just map variables)
            # Or get averages directly from raw_df
            cluster_means = raw_df[raw_df['Cluster'] == pred_cluster].mean(numeric_only=True)
            
            with g_cols[0]:
                fig_age = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = age,
                    title = {'text': "Customer Age"},
                    gauge = {
                        'axis': {'range': [18, 75]},
                        'bar': {'color': cluster_details['color']},
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': cluster_means['Age']
                        }
                    }
                ))
                fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"}, height=220)
                st.plotly_chart(fig_age, use_container_width=True, theme=None)
                st.caption(f"White line shows cluster average: **{cluster_means['Age']:.1f} yrs**")
                
            with g_cols[1]:
                fig_inc = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = income,
                    title = {'text': "Annual Income ($k)"},
                    gauge = {
                        'axis': {'range': [10, 150]},
                        'bar': {'color': cluster_details['color']},
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': cluster_means['Annual Income (k$)']
                        }
                    }
                ))
                fig_inc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"}, height=220)
                st.plotly_chart(fig_inc, use_container_width=True, theme=None)
                st.caption(f"White line shows cluster average: **${cluster_means['Annual Income (k$)']:.1f}k**")
                
            with g_cols[2]:
                fig_spend = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = spending_score,
                    title = {'text': "Spending Score"},
                    gauge = {
                        'axis': {'range': [1, 100]},
                        'bar': {'color': cluster_details['color']},
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': cluster_means['Spending Score (1-100)']
                        }
                    }
                ))
                fig_spend.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"}, height=220)
                st.plotly_chart(fig_spend, use_container_width=True, theme=None)
                st.caption(f"White line shows cluster average: **{cluster_means['Spending Score (1-100)']:.1f}**")

# ================= PAGE 4: BATCH SEGMENTATION =================
elif page == "📁 Batch Segmentation":
    st.title("📁 Batch Customer Segmenter")
    st.markdown("<div class='header-decor'></div>", unsafe_allow_html=True)
    st.markdown("Upload a CSV file containing lists of customer demographic records to segment them all at once.")
    
    if scaler is None or predictor_knn is None:
        st.warning("Model and Preprocessor components are missing. Run train_model.py first.")
    else:
        st.markdown("### 1. Upload CSV Data File")
        uploaded_file = st.file_uploader("Upload CSV file here", type=["csv"])
        
        # Sample schema help
        with st.expander("Expected CSV Schema Structure"):
            st.markdown("Your CSV file must contain the following columns:")
            st.code("Gender,Age,Annual Income (k$),Spending Score (1-100)\nFemale,23,16,77\nMale,19,15,39\n...")
            st.markdown("*Note: Column headers are case-insensitive. Spaces and parentheses must be exactly matched, but the app will try to clean column names automatically.*")
            
        if uploaded_file is not None:
            try:
                # Load CSV
                batch_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Standardize columns to match required names
                col_mappings = {}
                for col in batch_df.columns:
                    col_lower = col.strip().lower()
                    if 'gender' in col_lower:
                        col_mappings[col] = 'Gender'
                    elif 'age' in col_lower:
                        col_mappings[col] = 'Age'
                    elif 'annual income' in col_lower or 'income' in col_lower:
                        col_mappings[col] = 'Annual Income (k$)'
                    elif 'spending score' in col_lower or 'spending' in col_lower:
                        col_mappings[col] = 'Spending Score (1-100)'
                        
                batch_df_cleaned = batch_df.rename(columns=col_mappings)
                
                # Verify required columns exist
                required_cols = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
                missing_cols = [col for col in required_cols if col not in batch_df_cleaned.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns in CSV: {missing_cols}. Please structure your file according to the schema template.")
                else:
                    st.markdown("### 2. Segmenting Data...")
                    
                    # Prepare data for model
                    input_df = batch_df_cleaned[required_cols].copy()
                    
                    # Map Gender
                    input_df['Gender'] = input_df['Gender'].astype(str).str.strip().str.capitalize()
                    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
                    
                    # Check for nulls in mapping (e.g. invalid string like 'other')
                    if input_df['Gender'].isnull().any():
                        st.warning("Some 'Gender' values could not be parsed. Defaulting unparsed records to Female (0). Ensure inputs are strictly 'Male' or 'Female'.")
                        input_df['Gender'] = input_df['Gender'].fillna(0)
                        
                    # Scale features
                    scaled_batch = scaler.transform(input_df)
                    
                    # Predict clusters
                    predictions = predictor_knn.predict(scaled_batch)
                    
                    # Add results back to DataFrame
                    batch_df['Cluster'] = predictions
                    batch_df['Cluster Name'] = batch_df['Cluster'].map(lambda x: CLUSTER_INFO[x]['name'])
                    
                    st.markdown("### 3. Segmentation Results & Analytics")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        # KPI cards for batch
                        st.metric("Total Records Uploaded", len(batch_df))
                        
                        # Cluster representation pie chart
                        fig_batch_pie = px.pie(
                            batch_df,
                            names="Cluster Name",
                            title="Batch Segment Distribution",
                            color="Cluster Name",
                            color_discrete_map={info['name']: info['color'] for info in CLUSTER_INFO.values()},
                            hole=0.3,
                            template="plotly_dark"
                        )
                        fig_batch_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_batch_pie, use_container_width=True, theme=None)
                        
                    with col2:
                        # Dataframe Preview
                        st.markdown("#### Segmented Customer Preview")
                        st.dataframe(batch_df.head(100), use_container_width=True)
                        
                        # Export / Download options
                        # Convert df to csv bytes
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Download Segmented Data as CSV",
                            data=csv_data,
                            file_name="segmented_customers.csv",
                            mime="text/csv"
                        )
                        
                    # Detailed batch profile breakdown
                    st.markdown("---")
                    st.markdown("#### Batch Summary Profiles")
                    
                    num_clusters = len(CLUSTER_INFO)
                    summary_cols = st.columns(num_clusters)
                    for idx, c_id in enumerate(sorted(CLUSTER_INFO.keys())):
                        c_subset = batch_df[batch_df['Cluster'] == c_id]
                        with summary_cols[idx]:
                            st.markdown(f"""
<div style='background-color: #171b26; border-top: 3px solid {CLUSTER_INFO[c_id]['color']}; padding: 15px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); height: 100%; text-align: center;'>
    <div style='font-size: 13px; font-weight: 700; color: #8c96a8;'>{CLUSTER_INFO[c_id]['name']}</div>
    <div style='font-size: 32px; font-weight: 800; color: {CLUSTER_INFO[c_id]['color']}; margin-top: 5px;'>{len(c_subset)}</div>
    <div style='font-size: 12px; color: #d1d5db; margin-top: 2px;'>records ({len(c_subset)/len(batch_df)*100:.1f}%)</div>
</div>
""", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred while processing the CSV file: {e}")
