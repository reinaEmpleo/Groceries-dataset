import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import mlxtend

# Debug: Check Versions
print(f"mlxtend version: {mlxtend.__version__}")
print(f"Seaborn Version: {sns.__version__}")

# Streamlit Config
st.set_page_config(page_title="Grocery Dataset Using Apriori Algorithm", page_icon="🛒")
st.title("🛒 Grocery Dataset Using Apriori Algorithm")

# Load Data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data/Groceries_dataset.csv", parse_dates=['Date'], dayfirst=True)
        return df
    except FileNotFoundError:
        st.error("Error: Dataset file not found.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# Data Preprocessing
groceriesDS_clean = df.copy()
groceriesDS_clean.set_index('Date', inplace=True)

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Visualization", 
    "🔍 Apriori Calculation", 
    "📈 Association Rules", 
    "🗂️ Dataset Overview"
])

# **Tab 1: Data Visualization**
with tab1:
    st.subheader("📊 Summary Statistics")
    total_items = len(groceriesDS_clean)
    total_days = len(np.unique(groceriesDS_clean.index.date))
    total_months = len(np.unique(groceriesDS_clean.index.month))
    average_items = total_items / total_days
    unique_items = groceriesDS_clean['itemDescription'].nunique()

    st.write(f"**Unique Items Sold:** {unique_items}")
    st.write(f"**Total Items Sold:** {total_items}")
    st.write(f"**Total Sales Days:** {total_days}")
    st.write(f"**Total Sales Months:** {total_months}")
    st.write(f"**Average Daily Sales:** {average_items:.2f}")

    # Visualization: Items Sold by Date
    st.subheader("📅 Items Sold by Date")
    fig, ax = plt.subplots(figsize=(12, 5))
    groceriesDS_clean.resample("D")['itemDescription'].count().plot(ax=ax, grid=True)
    ax.set_title("Total Number of Items Sold by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Number of Items Sold")
    st.pyplot(fig)

    # Visualization: Items Sold by Month
    st.subheader("📅 Items Sold by Month")
    fig, ax = plt.subplots(figsize=(12, 5))
    groceriesDS_clean.resample("M")['itemDescription'].count().plot(ax=ax, grid=True)
    ax.set_title("Total Number of Items Sold by Month")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Number of Items Sold")
    st.pyplot(fig)

    # Word Cloud Visualization
    st.subheader("☁️ Word Cloud of Items")
    wordcloud = WordCloud(
        background_color="white", width=1200, height=1200, max_words=150
    ).generate(' '.join(groceriesDS_clean['itemDescription'].tolist()))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Items Word Cloud", fontsize=20)
    st.pyplot(fig)

# **Tab 2: Apriori Calculation**
transactions = [group['itemDescription'].tolist() for _, group in groceriesDS_clean.groupby(['Member_number', 'Date'])]

if not transactions:
    st.error("No transactions found. Please check the dataset.")
    st.stop()

# Transaction Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate Frequent Itemsets
freq_items = pd.DataFrame()  # Initialize
try:
    freq_items = apriori(transaction_df, min_support=0.001, use_colnames=True)

    if freq_items.empty:
        st.error("No frequent itemsets found. Try adjusting the minimum support value.")
        st.stop()

    st.subheader("🔍 Frequent Itemsets Found")
    st.dataframe(freq_items.head(10))
except Exception as e:
    st.error(f"Error generating frequent itemsets: {str(e)}")
    st.stop()

# **Tab 3: Association Rules**
with tab3:
    st.subheader("📈 Association Rules")
    try:
        # Generate Association Rules
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.001)
        
        if rules.empty:
            st.error("No association rules found. Try adjusting the minimum threshold.")
            st.stop()

        # Display Top Rules
        st.write("**Top 10 Association Rules:**")
        st.dataframe(rules.head(10))

        # Support vs Confidence Scatter Plot
        st.subheader("📊 Support vs Confidence")
        fig = px.scatter(
            rules, x='support', y='confidence', 
            title='Support vs Confidence',
            labels={'support': 'Support', 'confidence': 'Confidence'},
            template="plotly_white"
        )
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Support vs Lift Scatter Plot
        st.subheader("📊 Support vs Lift")
        fig = px.scatter(
            rules, x='support', y='lift',
            title='Support vs Lift',
            labels={'support': 'Support', 'lift': 'Lift'},
            template="plotly_white"
        )
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Lift vs Confidence with Regression Line
        st.subheader("📈 Lift vs Confidence (Regression)")
        fit = np.polyfit(rules['lift'], rules['confidence'], 1)
        fit_fn = np.poly1d(fit)

        plt.figure(figsize=(10, 6))
        plt.plot(rules['lift'], rules['confidence'], 'yo', label="Data Points")
        plt.plot(rules['lift'], fit_fn(rules['lift']), '-', label="Fit Line")
        plt.xlabel('Lift')
        plt.ylabel('Confidence')
        plt.title('Lift vs Confidence')
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error generating association rules: {str(e)}")

# **Tab 4: Dataset Overview**
with tab4:
    st.header("Dataset Overview")
    st.write("""For our final project, we utilized the groceries dataset sourced from Kaggle. 
             The dataset has 38765 rows of the purchase orders of people from the grocery stores. 
             The dataset captures transactional data from a grocery store, where each row corresponds 
             to an individual purchase order. To uncover meaningful insights, we applied the Apriori algorithm and K-means clustering.""")
    
    st.markdown('#### **Why did we choose Apriori Algorithm and K-means Clustering?**')
    st.markdown('##### **Apriori Algorithm**')
    st.markdown('**Purpose:** The Apriori algorithm is used for frequent itemset mining and learning association rules. It identifies frequently purchased items and calculates metrics like support, confidence, and lift to derive patterns in customer behavior.')
    st.markdown("""
        -  Enables discovery of significant trends in purchase data.
        -  Provides actionable insights for product bundling, promotions, and inventory management.
        """)

    st.markdown('##### **K-means Clustering**')
    st.markdown('**Purpose:** K-means clustering groups transactions or customers based on purchasing patterns to identify meaningful clusters.')
    st.markdown("""
        -  Segments customers into groups with similar preferences (e.g., frequent shoppers, gourmet buyers).
        -  Enhances the effectiveness of Apriori by applying it to specific clusters, uncovering nuanced patterns.
        """)
    
    st.markdown('### **Research Questions**')
    st.markdown('(We can answer these questions after we get the results from the selected techniques.)')
    st.markdown("""
        - What are the most common product combinations purchased together?
        - How do purchase patterns vary by time of day or day of the week?
        """)
    
    st.markdown('##### This table shows the first 50 rows of data from the groceries dataset.')
    st.table(df.head(50))
