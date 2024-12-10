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
st.set_page_config(page_title="DataHarvest", page_icon="🛒")
st.markdown("<h1 style='text-align: center;'>🛒 DataHarvest Grocery Dataset Using Apriori Algorithm 🛒</h1>", unsafe_allow_html=True)

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
tab1, tab2, tab3, tab4, tab5= st.tabs([
    "🗂️ Dataset Overview",
    "📊 Data Visualization",
    "🔍 Apriori Calculation", 
    "📈 Association Rules",
    "🚀 Conclusion"
])

# **Tab 1: Data Overview**
with tab1:
    st.header("Dataset Overview")
    st.write("""For our final project, we utilized the groceries dataset sourced from Kaggle. 
                The dataset has 38765 rows of the purchase orders of people from the grocery stores. 
                The dataset captures transactional data from a grocery store, where each row corresponds 
                to an individual purchase order. To uncover meaningful insights, we applied the Apriori algorithm and K-means clustering.""")
        
    st.markdown('#### 🤔 **Why did we choose Apriori Algorithm and K-means Clustering?**')
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
        
    st.divider()
    st.markdown('### **Research Questions**')
    st.markdown('(We can answer these questions after we get the results from the selected techniques.)')
    st.markdown("""
            - What are the most common product combinations purchased together?
            - How do purchase patterns vary by time of day or day of the week?
            """)
        
    st.markdown('##### This table shows the first 50 rows of data from the groceries dataset.')
    st.table(df.head(50))

with tab2:
    # Data Visualization
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

    st.markdown("Overall Sales Performance")
    st.markdown("""
    The store sold a total of **38,765 items** over **728 days**, averaging **53.25 items sold per day**. 
    This indicates a consistent level of sales throughout the year.
    """)

    st.markdown("Product Diversity")
    st.markdown("""
    The store offered **167 unique items**, suggesting a diverse product range. 
    This variety likely contributes to attracting a wide range of customers.
    """)

    st.markdown("Seasonal Trends")
    st.markdown("""
    While not explicitly shown in the summary statistics, the previous analysis of items sold by date and month revealed **seasonal trends**. 
    This information, combined with the consistent daily sales, suggests that the store may adjust its inventory and marketing strategies to cater to seasonal demand.
    """)

    # Visualization: Items Sold by Date
    st.divider()
    st.subheader("📅 Items Sold by Date")
    # Resample the data to daily counts
    daily_counts = groceriesDS_clean.resample("D")['itemDescription'].count().reset_index()
    
    # Create an interactive Altair line plot
    chart = alt.Chart(daily_counts).mark_line(
    color='darkblue',  # Set line color to dark blue
    interpolate='basis',  # Make line smooth with rounded ends
).encode(
    x=alt.X('Date:T', title='Date', axis=alt.Axis(titleColor='black', labelColor='black')),  # Set axis labels and title to black
    y=alt.Y('itemDescription:Q', title='Total Number of Items Sold', axis=alt.Axis(titleColor='black', labelColor='black')),  # Set axis labels and title to black
    tooltip=['Date:T', 'itemDescription:Q']
).properties(
    title="Total Number of Items Sold by Date",
    width=700,
    height=400,
    background='rgba(255, 255, 255, 0.9)',  # Set background transparency to 50%
).configure_title(
    fontSize=20,
    font='Serif',
    color='black',  # Set title color to black
    anchor='middle',  # Center the title
).interactive() # Enable zooming and panning
    
    # Display the chart
    st.altair_chart(chart)

    st.markdown("""
    This line chart visualizes the **daily fluctuations in the number of items sold** over a one-year period, spanning from **January 2014 to December 2015**. 
    The y-axis represents the **total number of items sold on a given day**, while the x-axis shows the **corresponding date**.
    """)

    st.markdown("Key Observations")
    st.markdown("""
    - **Seasonal Patterns:** The chart reveals distinct seasonal patterns in sales. There are noticeable peaks in sales during certain periods, likely corresponding to holidays or special occasions. Conversely, there are also periods of lower sales, which might be influenced by factors like weather or economic conditions.
    - **Daily Fluctuations:** The line chart highlights the daily variability in sales. There are days with significantly higher or lower sales compared to the average.
    - **Overall Trend:** While there are fluctuations, the overall trend suggests a relatively consistent level of sales throughout the year.
    """)

    st.divider()
    # Visualization: Items Sold by Month
    st.subheader("📅 Items Sold by Month")
    fig, ax = plt.subplots(figsize=(12, 5))
    groceriesDS_clean.resample("M")['itemDescription'].count().plot(ax=ax, grid=True)
    ax.set_title("Total Number of Items Sold by Month")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Number of Items Sold")
    st.pyplot(fig)

    st.markdown("""
    This line chart illustrates the **monthly sales trends for the years 2014 and 2015**. 
    The y-axis represents the **total number of items sold**, while the x-axis shows the **corresponding month**.
    """)
    st.markdown("""
    - **Seasonal Patterns:** The chart reveals distinct seasonal patterns in sales. There are periods of higher sales, particularly in the latter half of the year, likely driven by holiday seasons and increased consumer spending. Conversely, there are months with lower sales, which might be influenced by factors like weather or economic conditions.
    - **Monthly Fluctuations:** The line chart highlights the month-to-month variability in sales. There are months with significantly higher or lower sales compared to the average.
    - **Overall Trend:** While there are fluctuations, the overall trend suggests a slight upward trajectory in sales over the two-year period.
    """)

    st.divider()
    # Word Cloud Visualization
    wordcloud = WordCloud(
    background_color="white", width=1500, height=800, max_words=121
    ).generate(' '.join(groceriesDS_clean['itemDescription'].tolist()))
    # Create a Plotly figure using the wordcloud image
    wordcloud_img = wordcloud.to_image()
    fig = px.imshow(wordcloud_img)

    # Make the image interactive by disabling axes
    fig.update_layout(
    title="Items Word Cloud",
    xaxis_visible=False,
    yaxis_visible=False,
    autosize=True,
    width=1000,  # Adjust width to make it larger
    height=670, 
)
    # Show the interactive word cloud in the streamlit app
    st.subheader("☁️ Interactive Word Cloud of Items")
    st.plotly_chart(fig)
    st.markdown("Dominant Product Categories")
    st.markdown("""
    - **Dairy Products🥛🧀:** Words like *milk*, *cheese*, *yogurt*, and *butter* are prominently displayed, indicating a strong emphasis on dairy products.
    - **Fresh Produce🥦🍎🍊:** Terms like *vegetables*, *fruit*, and *tropical fruit* highlight the importance of fresh produce offerings.
    - **Baked Goods🍞🥐🥖:** Words such as *bread*, *rolls*, *buns*, and *pastry* suggest a focus on bakery items.
    - **Meat and Poultry🍖🍗🥩:** *Meat*, *sausage*, *beef*, and *chicken* indicate a variety of meat and poultry options.
    """)

# **Tab 3: Apriori**
with tab3:
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
        st.markdown("Popular Items 🍺🥩")
        st.markdown("""
        - **Bottled beer (4.53%)** and **beef (3.4%)** are the most frequently purchased items, indicating high consumer demand for these staples.
        - **UHT-milk (2.14%)** and **berries (2.18%)** also show consistent popularity, reflecting their role as household essentials or preferred consumables.
        """)

        st.markdown("Less Frequent Items 🚽🧴🍜")
        st.markdown("""
        - Items like **bathroom cleaner (0.11%)**, **abrasive cleaner (0.15%)**, and **Instant food products (0.4%)** are purchased less often, potentially due to lower demand or niche use cases.
        """)

        st.markdown("Key Insights 💡")
        st.markdown("""
        - Essentials like milk, meat, and beverages dominate purchases, aligning with daily consumption patterns.
        - Less frequent items may benefit from promotions or bundling strategies to increase visibility and sales.
        """)

        st.markdown("Opportunities 📈🎯")
        st.markdown("""
        - Promote underperforming items through discounts or product bundles with popular items.
        - Use this data to identify combinations of frequently bought items for cross-selling opportunities.
        """)
    except Exception as e:
        st.error(f"Error generating frequent itemsets: {str(e)}")
        st.stop()

# **Tab 4:Association Rules**
with tab4:
        st.subheader("📈 Association Rules")
        try:
            # Generate Association Rules
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.001)
            rule= rules[['antecedents','consequents','support','confidence','lift']]
            
            if rules.empty:
                st.error("No association rules found. Try adjusting the minimum threshold.")
                st.stop()

            # Display Top Rules
            st.write("**Top 10 Association Rules:**")
            st.dataframe(rules.head(10))
            st.markdown("""
            📦 Common Products:
            The rule indicates that **UHT-milk** is frequently purchased alongside other products such as **bottled water**, **other vegetables**, **rolls/buns**, **root vegetables**, and **sausage**. This suggests that **UHT-milk** is a staple item that often accompanies a variety of complementary products.

            🔗 Strength of Association:
            The **lift** values in the table are around **0.76 to 0.82**, which suggests a **moderate positive correlation** between **UHT-milk** and its associated products. This implies that these items tend to be bought together, but the association is not extremely strong.

            💡 Confidence Levels:
            The **confidence** values range from **0.05 to 0.1**, indicating that while there is some likelihood that customers purchasing one item will also buy the associated item, the probability is relatively low. For example, if a customer buys **UHT-milk**, there’s about a **5-10%** chance they will also purchase the other associated product.

            🤝 Potential for Cross-Selling:
            Given these relationships, marketing strategies such as **bundling** **UHT-milk** with the associated products (e.g., **other vegetables** or **bottled water**) could be effective in increasing sales. Offering discounts on complementary items could encourage customers to purchase them together.

            ✅ Conclusion:
            While **UHT-milk** shows moderate associations with other products, the **confidence** and **lift** values suggest there is room for improvement in making these associations more robust. Focused marketing and **product bundling** strategies could be effective in capitalizing on these moderate associations.
            """)

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
            st.markdown("""The Support vs. Confidence plot reveals the relationship between the frequency (support) and strength (confidence) of association rules. Most rules have low support and confidence, indicating weak associations. However, a few rules exhibit high confidence, suggesting strong relationships between items. These high-confidence rules can be valuable for cross-selling and upselling strategies.""")

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
            st.markdown("""The Support vs. Lift plot visualizes the relationship between the frequency (support) and strength (lift) of association rules. It shows a cluster of points towards the lower left corner, indicating that most rules have low support and lift. However, there are a few points scattered towards the upper right corner, representing rules with higher support and lift. These rules suggest strong associations between items, making them potentially valuable for cross-selling and upselling strategies.""")

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
            st.markdown("""The Lift vs. Confidence (Regression) plot shows the relationship between the strength (lift) and the confidence of association rules. The regression line suggests a positive correlation, indicating that as the lift increases, the confidence also tends to increase. However, there is significant scatter in the data points, suggesting that while there is a general trend, individual rules can deviate from this pattern.""")

        except Exception as e:
            st.error(f"Error generating association rules: {str(e)}")

with tab5: 
    st.markdown("**Apriori Algorithm: Unlocking Grocery Insights** 🛒💡\n\n"
            "The Apriori algorithm, a powerful data mining tool, has proven invaluable in analyzing our grocery dataset. By identifying frequent itemsets and strong associations, we've gained deeper insights into customer behavior and preferences.\n\n"
            "**Key Findings:**\n\n"
            "* **Frequent Itemsets:** Certain products are frequently purchased together, suggesting strong associations. For example, bread and butter often go hand-in-hand. 🍞🧈 By strategically placing these items together, we can boost sales.\n"
            "* **Strong Associations:** High-confidence and high-lift association rules highlight significant relationships between products. Offering discounts on complementary items can incentivize additional purchases. 🛍️💰\n"
            "* **Seasonal Patterns:** Our analysis reveals seasonal trends in purchasing behavior. By understanding these patterns, we can optimize inventory management and tailor marketing campaigns to specific seasons. 📅📈\n"
            "* **Customer Segmentation:** While not explicitly explored here, the dataset holds the potential to identify distinct customer segments based on their purchasing habits. This segmentation can inform targeted marketing strategies and personalized product recommendations. 🎯\n\n"
            "By leveraging these insights, we can optimize product placement, enhance cross-selling and upselling strategies, and implement targeted marketing campaigns. As we continue to explore the dataset, we'll uncover even more valuable insights to drive business growth. 📈")

