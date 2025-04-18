import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

nltk.download('vader_lexicon', quiet=True)  # Add quiet=True to suppress download messages

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("Survey Verbatims.csv", parse_dates=["EscalationDate"])

df = load_data()

st.title("\U0001F4CA Escalation Sentiment Dashboard")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(df)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Copy original DataFrame for processing
df_sentiment = df.copy()
df_sentiment['Customer Feedback'] = df_sentiment['Customer Feedback'].astype(str).fillna("")

# Apply sentiment analysis
def get_sentiment(comment):
    score = sia.polarity_scores(comment)
    return "Positive" if score['compound'] > 0.05 else "Negative" if score['compound'] < -0.05 else "Neutral"

df_sentiment['Sentiment'] = df_sentiment['Customer Feedback'].apply(get_sentiment)
sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df_sentiment['SentimentNumeric'] = df_sentiment['Sentiment'].map(sentiment_map)

# Sidebar: Date range filter
st.sidebar.header("\U0001F4C5 Filter by Date")
min_date = df_sentiment['EscalationDate'].min()
max_date = df_sentiment['EscalationDate'].max()
start_date, end_date = st.sidebar.date_input("Select date range:", [min_date, max_date], min_value=min_date, max_value=max_date)

# Apply date filter
mask = (df_sentiment['EscalationDate'] >= pd.to_datetime(start_date)) & \
       (df_sentiment['EscalationDate'] <= pd.to_datetime(end_date))
filtered_df = df_sentiment[mask]

# Sentiment over time by service type
st.header("Sentiment Over Time by Service Type")
sentiment_by_service = (
    filtered_df.groupby(["EscalationDate", "ServiceType"])["SentimentNumeric"]
    .mean()
    .unstack()
)
st.line_chart(sentiment_by_service)

# Sentiment distribution
st.header("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="Sentiment", palette="Set2", ax=ax)
st.pyplot(fig)

# Average sentiment by category
st.header("Average Sentiment by Category")
sentiment_by_category = (
    filtered_df.groupby("CategoryName")["SentimentNumeric"]
    .mean()
    .sort_values()
)
st.bar_chart(sentiment_by_category)

# Average sentiment over time
st.header("Average Sentiment Over Time")
sentiment_by_date = filtered_df.groupby('EscalationDate')['SentimentNumeric'].mean()
fig, ax = plt.subplots()
sentiment_by_date.plot(ax=ax, title='Average Sentiment Over Time')
st.pyplot(fig)

# Average sentiment by service type
st.header("Average Sentiment by Service Type")
fig, ax = plt.subplots()
filtered_df.groupby('ServiceType')['SentimentNumeric'].mean().plot(kind='barh', title='Avg Sentiment by Service Type', ax=ax)
st.pyplot(fig)

# Escalation volume over time
st.header("Escalation Volume Over Time")
escalation_counts = filtered_df['EscalationDate'].value_counts().sort_index()
fig, ax = plt.subplots()
escalation_counts.plot(ax=ax, title='Escalation Volume Over Time')
ax.set_xlabel("Date")
ax.set_ylabel("Number of Escalations")
ax.grid(True)
st.pyplot(fig)


# Boxplot of sentiment by escalation reason
st.header("\U0001F4E6 Sentiment Distribution by Escalation Reason")
top_reasons = filtered_df['Reason for Escalation'].value_counts().head(10).index
box_df = filtered_df[filtered_df['Reason for Escalation'].isin(top_reasons)]
order = box_df.groupby("Reason for Escalation")["SentimentNumeric"].median().sort_values().index
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=box_df, x='SentimentNumeric', y='Reason for Escalation', order=order, palette='Set2', ax=ax)
ax.set_title('Sentiment by Escalation Reason (Top 10)', fontsize=14)
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Escalation Reason')
plt.tight_layout()
st.pyplot(fig)


# Word cloud of feedback
st.header("\U0001F4AC Word Cloud of Customer Feedback")
sentiment_option = st.selectbox("Filter feedback by sentiment:", ["All", "Positive", "Neutral", "Negative"])
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["customer", "feedback", "support", "service", "help", "issue"])
if sentiment_option == "All":
    wc_text = filtered_df['Customer Feedback'].dropna()
else:
    wc_text = filtered_df[filtered_df['Sentiment'] == sentiment_option]['Customer Feedback'].dropna()
text = ' '.join(wc_text.astype(str))
if not text.strip():
    st.warning("No feedback available for the selected sentiment.")
else:
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud of {sentiment_option} Feedback', fontsize=16)
    st.pyplot(fig)

df['Root cause (extracted)'] = df['Agent Comment'].str.extract(
    r'Root cause\s*-\s*(.*?)(?=\n-|$)',
    flags=re.IGNORECASE | re.DOTALL
)

# Streamlit app layout
st.title('Top Root Causes from Agent Comments')

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
df['Root cause (extracted)'].value_counts().head(10).plot(kind='barh', ax=ax)
ax.set_title('Top 10 Root Causes')
ax.set_xlabel('Count')

# Display the plot in Streamlit
st.pyplot(fig)
