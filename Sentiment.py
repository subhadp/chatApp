import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

column_names = ["polarity", "title", "text"]
data = pd.read_csv('DataSet/train.csv', header=None, names=column_names)

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

for index, row in data.head(10).iterrows():
    review = row['text']
    sentiment = sia.polarity_scores(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print("-" * 60)

#Sentiment: {'neg': 0.0, 'neu': 0.79, 'pos': 0.21, 'compound': 0.867}
#The sentiment analysis indicates that the text is 0% negative, 79% neutral, 21% positive,
# and has a strong positive compound score of 0.867, suggesting an overall positive sentiment.






