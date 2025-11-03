import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('subjectivity')
nltk.download('movie_reviews') 

sia=SentimentIntensityAnalyzer()

while True:
    analyse_msg=input("Message : " )
    score=sia.polarity_scores(analyse_msg)
    compound=score['compound']
    if compound >0:
        print("positive comment")
    elif compound < -0.1:
        print('Negative Comment !!')
    else:
        print('Neutral comment') 