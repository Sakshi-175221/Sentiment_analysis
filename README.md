# Sentiment_analysis
Sentiment Analysis is the process determining the emotional tone of text and classifying it as positive, negative or neutral. The purpose of this analysis is to determine the review as positive or negative based Restaurant review dataset. The dataset contains two columns 'Review' and 'Liked'. Liked variable takes two values 0 and 1. The motive of this project is to classify the review into positive or negative. It is a balanced data. Models Used:
Stopwords
library is imported to eliminate words that are so widely used that they carry very little useful information.
Stemmers
are the algorithms used to reduce different forms of a word to a base form. Essentially, they do this by removing specific character strings from the end of word tokens.
CountVectorizer
is used to transform the words into vectors on the basis of count.
Multinomial Naive Bayes
is suitable for classification with discrete features (e.g., word counts for text classification)
