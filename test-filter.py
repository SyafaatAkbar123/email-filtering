# 1. Load Dataset
import pandas as pd
dataFrame = pd.read_csv("dataset/spam_nonspam_data.csv")

# 2. Preprocessing
dataFrame['text'] = dataFrame['text'].str.lower() # Case folding convert text to lowercase

import nltk
nltk.download('punkt')
dataFrame['text'] = dataFrame['text'].apply(nltk.word_tokenize) # Word Tokenization

nltk.download('stopwords')
stopWord = nltk.corpus.stopwords.words('english') # Stopwords
dataFrame['text'] = dataFrame['text'].apply(lambda words: [word for word in words if word not in stopWord]) # Remove Stopwords 

stemming = nltk.stem.PorterStemmer()
dataFrame['text'] = dataFrame['text'].apply(lambda words: [stemming.stem(word) for word in words]) # Stemming

dataFrame['text'] = dataFrame['text'].apply(lambda words: ' '.join(words)) # Join words

dataFrame['text'] = dataFrame['text'].str.replace(r'[^a-z\s]', '', regex=True) # Remove symbols and number
print(dataFrame)

# 3. Split Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataFrame['text'], dataFrame['label'], test_size=0.2, random_state=42)

print(f"Data Train: {len(X_train)}")
print(f"Data Test: {len(X_test)}\n")

# 4. Vectorization 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train) # Fit and transform
X_test_vect = vectorizer.transform(X_test) # Only transform

# 5. Model Training
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vect, y_train) # "y_train" not prosecced bcs it is no need to be vectorized

# 6. Model Prediction
y_pred = model.predict(X_test_vect)

# 7. Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Save Model
import joblib
joblib.dump(model, "model/spam-nonspam-model.pkl")
joblib.dump(vectorizer, "model/vector.pkl")
