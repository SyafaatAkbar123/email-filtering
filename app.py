
from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel

class Contact(BaseModel):
  message: str

@app.post("/")
async def root(contact: Contact):
  
  # --Preprocessing--
  # 1. Lower
  message = contact.message.lower()

  # 2. Word Tokenization  
  import nltk 
  message = nltk.tokenize.word_tokenize(message)
  
  # 3. Stop Words
  stopWords = nltk.corpus.stopwords.words("english")
  message = [word for word in message if word not in stopWords]
  
  # 4. Stemming
  stemming = nltk.stem.PorterStemmer()
  message = [stemming.stem(word) for word in message]
  
  # 5. Rejoin Words 
  message = ' '.join(message)
  
  # 6. Remove Symbols
  import re
  message = re.sub(r'[^a-z\s]', '', message)
  
  # 7. Load Model and Vector
  from joblib import load
  vector = load("model/vector.pkl")
  model = load("model/spam-nonspam-model.pkl")
  
  # 8. Predict
  message = vector.transform([message])
  message = model.predict(message)
  
  return {"category": message[0]}
