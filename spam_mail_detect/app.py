import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#loaad the message spam or ham
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
#print(data.head())

#Convert labels into numbers
# 'ham' means not-spam (0), 'spam' means spam (1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

#print("\nNumber of spam and ham messages:")
#print(data['label'].value_counts())

#Split data into train and test sets
X = data['message']
y = data['label']

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("\n✅ Data split complete!")
#print("Training samples:", len(X_train))
#print("Testing samples:", len(X_test))


#Step 5: Converting text data into numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# Learn vocabulary and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
# Only transform test data (don’t fit again!)
X_test_tfidf = vectorizer.transform(X_test)

#print("\n✅ Text has been converted into numeric features (TF-IDF).")

#Train a simple classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

#print("\n✅ Model training complete!")

#Make predictions on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
#print("\n📊 Model Evaluation Results:")
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with your own messages
sample_messages = ["myntra biggest festive sale is live "]

# Convert to TF-IDF and predict
sample_features = vectorizer.transform(sample_messages)
predictions = model.predict(sample_features)

print("\n🔍 Sample Predictions:")
for msg, label in zip(sample_messages, predictions):
    print(f"Message: {msg}")
    print("Prediction:", "SPAM" if label == 1 else "HAM")
    