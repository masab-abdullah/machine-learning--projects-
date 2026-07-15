import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("emails.csv")

X = df["email"]

y = df["label"]

vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()


model.fit(X_vectorized, y)

user_email = input("Enter an email message: ")


user_email_vectorized = vectorizer.transform([user_email])


prediction = model.predict(user_email_vectorized)

# Display the result
if prediction[0] == "spam":
    print("Prediction: Spam Email")
else:
    print("Prediction: Not Spam Email")