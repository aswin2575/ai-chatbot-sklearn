import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1️⃣ Sample dataset of questions (inputs) and intents (labels)
data = [
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("How are you?", "greeting"),
    ("Bye", "goodbye"),
    ("See you later", "goodbye"),
    ("What is your name?", "name"),
    ("Who are you?", "name"),
    ("Tell me a joke", "joke"),
    ("Make me laugh", "joke"),
]

# 2️⃣ Responses for each intent
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
    "goodbye": ["Bye!", "See you later!", "Goodbye!"],
    "name": ["I am a chatbot.", "My name is Chatty.", "You can call me Chatbot!"],
    "joke": [
        "Why did the computer go to the doctor? It caught a virus!",
        "I would tell you a UDP joke, but you might not get it."
    ],
    "fallback": ["I’m not sure how to answer that. Can you ask something else?", 
                 "Sorry, I didn’t understand. Try asking differently."],
}

# 3️⃣ Preprocess the text (convert text to numbers)
questions = [q for q, intent in data]
labels = [intent for q, intent in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
y = labels

# 4️⃣ Train the classifier
model = LogisticRegression()
model.fit(X, y)

# 5️⃣ Chatbot function
def chatbot():
    print("Chatbot: Hi! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Bye!")
            break
        # Convert user input to vector
        user_vec = vectorizer.transform([user_input])
        try:
            # Predict intent
            intent = model.predict(user_vec)[0]
            # Check probability of prediction
            prob = max(model.predict_proba(user_vec)[0])
            if prob < 0.4:  # low confidence → fallback
                intent = "fallback"
        except NotFittedError:
            intent = "fallback"
        # Choose a random response for that intent
        response = random.choice(responses[intent])
        print("Chatbot:", response)

# 6️⃣ Run the chatbot
chatbot()
