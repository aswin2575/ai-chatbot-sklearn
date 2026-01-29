import random
import tkinter as tk  # 🔹 CHANGED: imported Tkinter
from tkinter import scrolledtext  # 🔹 CHANGED: for scrollable chat area
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
data += [
    ("bye", "goodbye"),
    ("see you", "goodbye"),
    ("catch you later", "goodbye"),
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
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
questions = [q.lower() for q, intent in data] 
labels = [intent for q, intent in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
y = labels

# 4️⃣ Train the classifier
model = LogisticRegression()
model.fit(X, y)

# 5️⃣ Chatbot function
def get_response(user_input):  # 🔹 CHANGED: wrapped previous logic in function
    user_input = user_input.lower()
    user_vec = vectorizer.transform([user_input])
    try:
        intent = model.predict(user_vec)[0]
        prob = max(model.predict_proba(user_vec)[0])
        print(prob)
        if prob < 0.2:  # low confidence → fallback
            intent = "fallback"  # 🔹 CHANGED: use fallback in function
    except NotFittedError:
        intent = "fallback"  # 🔹 CHANGED: handle error
    return random.choice(responses[intent])  # 🔹 CHANGED: return response

def send_message(event=None):  # 🔹 CHANGED: called when Send button clicked
    user_input = user_entry.get()
    if user_input.strip() == "":
        return
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_input + "\n","user")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)

    response = get_response(user_input)
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "Chatbot: " + response + "\n\n", "bot")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)
    user_entry.delete(0, tk.END)

window = tk.Tk()  # 🔹 CHANGED
window.title("AI Chatbot")  # 🔹 CHANGED
window.geometry("500x500")  # 🔹 CHANGED

chat_area = scrolledtext.ScrolledText(window, state=tk.DISABLED, wrap=tk.WORD)  # 🔹 CHANGED
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)  # 🔹 CHANGED

chat_area.tag_config("user", foreground="blue")     # user messages in blue
chat_area.tag_config("bot", foreground="green")  

user_entry = tk.Entry(window, width=50)  # 🔹 CHANGED
user_entry.pack(padx=10, pady=10, side=tk.LEFT, expand=True, fill=tk.X)  # 🔹 CHANGED
user_entry.bind("<Return>", send_message)  # 🔹 CHANGED: press Enter to send

send_button = tk.Button(window, text="Send", width=10, command=send_message)  # 🔹 CHANGED
send_button.pack(padx=10, pady=10, side=tk.RIGHT)  # 🔹 CHANGED

# Start GUI loop  # 🔹 CHANGED
window.mainloop()  # 🔹 CHANGED

# 6️⃣ Run the chatbot
# chatbot()
