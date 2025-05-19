import nltk
from nltk.tokenize import word_tokenize
import random

nltk.download('punkt_tab')


knowledge_base = {
    "greetings": [
        "Hello! How are you doing today?",
        "Hi there! How can I assist you?",
        "Hey! What's up?",
        "Greetings! How can I help?"
    ],
    "farewells": [
        "Goodbye! Have a wonderful day!",
        "See you later! Take care!",
        "Bye! Hope to chat again soon.",
        "Take care! Talk to you later."
    ],
    "how_are_you": [
        "I'm just a friendly bot, feeling great! How about you?",
        "Doing well, thanks! How are you feeling?",
        "I'm good! What about you?",
        "All systems running smoothly! How's your day?"
    ],
    "thanks": [
        "You're welcome!",
        "No problem!",
        "Happy to help!",
        "Anytime!"
    ],
    "default": [
        "Sorry, I didn't catch that. Can you say it differently?",
        "I'm not sure I understand. Could you rephrase?",
        "Hmm, that's interesting. Tell me more!",
        "Let's talk about something else. What do you like?"
    ]
}

def respond(user_input):
    tokens = set(word_tokenize(user_input.lower()))

    if tokens.intersection({"hello", "hi", "hey", "greetings"}):
        return random.choice(knowledge_base["greetings"])
    elif tokens.intersection({"bye", "goodbye", "exit", "quit"}):
        return random.choice(knowledge_base["farewells"])
    elif tokens.intersection({"how", "are", "you"}):
        return random.choice(knowledge_base["how_are_you"])
    elif tokens.intersection({"thanks", "thank", "thankyou", "thank you"}):
        return random.choice(knowledge_base["thanks"])
    else:
        return random.choice(knowledge_base["default"])

print("NLTK Chatbot running! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye! Have a great day!")
        break
    print("Bot:", respond(user_input))



    
