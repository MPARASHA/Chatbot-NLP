import random
import json

import torch

from NN import NN
from preprocess import bagOfWords, tokenize
import nltk

# Session Variables
name = ""

def updateName(response):
    if(name != "" and name is not None):
        return response.replace("<HUMAN>", name)
    else:
        return response

# Extracts the name from the input sentence using ne_chunk function of nltk library
def extract_entities(text):
    for sent in nltk.sent_tokenize(text):
         for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if (hasattr(chunk, 'label') and chunk[0][1] == 'NNP'):
                return (' '.join(c[0] for c in chunk.leaves()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "YoBot"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentenceI = input("You: ")
    if sentenceI == "quit":
        break

    sentence = tokenize(sentenceI)
    X = bagOfWords(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # if its a greeting response it will have user's name
    if(tag == "GreetingResponse" or tag == "CourtesyGreetingResponse"):
        name = extract_entities(sentenceI)


    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["intent"]:

                print(f"{bot_name}: {updateName(random.choice(intent['responses']))}")
    else:
        print(f"{bot_name}: I do not understand...")