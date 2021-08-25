import json
from preprocess import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ChatDataset import ChatDataset
from NN import NN
import torch
import torch.nn as nn


with open('intent.json', 'r') as f:
    data = json.load(f)

words = []
intents = []
xy = []

# PREPROCESS (Tokenize + Stemming + Stop Word Removal)
for intent in data['intents']:
    tag = intent['intent']
    intents.append(tag)
    for text in intent['text']:
        w = tokenize(text)
        words.extend(w)
        xy.append((w, tag))

stop_words = ['?', ",", ".", "!", "a", "the"]

words = [stem(w) for w in words if w not in stop_words]
words = sorted(set(words))
intents = sorted(set(intents))

x_train = []
y_train = []

for (s, i) in xy:
    bag = bagOfWords(s, words)
    x_train.append(bag)

    tags = intents.index(i) # Have numbers for intents
    y_train.append(tags) 

x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 8
input_size = len(words)
hidden_size = 8
output_size = len(intents)
learning_rate = .01
num_epochs = 1000

dataset = ChatDataset(x_train, y_train)
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NN(input_size, hidden_size, output_size).to(device)

# crossentropy loss

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop

for epoch in range(num_epochs):
    for (wrds, labls) in train_loader:
        wrds = wrds.to(device)
        labls = labls.to(device)

        # forward
        outputs = model(wrds)
        labls = labls.long()
        loss = criterion(outputs, labls)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": words,
"tags": intents
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')