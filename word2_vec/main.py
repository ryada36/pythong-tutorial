import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cbow_model.pth')


# Step 1: Download the necessary NLTK data
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 2: Load the Reuters corpus and preprocess the data
corpus = reuters.sents()
stop_words = set(stopwords.words('english'))

# # Preprocess the corpus (remove stopwords, non-alphabetic words)
processed_corpus = []

for sentence in corpus:
    sentence = [word.lower() for word in sentence if word.isalpha() and word not in stop_words]
    processed_corpus.append(sentence)

# # Flatten the list of sentences into a list of all words
all_words = [word for sentence in processed_corpus for word in sentence]

# Create vocabulary (unique words)
vocab = set(all_words)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 3: Create CBOW training data
window_size = 2  # Number of context words on each side of the target word
X = []
y = []

for sentence in processed_corpus:
    for i in range(window_size, len(sentence) - window_size):
        context = sentence[i - window_size:i] + sentence[i + 1:i + window_size + 1]
        target = sentence[i]
        
        # Convert words to indices
        context_idx = [word_to_idx[word] for word in context]
        target_idx = word_to_idx[target]
        
        X.append(context_idx)
        y.append(target_idx)

# Convert to numpy arrays for easier manipulation
X = np.array(X)
y = np.array(y)

# Step 4: Define the CBOW model in PyTorch
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        emb = self.embeddings(context)
        context_vector = emb.mean(dim=1)  # Average context word embeddings
        out = self.out(context_vector)  # Output layer
        return out

# Initialize the model, loss function, and optimizer
embedding_dim = 100  # Size of the word embeddings
model = CBOWModel(len(vocab), embedding_dim)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode for inference
    print("Model loaded successfully.")
else:
    print("No saved model found. Training from scratch.")

# Step 5: Train the model
# this model takes a lot of time to train
# either introduce negative sampling or use a smaller dataset
# For demonstration, we will train for a few epochs
epochs = 1
for epoch in range(epochs):
    total_loss = 0
    for i in tqdm(range(len(X)), desc="Training"):
        # Get a batch of data
        context = torch.tensor(X[i], dtype=torch.long).unsqueeze(0)  # Add batch dimension
        target = torch.tensor(y[i], dtype=torch.long).unsqueeze(0)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(context)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)

# Step 6: Testing the model - Get word vector and test similarity
def get_word_vector(word):
    word_idx = word_to_idx.get(word, None)
    if word_idx is None:
        print(f"Word '{word}' not in vocabulary.")
        return None
    word_tensor = torch.tensor([word_idx], dtype=torch.long)
    word_vector = model.embeddings(word_tensor)
    return word_vector

# Example: Get vector for the word 'economy'
word_vector = get_word_vector('economy')
if word_vector is not None:
    print(f"Vector for 'economy': {word_vector}")

# Step 7: Checking similarity (example: cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity(word1, word2):
    vector1 = get_word_vector(word1)
    vector2 = get_word_vector(word2)
    
    if vector1 is not None and vector2 is not None:
        similarity = cosine_similarity(vector1.detach().numpy(), vector2.detach().numpy())
        return similarity[0][0]
    return None

# function to get all the similar words
def get_similar_words(word, top_n=5):
    word_vector = get_word_vector(word)
    if word_vector is None:
        return []
    
    similarities = []
    for other_word in vocab:
        if other_word != word:
            other_vector = get_word_vector(other_word)
            if other_vector is not None:
                similarity = cosine_similarity(word_vector.detach().numpy(), other_vector.detach().numpy())
                similarities.append((other_word, similarity[0][0]))
    
    # Sort by similarity and return top_n words
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Example: Get similarity between 'economy' and 'finance'
similarity = get_cosine_similarity('economy', 'finance')
get_similar_words('economy')
print(f"Top 5 similar words to 'economy': {get_similar_words('economy')}")
if similarity is not None:
    print(f"Cosine similarity between 'economy' and 'finance': {similarity}")
