import re 
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
torch.manual_seed(7)

device = 'cuda'

class CharLevelTokenizer():
    def __init__(self,):
        self.OOV = -1

    def fit_transform(self, corpus):
        # remove special chars
        corpus = re.sub(r'[^a-zA-Z\s\sàáâãäèéêëìíîïòóôõöùúûüçß]' , '', corpus)
        vocab = sorted(set(corpus))
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # mapping 
        self.char_to_int = {c:i  for i,c in enumerate(vocab)}
        self.int_to_char = {i:c  for i,c in enumerate(vocab)}
        return corpus
    
    def encode(self, txt): # dont threat OOV 
        return torch.tensor([self.char_to_int[c] for c in txt], dtype=torch.long)
    
    def decode(self, token_seq):
        token_seq = token_seq.to('cpu').numpy()
        return ''.join([self.int_to_char[i] for i in token_seq])

def get_batch(batch_size, block_size, from_train: bool):
    data = train_corpus if from_train else val_corpus
    # get random start batches indexes (one per batch)
    idx_start = torch.randint(len(train_corpus)-block_size, (batch_size,))
    # get each batch and stack them
    x = torch.stack([data[i:i+block_size] for i in idx_start])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx_start])
    return x.to(device), y.to(device)

class BigramModel(nn.Module):
    def __init__(self,num_embeddings, emb_dim):
        super().__init__()
        # maps each token integer in a vector of emb_dim dimensions
        self.token_embedding_table = nn.Embedding(num_embeddings, emb_dim)
    
    def forward(self, token_seq):
        logits = self.token_embedding_table(token_seq)
        # produces a score for each other token, indicating the chance of it be the next
        return logits 

    def compute_loss(self, real_y, pred_y_logits):
        # logits = isn't normalized to probabilities
        # real_y contais the token index. logits contains one score per vocab possibility
        B, T, C = pred_y_logits.shape
        loss = F.cross_entropy(pred_y_logits.view(B*T, C), real_y.view(-1))# itsnot batch_first, need spreedout feature dim
        return loss
    
    @torch.no_grad
    def generate(self, idx, next_steps): # generate for each batch
        self.eval()
        idx = idx.clone()
        # idx (B, T) is the array of token indexes of current history/context
        for i in range(next_steps):
            # print(idx[0])
            last_step_logits = self(idx)[:, -1, :] # (B, T, vocab_size)
            probs = F.softmax(last_step_logits, dim=-1) # (B, vocab_size) normalized logits = probability of each token be the next
            # sample from probability
            idx_next = torch.multinomial(probs, num_samples=1) # get 1 token_id per batch (B, 1)
            # concatenate in each batch along the time dimension
            idx = torch.concat((idx, idx_next), dim=1)
        self.train()
        return idx
    
    @torch.no_grad()
    def estimate_loss(self):
        self.eval()
        losses = []
        for i in range(100):# 100 batches
            batch_X, batch_y = get_batch(batch_size, block_size, True)
            loss = self.compute_loss(batch_y, self(batch_X)).item()
            losses.append(loss)
        self.train()
        return np.mean(losses)
        


#
# Data
#
corpus = open('text.txt','r').readlines()
print(f'There are {len(corpus)} lines of PT raw text')
corpus = ' '.join(corpus)
print(f'There are {len(corpus)} chars')
tokenizer = CharLevelTokenizer()
corpus = tokenizer.fit_transform(corpus)
print(f'There are {tokenizer.vocab_size} chars (without special chars)-> {tokenizer.vocab}')


# 80% train
train_corpus = tokenizer.encode(corpus[:int(len(corpus)*0.8)])
val_corpus = tokenizer.encode(corpus[int(len(corpus)*0.8):])
print(f'Train corpus:      {len(train_corpus)} tokens')
print(f'Validation corpus: {len(val_corpus)} tokens')
torch.manual_seed(7)
#
# Train Config
#
epochs = 20_000
batch_size = 512
block_size = 17
lr = 1e-3
model = BigramModel(tokenizer.vocab_size, tokenizer.vocab_size).to(device)
#
# Train
#
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(epochs):
    X, y = get_batch(batch_size, block_size, True)
    pred_y = model(X)
    loss = model.compute_loss(y, pred_y)
    #
    optimizer.zero_grad() # current batch zero-out the loss
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f'Batch {step+1}/{epochs}-> Estimated loss: {model.estimate_loss()}')
#
#
#
new_tokens = 1000
prompt = 'O homem está indo até a '
print(f'Generation for prompt "{prompt}":')
prompt = tokenizer.encode(prompt).unsqueeze(0).to(device) # add batch dimension (1, T)
gen = model.generate(prompt, new_tokens)[0]

print('--> ', tokenizer.decode(gen))