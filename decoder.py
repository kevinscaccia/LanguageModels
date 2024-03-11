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


def get_batch(data, batch_size, block_size):
    # get random start batches indexes (one per batch)
    idx_start = torch.randint(len(data)-block_size, (batch_size,))
    # get each batch and stack them
    x = torch.stack([data[i:i+block_size] for i in idx_start])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx_start])
    return x.to(device), y.to(device)


class AttentionHead(nn.Module):
    def __init__(self, num_features, num_steps, head_size, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(num_features, head_size, bias=False)
        self.key = nn.Linear(num_features, head_size, bias=False)
        self.value = nn.Linear(num_features, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        # tensor that aren't parameters
        self.register_buffer('tril_mask', torch.tril(torch.ones((num_steps, num_steps))))
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, H) - H=head_size
        k = self.key(x) # (B, T, H)
        v = self.value(x) # (B, T, H)
        # affinities inter tokens (and scale)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5) # (B, T, T)
        # mask future tokens
        wei = wei.masked_fill(self.tril_mask[:T, :T] == 0, float('-inf'))
        # normalize each row(each token interactions in last dim)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # dropout some of the affinities
        y = wei @ v # apply affinities (B, T, H) weighted agreggation
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_steps, head_size, num_heads, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embed, num_steps, head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.concat([h(x) for h in self.heads], dim=-1)   
        x = self.linear(x)# projection to back from residual path
        x = self.dropout(x)
        return  x    

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        feed_forward_dim = n_embed * 4 # can be a hyperparameter (its a projection)
        self.net = nn.Sequential(
            nn.Linear(n_embed, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, n_embed),# projection to back to residual path
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, num_steps, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.self_attention = MultiHeadAttention(n_embed, num_steps, head_size, num_heads)
        self.feedforward = FeedForward(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed) # per token normalization
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))  # residual connections
        x = x + self.feedforward(self.layer_norm_2(x))   # residual connections
        return x

class AttentionModel(nn.Module):
    def __init__(self, model_params):
        super(AttentionModel, self).__init__()
        # Set model vars
        expected_vars = ['vocab_size','block_size','emb_dim','num_heads','n_layers']
        for v in expected_vars:
            assert v in model_params.keys(), f'Key "{v}" is missing on params dict'
            vars(self)[v] = model_params[v]
        #
        assert (self.emb_dim % self.num_heads == 0), 'emb_dim must be divisible by num_heads'
        # maps each token integer in a vector of emb_dim dimensions
        # learnable embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.block_size, self.emb_dim) # positional embeddig (for each position from 0 to blocksize-1 it will return an embedding(different vector))
        
        self.blocks = nn.Sequential(*[TransformerBlock(self.emb_dim, self.block_size, self.num_heads) for _ in range(self.n_layers)])
        self.layer_norm = nn.LayerNorm(self.emb_dim) # model final layer norm
        
        # self.mh_attention = MultiHeadAttention(self.emb_dim, self.block_size, self.head_size//self.num_heads, self.num_heads)
        self.feedforward = FeedForward(self.emb_dim)
        self.dense = nn.Linear(self.emb_dim, self.vocab_size) # final scores
        #
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, token_seq):
        B, T = token_seq.shape
        token_emb = self.token_emb(token_seq) # (B, T) --> (B, T, Emb)
        pos_emb = self.pos_emb(torch.arange(T).to(device))# (B, T) --> (B, T, Emb)
        x = token_emb + pos_emb # # concat embeddings x is holds token value(identity, embeddings from value) and positional information of this token (encoded as numbers(embs..))
        # x = self.mh_attention(x) # feature extraction outputs (B, T, head_size)
        # # computation
        # # x = self.feedforward(x) # outputs (B, T, head_size)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.dense(x) # outputs (B, T, vocab_size)
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
            logits = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] # (B, T, vocab_size)
            probs = F.softmax(logits, dim=-1) # (B, vocab_size) normalized logits = probability of each token be the next
            # sample from probability
            idx_next = torch.multinomial(probs, num_samples=1) # get 1 token_id per batch (B, 1)
            # concatenate in each batch along the time dimension
            idx = torch.concat((idx, idx_next), dim=1)
        self.train()
        return idx
    
    @torch.no_grad()
    def estimate_loss(self, data):
        self.eval()
        losses = []
        for i in range(100):# 100 batches
            batch_X, batch_y = get_batch(data, batch_size, self.block_size)
            loss = self.compute_loss(batch_y, self(batch_X)).item()
            losses.append(loss)
        self.train()
        return np.mean(losses)
#
#
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
batch_size = 32
block_size = 256
model_params = {
    'block_size': block_size,
    'emb_dim': 256,
    'num_heads': 8,
    'n_layers':6, # number of stacked attention+computation blocks
    'vocab_size':tokenizer.vocab_size,
}
model = AttentionModel(model_params).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Num of weights:',pytorch_total_params)
#
#
lr = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
epochs = 10_000

import time
for step in range(epochs):
    timr = time.time()
    X, y = get_batch(train_corpus, batch_size, block_size)
    pred_y = model(X)
    loss = model.compute_loss(y, pred_y)
    #
    optimizer.zero_grad() # current batch zero-out the loss
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        timr = time.time() - timr
        print(f'Step {step+1}/{epochs}  [{timr:.3f}secs] --> Extimated(val) loss: {model.estimate_loss(val_corpus)} ')
    
    if step % 50 == 0:
        torch.save(model, f'transformer_pt_{step}.model')

new_tokens = 100
prompt = 'Todas as pessoas gostam de'
print(f'Generation for prompt "{prompt}":')
prompt = tokenizer.encode(prompt).unsqueeze(0).to(device) # add batch dimension (1, T)
print(prompt.shape)
gen = model.generate(prompt, new_tokens)[0]
print('--> ', tokenizer.decode(gen))