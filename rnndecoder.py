import torch.nn as nn
import torch
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.query_w = nn.Linear(query_dim, hidden_dim)
        self.key_w = nn.Linear(key_dim, hidden_dim)
        self.values_w = nn.Linear(key_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, key):
        query = query.unsqueeze(1)
        score = torch.tanh(self.query_w(query) + self.key_w(key))
        values = self.values_w(key)
        attention_weights = F.softmax(self.proj(score), dim = 1)

        context = attention_weights * values
        context = torch.sum(context,dim = 1)
        return context, attention_weights




class RNNDecoder(nn.Module):


    def __init__(self, hidden_dim , embedding_size, vocab_size):
        super(RNNDecoder,self).__init__()
        self.hidden_units = hidden_dim
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_dim, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim,embedding_size, hidden_dim )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self,input, keys, query):
        input = self.embedding(input)
        context_vector, attention_weights = self.attention(query, keys)

        # x = torch.concat((context_vector.unsqueeze_(0), query), dim = 1)

        context_vector = context_vector.unsqueeze(0)
        # input = torch.cat((input, context_vector), dim = 1)
        output, state = self.gru(input, context_vector)
        output = F.log_softmax(self.out(output), dim=2)
        output = output.squeeze(1)
        state = state.squeeze(0)
        return output, state, attention_weights

    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_units))





class CNNEncoder(nn.Module):

    def __init__(self, input_dim, embedding_dim ):
        super(CNNEncoder, self).__init__()
        self.affine = nn.Linear(input_dim, embedding_dim)

    def forward(self, features):
        out = self.affine(features)
        return out







