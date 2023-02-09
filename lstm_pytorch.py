
import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
from torch.nn.modules.dropout import Dropout
from google.colab import files

moby_d = files.upload()

class Get_dictionary(object):
    
    def __init__(self):
        
        self.index = 0
        self.word_2_index = {}
        self.index_2_word = {}
       
    
    def convert_word(self, word):
        
        if word not in self.word_2_index:
            self.word_2_index[word] = self.index
            self.index_2_word[self.index] = word
            self.index +=1
    
    def __len__(self):
        
        return len(self.word_2_index)
  



class Preprocessing_doc(object):
    
    def __init__(self):
        
        self.Get_dictionary = Get_dictionary()
    
    def clean_text(self, path, batch_size=32):

        token = 0
        
        with open(path, "r") as f:
            
            for sentence in f:
                words = sentence.split() + ["<eos>"]
                
                token += len(words)
                
                for word in words:
                    if word not in '\n\n \n\n\n!"-#$%&()--*+-/:;<=>?@[\\]^_`{|}~\t\n':
         
                        self.Get_dictionary.convert_word(word)
                        
                        
        tensor = torch.LongTensor(token)
        
        index = 0
        
        with open(path, "r") as f:
            for sentence in f:
                words = sentence.split() + ["<eos>"]
                
                for word in words:
                    if word not in '\n\n \n\n\n!"-#$%&()--*+-/:;<=>?@[\\]^_`{|}~\t\n':
                        tensor[index] = self.Get_dictionary.word_2_index[word]
                        index +=1
        
        batch_amount = tensor.shape[0]//batch_size
    
        tensor = tensor[:batch_amount*batch_size]
    
        tensor = tensor.view(batch_size, -1)

        return tensor

    

batch_size = 32
embed_size = 300
hidden_size = 128
num_layers = 2
epoch = 100
timesteps = 25

text_tensor = Preprocessing_doc()

text = text_tensor.clean_text("moby_dick_four_chapters.txt")

print(text.shape)

vocab_len = len(text_tensor.Get_dictionary)



class LSTM_net(nn.Module):

  def __init__(self, vocab_len, embed_size, hidden_size, num_layers):

    super(LSTM_net, self).__init__()

    self.embedding = nn.Embedding(vocab_len, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.2, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_len)
    
  def forward(self, x, h):

    x = self.embedding(x)
    output, (h,c) = self.lstm(x, h)

    output = output.reshape(output.size(0)*output.size(1), output.size(2))
    output = self.fc(output)

    return output, (h,c)

model = LSTM_net(vocab_len, embed_size, hidden_size, num_layers)

print(model.share_memory)

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.002)

for e in range(epoch):

  hidden = (torch.zeros(num_layers, batch_size, hidden_size),
           torch.zeros(num_layers, batch_size, hidden_size) )
  
  for i in range(0, text.size(1) - timesteps, timesteps):
    
    inputs = text[:, i:i+timesteps]

    target = text[:, i+1:(i+1)+timesteps]
 
    outputs, (h,c) = model(inputs, hidden)
    loss = loss_fun(outputs, target.reshape(-1))



    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(model.parameters(), 0.4)
    optimizer.step()

    step = (i+1)// timesteps
    if step % 100 ==0:
      print(f"Epoch {e+1}/{epoch} Loss|: {np.mean(loss.item())}")

      

with torch.no_grad():
  
  with open("moby_results.txt", "w") as f:

    hidden = (torch.zeros(num_layers,1,hidden_size),
             torch.zeros(num_layers,1,hidden_size))

    input = torch.randint(0,vocab_len, (1,)).long().unsqueeze(1)

    for i in range(1000):

      output, (h, c)= model(input, hidden)
      prob = output.exp()
      word_id = torch.multinomial(prob, num_samples=1, replacement=True)

      word_id = word_id.item()

      word = text_tensor.Get_dictionary.index_2_word[word_id]
      word = "\n" if word == "<eos>" else word + " "

      f.write(word)

      if(i+1) % 100 == 0:
        print(f" Sample no: [{i+1}/{1000}], saving file  {'moby_results.txt'}")



