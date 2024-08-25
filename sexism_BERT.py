#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10240"


# # Installing Transformers

# In[2]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install transformers[sentencepiece]')


# In[3]:


import transformers


# # Importing libraries

# In[4]:


import keras
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline

import tensorflow as tf
import pandas as pd
import json
import gc
import numpy as np
from sklearn.model_selection import train_test_split

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopw = stopwords.words('english')

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot

from tqdm import tqdm


# # Read Train dataset

# In[5]:


df = pd.read_csv('/kaggle/input/sexism/starting_ki/train_all_tasks.csv')
df.head()


# In[6]:


df.shape
data = df


# # Taking label_vectors as catagories

# In[7]:


category_count = df['label_vector'].value_counts()

categories = category_count.index

categories


# # Define veriables

# In[8]:


category_count
target_list = categories
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05


# In[9]:


fig = plt.figure(figsize= (12, 5))

ax = fig.add_subplot(111)

sns.barplot(x = category_count.index, y = category_count )

for a, p in enumerate(ax.patches):
    ax.annotate(f'{categories[a]}\n' + format(p.get_height(), '.0f'), xy = (p.get_x() + p.get_width() / 2.0, p.get_height()), xytext = (0,-25), size = 13, color = 'white' , ha = 'center', va = 'center', textcoords = 'offset points', bbox = dict(boxstyle = 'round', facecolor='none',edgecolor='white', alpha = 0.5) )
    
plt.xlabel('Categories', size = 15)

plt.ylabel('The Number of Comments', size= 15)

plt.xticks(size = 20)

plt.title("The number of Comments categories" , size = 40)

plt.show()


# In[10]:


df.head()


# # Adding categories as features columns and droping extra columns

# In[11]:


copydf = df
for name in categories:
    x = copydf['label_vector'].to_list()
    y = []
    for i in x:
        if name == i:
            y.append(1)
        else:
            y.append(0)
    copydf[name] = y
copydf = copydf.drop(['label_sexist', 'label_vector', 'label_category','rewire_id'], axis=1)


# In[12]:


from transformers import BertTokenizer, BertModel


# # Tokenizer

# In[13]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[14]:


import torch


# # Custom dataset class for custom dataset processing

# In[15]:


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['text']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }
     


# # Train test Split

# In[16]:


from sklearn.model_selection import train_test_split
import numpy
train_df ,val_df = train_test_split(copydf,test_size=0.3, random_state=42)


# 
# # Fixing unordered index

# 
# 
# 

# In[17]:


val_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)


# # Creating Datasets

# In[18]:


train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)


# # Making Data Loader

# In[19]:


train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# # Functions to load and save checkpoint

# In[20]:


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# In[21]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# # Model Class

# In[22]:


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 12)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

model = BERTClass()
model.to(device)


# # Loss Function

# In[23]:


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# In[24]:


val_targets=[]
val_outputs=[]


# In[25]:


len(val_data_loader)


# In[26]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[27]:


def accuracy_fn(output_p, target_o):
    return torch.sum(output_p == target_o)


# In[28]:


import gc
import re


# # Custom Function for training Model

# In[45]:


def train_model(n_epochs, training_loader, validation_loader, model, 
                optimizer, checkpoint_path, best_model_path):
   
  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf
   
 
  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0
    total_accuracy = 0
    total_recal = 0
    total_prec = 0
    total_f1 = 0
    count = 0
    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        #print('yyy epoch', batch_idx)
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        out = outputs.cpu().detach().numpy()
        out = (out > 0.5)
        out = out*1
        tar = targets.cpu().detach().numpy()
        tar = tar.astype(int)
        train_acc = accuracy_score(tar, out)
        total_accuracy = total_accuracy+train_acc
        total_recal = total_recal+recall_score(tar, out, average='micro')
        total_prec = total_prec+precision_score(tar, out, average='micro')
        total_f1 = total_f1+f1_score(tar, out, average='micro')
        if batch_idx%50==0:
            print(f'Epoch: {epoch}, Training Loss:  {loss.item()}, Accuracy: {train_acc}')
        count+=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    print(f'Avg Accuracy: {total_accuracy/count}, Avg Racall:  {total_recal/count}, Avg Precision: {total_prec/count}, Avg F1: {total_f1/count}')
    total_accuracy = 0
    total_recal = 0
    total_prec = 0
    total_f1 = 0
    count = 0
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            out = outputs.cpu().detach().numpy()
            out = (out > 0.5)
            out = out*1
            tar = targets.cpu().detach().numpy()
            tar = tar.astype(int)
        
            loss = loss_fn(outputs, targets)
            val_acc = accuracy_score(tar, out)
            total_accuracy = total_accuracy+val_acc
            total_recal = total_recal+recall_score(tar, out, average='micro')
            total_prec = total_prec+precision_score(tar, out, average='micro')
            total_f1 = total_f1+f1_score(tar, out, average='micro')
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            torch.cuda.empty_cache()
            count+=1

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f} '.format(
            epoch, 
            train_loss,
            valid_loss
            ))
      print(f'Avg Accuracy: {total_accuracy/count}, Avg Racall:  {total_recal/count}, Avg Precision: {total_prec/count}, Avg F1: {total_f1/count}')
      
      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
        
        # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model


# In[46]:


import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


# In[47]:


import shutil
torch.cuda.empty_cache()


# In[48]:


trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, 'ckpt', 'model')


# # Training Model

# # Loading Unlabelled data for prediction

# In[49]:


load = pd.read_csv('/kaggle/input/sexism/starting_ki/gab_1M_unlabelled.csv')
load = load['text'].to_list()
text = load[610]
text


# # Function for prediction

# In[50]:


def Make_Predictions(example):
    encodings = tokenizer.encode_plus(
        example,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    model.eval()
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device, dtype=torch.long)
        attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
        output = model(input_ids, attention_mask, token_type_ids)
        final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
        print(final_output)
        print(target_list[int(np.argmax(final_output, axis=1))])


# In[51]:


Make_Predictions(text)


# In[ ]:




