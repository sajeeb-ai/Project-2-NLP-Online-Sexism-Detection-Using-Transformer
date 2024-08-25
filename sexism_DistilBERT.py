#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
os.environ['WANDB_DISABLED'] = 'true'


# In[6]:


get_ipython().system('pip install -U tensorflow==2.11')


# In[7]:


import transformers


# In[8]:


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


# In[9]:


df = pd.read_csv('/kaggle/input/sexism/starting_ki/train_all_tasks.csv')
df.head()


# In[10]:


df.shape
data = df


# In[11]:


category_count = df['label_vector'].value_counts()

categories = category_count.index

categories


# In[12]:


category_count


# In[13]:


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


# In[14]:


df['encoded_categories'] = df['label_vector'].astype('category').cat.codes

df['encoded_categories'].unique().size


# In[15]:


df.head()


# In[16]:


x = df['text'].to_list()
y = df['encoded_categories'].to_list()


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


# In[18]:


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(x_train, truncation = True, padding = True  )

val_encodings = tokenizer(x_val, truncation = True, padding = True )


# In[19]:


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))


val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))


# In[20]:


model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=12)


# In[21]:


from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments


training_args = TFTrainingArguments(
    output_dir='./results',          
    num_train_epochs=20,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=1e-5,               
    logging_dir='./logs',            
    eval_steps=100                   
)

with training_args.strategy.scope():
    trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 12 )


trainer = TFTrainer(
    model=trainer_model,                 
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
)


# In[22]:


trainer.train()


# In[23]:


trainer.evaluate()


# In[26]:


save_directory = "/kaggle/working/saved_models" 

model.save_pretrained(save_directory)

tokenizer.save_pretrained(save_directory)


# In[27]:


#import shutil
#shutil.make_archive('results', 'zip', '/kaggle/working/results')
#shutil.make_archive('logs', 'zip', '/kaggle/working/logs')
shutil.make_archive('saved_models', 'zip', '/kaggle/working/saved_models')


# In[28]:


shutil.rmtree('results')
shutil.rmtree('logs')
shutil.rmtree('saved_models')


# In[29]:


get_ipython().system('pip install mega.py')


# In[30]:


from mega import Mega
mega = Mega()
m = mega.login('mail', 'pass')
# login using a temporary anonymous account
m.get_user()


# In[31]:


file = m.upload('saved_models.zip')
file1 = m.upload('results.zip')
file2 = m.upload('logs.zip')


# In[43]:


from zipfile import ZipFile
  
# loading the temp.zip and creating a zip object
with ZipFile('saved_models.zip', 'r') as zObject:
  
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path="saved_models")


# In[44]:


tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)

model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)


# In[85]:


test = pd.read_csv('/kaggle/input/sexism/starting_ki/gab_1M_unlabelled.csv')


# In[86]:


test = test['text'].to_list()
testn = test[0]
testn


# In[87]:


predict_input = tokenizer_fine_tuned.encode(
    testn,
    truncation = True,
    padding = True,
    return_tensors = 'tf'    
)

output = model_fine_tuned(predict_input)[0]

prediction_value = tf.argmax(output, axis = 1).numpy()[0]

prediction_value


# In[93]:


res = df.loc[df['encoded_categories'] == prediction_value]
res = res.iloc[0]


# In[92]:


res


# In[ ]:




