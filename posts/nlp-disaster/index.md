---
title: "Identificando desastres en Twitter con NLP"
subtitle: "Aprende a usar Natural Language Processing (NLP) para identificar desastres en Twitter"
author: "Diegulio"
date: "2023-03-01"
categories: [python, nlp, kaggle]
image: "posts/nlp-disaster/twitter_disaster.png"
---
[![](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/diegomachado/seqclass-nn-embed-rnn-lstm-gru-bert-hf)

# Goal 

I'm learning NLP. So to do that I decided pass trough diverse NLP models, study the teory and code them! 
I think that it is a good method to learn Machine Learning things. So, Disclaimer: All the content on the notebooks is what I understood from diverse references (I will put the links), somethings could be wrong, If you find any mistake please let me know, so I can learn of it. Also, if you have some doubt, it will be a pleasure to me to answer it (as long as I have the answer).

At the end, I achieve a score of 0.843 in the LB. Is beatifull to see how you are improving the solutions step by step!

So, this will be the embedding Notebook, I will put the link to each specific notebook here (So it will be more readable).

Methodologies & Notebooks:

|           Model Notebook          | Score |
|:-------------------------:|:-----:|
| [Simple Neural Network](post.html?slug=nlp-disaster/1-simple-nn) ( [View on kaggle](https://www.kaggle.com/code/diegomachado/seqclass-1-simple-nn-0-56))    | 0.56  |
| [Embeddings](post.html?slug=nlp-disaster/2-embeddings) ( [View on kaggle](https://www.kaggle.com/code/diegomachado/seqclass-2-embeddings-0-797))                | 0.797 |
| [Recurrent Neural Networks](post.html?slug=nlp-disaster/3-rnn) ( [View on kaggle](https://www.kaggle.com/code/diegomachado/seqclass-3-rnn-0-809/notebook)) | 0.809 |
| [BERT & HuggingFace](post.html?slug=nlp-disaster/4-bert) ( [View on kaggle](https://www.kaggle.com/code/diegomachado/seqclass-4-bert-tensorflow-huggingface-0-824/notebook))        | 0.824 |
| [MyBestSolution](post.html?slug=nlp-disaster/5-my-best-solution) ( [View on kaggle](https://www.kaggle.com/code/diegomachado/seqclass-5-mybestsolution-0-843/notebook))            | 0.843 |

# EDA 
Here I will do some preprocessing and split the data. I will use that data to each notebook!

I think there is a lot of notebooks with a beatifull EDA, So I won't take to much around this.

### Libraries

```python
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import gc

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Lambda
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
#import tensorflow_hub as hub
#import tensorflow_text as text # Bert preprocess uses this 
from tensorflow.keras.optimizers import Adam

import re
import nltk
from nltk.corpus import stopwords
import string
from gensim.models import KeyedVectors

#nltk.download('stopwords')
```

### Data

```python
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
train_df.head()
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>

```python
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
test_df.head()
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>
</div>

### EDA

```python
# Target Proportion
sns.countplot(data=train_df, x = "target")
```

```text
<AxesSubplot:xlabel='target', ylabel='count'>
```

![](assets/output-1.png)

I think it is balanced!

```python
# Random example of disaster tweet
train_df[train_df.target == 1].sample(1).text.values[0]
```

```text
'California Bush fires please evacuate affected areas ASAP when california govts advised you to do so http://t.co/ubVEVUuAch'
```

```python
# Random example of NO disaster tweet
train_df[train_df.target == 0].sample(1).text.values[0]
```

```text
'Someone split a mudslide w me when I get off work'
```

### Pre-processing

I will do some preprocessing with Tensorflow!

```python
# Input Tensor Data
text = tf.data.Dataset.from_tensor_slices(train_df.text)
text
```

```text
2022-12-11 15:20:58.687517: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
```

```text
<TensorSliceDataset shapes: (), types: tf.string>
```

Note that Im reading data from memory! If it would huge data I would be in troubles! 

One advantage of initialize a Tensorflow dataset is that I will be able to create a data pipeline (batch, fetch, shuffle, etc.)

```python
# some samples
list(text.take(2).as_numpy_iterator())
```

```text
[b'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all',
 b'Forest fire near La Ronge Sask. Canada']
```

We need to know that models don't understand text by itself! Just numbers! For this, we vectorize the sentences. I don't plan to use a model now, I would like to observe wich words are more present by target! (Also I don't want to consider **stopwords**) 

I will use the tensorflow layer: Text Vectorization. from behind, it apply lowercase and delete punctuation. I also wants to remove stopwords, so I will build a custom standarization that do: 
1. lowercase
2. strip punctuation
3. remove stop words! 

[Click here if you don't know what are stop words](https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/)

```python
#### COUNT WORDS BY TARGET

def custom_standardization(inputs):
    """
    Apply: lowercase, remove punctuation and stopwords
    """
    PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
    lowercase = tf.strings.lower(inputs) # lowercase
    strip = tf.strings.regex_replace(lowercase, PUNCTUATION, '') # strip punctuation
    stopwrd = tf.strings.regex_replace(strip, r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*', '')
    return stopwrd
    

# model to apply vectorize_layer with custom standardization
vectorize_layer = tf.keras.layers.TextVectorization(output_mode = 'multi_hot', standardize = custom_standardization)
vectorize_layer.adapt(text)

# model to vectorize
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)

# make counter
train_count = model.predict(text.batch(batch_size = len(text))) # predict to count 
token_counts = pd.DataFrame(columns = vectorize_layer.get_vocabulary(), data = train_count) # df with tokens and count
train_df.rename(columns = {"target":"disaster_target"}, inplace = True) # rename target because there is a word target in data
count_df = pd.concat([train_df, token_counts], axis = 1) #concat
group_count = count_df.iloc[:,4:].groupby("disaster_target", as_index = False).sum() # count token for each target
melt_count = pd.melt(group_count, id_vars=["disaster_target"], value_name = "count") # each token to row
melt_count.sort_values(by=["count"], ascending = False).head(30)
```

```text
2022-12-11 15:20:58.997911: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
/opt/conda/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3552: FutureWarning: This dataframe has a column name that matches the 'value_name' column name of the resulting Dataframe. In the future this will raise an error, please set the 'value_name' parameter of DataFrame.melt to a unique name.
  exec(code_obj, self.user_global_ns, self.user_ns)
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>disaster_target</th>
      <th>variable</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>like</td>
      <td>239.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>im</td>
      <td>221.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>amp</td>
      <td>174.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>new</td>
      <td>163.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>fire</td>
      <td>162.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>get</td>
      <td>158.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>dont</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>news</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>one</td>
      <td>122.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>via</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>body</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1</td>
      <td>california</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>suicide</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>people</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>police</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>disaster</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>via</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>amp</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>would</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1</td>
      <td>killed</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>video</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>people</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>like</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>1</td>
      <td>hiroshima</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>1</td>
      <td>fires</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0</td>
      <td>know</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>2</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>0</td>
      <td>full</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0</td>
      <td>love</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0</td>
      <td>time</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

After I build it I realized that it is not necessary to instantiate a model to use layers! 😅

# Split Data

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test, y_train, y_test = train_test_split(train_df[[col for col in train_df.columns if col != 'disaster_target']], train_df.disaster_target, test_size=0.2, random_state=13)
```

```python
# To csv (In some notebooks I will use this data)
pd.concat([X_train, y_train], axis = 1).to_csv('df_train.csv',index = False)
pd.concat([X_test, y_test], axis = 1).to_csv('df_test.csv',index = False)
```

This dataset is here: https://www.kaggle.com/datasets/diegomachado/df-split

```python
# Delete it from memory
del train_df, test_df, X_train, X_test, y_train, y_test
gc.collect()
```

```text
671
```
