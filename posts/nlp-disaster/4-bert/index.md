---
title: "4. BERT"
subtitle: ""
author: "Diegulio"
date: "2023-03-01"
categories: [python, nlp, kaggle]
image: "posts/nlp-disaster/disaster.jpeg"
---
# 4. BERT (Bidirectional Encoder Representations from Transformers)

As we see in the previous notebook, RNN improve the results and give us some benefits. But they are expensive to train, mainly because they are not paralellizable! Now we are in the transformers era. Transformers are models that leverage the mechanism of **self-attention**.

In this specific case we will apply BERT. I will let you with some references that help me to understand what is happend behind scenes:

- http://nlp.seas.harvard.edu/annotated-transformer/
- http://jalammar.github.io/illustrated-transformer/
- https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- https://www.youtube.com/watch?v=xI0HHN5XKDo&t=2s&pp=ugMICgJlcxABGAE%3D
- https://www.youtube.com/watch?v=7kLi8u2dJz0
- https://www.youtube.com/watch?v=TQQlZhbC5ps&list=PLTl9hO2Oobd_bzXUpzKMKA3liq2kj6LfE&index=1

> Remember that this belong to a NLP Notebook series where I am learning and testing different NLP approachs in this competition. Like NN, Embedding, RNN, Transformers, HuggingFace, etc.

> To see the other notebooks visit: https://www.kaggle.com/code/diegomachado/seqclass-nn-embed-rnn-lstm-gru-bert-hf

# Libraries

```python
# A dependency of the preprocessing for BERT inputs
!pip install -q -U "tensorflow-text==2.8.*"
```

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tfx-bsl 1.9.0 requires tensorflow!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,<3,>=1.15.5, but you have tensorflow 2.8.4 which is incompatible.
tensorflow-transform 1.9.0 requires tensorflow!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,<2.10,>=1.15.5, but you have tensorflow 2.8.4 which is incompatible.
tensorflow-serving-api 2.9.0 requires tensorflow<3,>=2.9.0, but you have tensorflow 2.8.4 which is incompatible.
tensorflow-io 0.21.0 requires tensorflow<2.7.0,>=2.6.0, but you have tensorflow 2.8.4 which is incompatible.
tensorflow-io 0.21.0 requires tensorflow-io-gcs-filesystem==0.21.0, but you have tensorflow-io-gcs-filesystem 0.28.0 which is incompatible.
pytorch-lightning 1.7.7 requires tensorboard>=2.9.1, but you have tensorboard 2.8.0 which is incompatible.
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

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
import tensorflow_hub as hub
import tensorflow_text as text # Bert preprocess uses this 
from tensorflow.keras.optimizers import Adam

import re
import nltk
from nltk.corpus import stopwords
import string
from gensim.models import KeyedVectors

#nltk.download('stopwords')
```

# Data

Due to BERT model could consume a lot of ram. This time we will read our data directly from the disc and so feed it to the model!

For this we will use Tensorflow datasets function [`make_csv_dataset`](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)

```python
train_file_path = "../input/df-split/df_split/df_train.csv"
val_file_path = "../input/df-split/df_split/df_test.csv"
```

```python
train_ds = tf.data.experimental.make_csv_dataset(
    train_file_path,
    batch_size=32, # Artificially small to make examples easier to show.
    label_name='target',
    select_columns=["text","target"],
    num_epochs=1, # Not repeat the dataset
    ignore_errors=True,)

test_ds = tf.data.experimental.make_csv_dataset(
    val_file_path,
    batch_size=32, # Artificially small to make examples easier to show.
    label_name='target',
    select_columns=["text","target"],
    num_epochs=1, # Not repeat the dataset
    ignore_errors=True,)
```

```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

👀 Note that we don't need to batch because we did it when we call make_csv_dataset function

# BERT 

Due to BERT is a pretrained encoder model. It was trained in a specific way with an specific preprocessing (See the references above). So it needs its own pre-processing. Luckily, we could Download both the model and the preprocessing pipeline from [Tensorflow Hub](https://www.tensorflow.org/hub).

To see more examples using Tensorflow and BERT visit : https://www.tensorflow.org/text/tutorials/classify_text_with_bert

Before, lets explore a little bit what about BERT in Tensorflow Hub

```python
# Different transformers arquitectures with their preprocessing pipelines in TFHub

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}
```

```python
# Choose one model
bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'

tfhub_handle_encoder = map_name_to_handle[bert_model_name] # Encoder
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name] # Preprocess

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
```

```text
BERT model selected           : https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3
Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
```

```python
# Download preprocess from TFHub
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
```

```python
# Lets explore preprocessing one train dataset sample 
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
sample = train_df.sample(1).text
processed_sample = bert_preprocess_model(sample)
del train_df
gc.collect()
```

```text
0
```

```python
print(sample)
print("=================")
print(processed_sample)
```

```text
1345    Burning buildings? Media outrage? http://t.co/...
Name: text, dtype: object
=================
{'input_type_ids': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>, 'input_word_ids': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[  101,  5255,  3121,  1029,  2865, 19006,  1029,  8299,  1024,
         1013,  1013,  1056,  1012,  2522,  1013, 13569,  2595,  2480,
         2078,  2615,  2487,  6038,   102,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0]], dtype=int32)>, 'input_mask': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>}
```

Look that our preprocess pipeline outputs 3 tensors:

```python
processed_sample.keys()
```

```text
dict_keys(['input_type_ids', 'input_word_ids', 'input_mask'])
```

**Input Ids:** Maybe the most important is the integer encoding (from token to id)

```python
# Integer Embedding by BERT vocabulary
processed_sample["input_word_ids"]
# note:
# 101 : CLS
# 102: SEP
```

```text
<tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[  101,  5255,  3121,  1029,  2865, 19006,  1029,  8299,  1024,
         1013,  1013,  1056,  1012,  2522,  1013, 13569,  2595,  2480,
         2078,  2615,  2487,  6038,   102,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0]], dtype=int32)>
```

**Mask**: To tell bert whats inputs take care of.
This is usefull to tell BERT that we should not take care of padding tokens

```python
processed_sample["input_mask"]
```

```text
<tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>
```

We will not use input_type_ids. Though this is necessary because one of the task which BERT was trained was predict next sentence (For this the input was two sentences) so input_type_ids helps BERT to distinguis between sentence. As we now only have one sentence, it is the same to each token. Please see the references to understand more about it.

**Now we will do one pass trough BERT**

```python
# One pass trough bert 
bert_layer = hub.KerasLayer(tfhub_handle_encoder)
sample_result = bert_layer(processed_sample)
```

```python
sample_result.keys()
```

```text
dict_keys(['encoder_outputs', 'default', 'sequence_output', 'pooled_output'])
```

We note that there is some outputs. Here is a summary of them:

- `pooled_output` represents each input sequence as a whole. The shape is [batch_size, H]. You can think of this as an embedding for the entire twitt.
- `sequence_output` represents each input token in the context. The shape is [batch_size, seq_length, H]. You can think of this as a contextual embedding for every token in the twitt.
- `encoder_outputs` are the intermediate activations of the L Transformer blocks. outputs["encoder_outputs"][i] is a Tensor of shape [batch_size, seq_length, 1024] with the outputs of the i-th Transformer block, for 0 <= i < L. The last value of the list is equal to sequence_output.

# Model

Now we will apply a functional arquitecture model (Instead of sequential as the preovious notebooks)

```python
def bert_model():
    text_input = layers.Input(shape=(), dtype=tf.string, name = "text")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name = 'preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder_layer = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    bert_outputs = encoder_layer(encoder_inputs)
    net = bert_outputs['pooled_output'] # We will use the pooled_output
    net = tf.keras.layers.Dropout(0.1)(net)
    #net = tf.keras.layers.Dense(16, activation="relu", name='dense')(net)
    #net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid", name='output')(net)
    return tf.keras.Model(text_input, net)

model = bert_model()
```

```python
model.summary()
```

```text
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 text (InputLayer)              [(None,)]            0           []

 preprocessing (KerasLayer)     {'input_word_ids':   0           ['text[0][0]']
                                (None, 128),
                                 'input_type_ids':
                                (None, 128),
                                 'input_mask': (Non
                                e, 128)}

 BERT_encoder (KerasLayer)      {'encoder_outputs':  109482241   ['preprocessing[0][0]',
                                 [(None, 128, 768),               'preprocessing[0][1]',
                                 (None, 128, 768),                'preprocessing[0][2]']
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768),
                                 (None, 128, 768)],
                                 'sequence_output':
                                 (None, 128, 768),
                                 'default': (None,
                                768),
                                 'pooled_output': (
                                None, 768)}

 dropout (Dropout)              (None, 768)          0           ['BERT_encoder[0][13]']

 output (Dense)                 (None, 1)            769         ['dropout[0][0]']

==================================================================================================
Total params: 109,483,010
Trainable params: 109,483,009
Non-trainable params: 1
__________________________________________________________________________________________________
```

```python
plot_model(model)
```

![](assets/output-1.png)

```python
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=tf.metrics.BinaryAccuracy(threshold=0.5))
```

🚀🚀🚀🚀 **To be faster, we will use GPU in this notebook!**

```python
early_stop_callback = EarlyStopping(patience = 5)
epochs = 100
history = model.fit(
    train_ds,
    validation_data = test_ds,
    epochs=epochs,
    callbacks = [early_stop_callback]
)
```

```text
Epoch 1/100
191/191 [==============================] - 135s 627ms/step - loss: 0.4655 - binary_accuracy: 0.7924 - val_loss: 0.4008 - val_binary_accuracy: 0.8286
Epoch 2/100
191/191 [==============================] - 118s 619ms/step - loss: 0.3522 - binary_accuracy: 0.8571 - val_loss: 0.4128 - val_binary_accuracy: 0.8339
Epoch 3/100
191/191 [==============================] - 118s 621ms/step - loss: 0.2911 - binary_accuracy: 0.8880 - val_loss: 0.4430 - val_binary_accuracy: 0.8293
Epoch 4/100
191/191 [==============================] - 119s 622ms/step - loss: 0.2310 - binary_accuracy: 0.9169 - val_loss: 0.4897 - val_binary_accuracy: 0.8221
Epoch 5/100
191/191 [==============================] - 119s 620ms/step - loss: 0.1832 - binary_accuracy: 0.9355 - val_loss: 0.5471 - val_binary_accuracy: 0.8102
Epoch 6/100
191/191 [==============================] - 118s 620ms/step - loss: 0.1458 - binary_accuracy: 0.9511 - val_loss: 0.6148 - val_binary_accuracy: 0.8155
```

# Predict Test

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

```python
topred_ds = tf.data.Dataset.from_tensor_slices(test_df.text)
AUTOTUNE = tf.data.AUTOTUNE
topred_ds = topred_ds.batch(32).cache().prefetch(buffer_size=AUTOTUNE)
```

```python
preds = model.predict(topred_ds)
```

```python
# Now we not apply sigmoid function here because or activation function
test_df["target"] = tf.round(preds)
test_df["target"] = test_df["target"].astype(int)
```

```python
sub = test_df[["id", "target"]]
sub
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3258</th>
      <td>10861</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>10865</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>10868</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3261</th>
      <td>10874</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3262</th>
      <td>10875</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3263 rows × 2 columns</p>
</div>
</div>

```python
#sub.to_csv("BERT_submission.csv", index = False)
sub.to_csv("submission.csv", index = False)
```

![image.png](assets/a5bf63e9-1dc2-4c8a-921e-1f354f3b3cf3.png)

# Observations

🎉🎉🎉🎉🎉🎉 🥳🥳🥳🥳🥳🥳

Wow! We improve our score in more than 2%! So we could see the benefits of use transformers! 

There is a lot of pre-trained models. We could try other types. Now we will use HuggingFace🤗 to apply transformers models.

# 4.2 HuggingFace 🤗

It is a framework to easily develop machine learning models and become well known due to their transformers aplications!

To know more about visit : https://huggingface.co. I encourage you do the course in their website!

```python
# Imports 
from transformers import pipeline
```

## HF 🤗 Pipelines - Zero Shot Classification

As you can see in the hf documentation. HuggingFace has pipelines, where we can develop a variety of nlp (and more) applications! (Text generation, classification, summarization, etc)

Now we will do [Zero shot classification](https://en.wikipedia.org/wiki/Zero-shot_learning). That means that we will predict classes that wasn't observe during traning. It is a difficult task

```python
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
```

```text
Downloading:   0%|          | 0.00/1.13k [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/1.52G [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/26.0 [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/878k [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/1.29M [00:00<?, ?B/s]
```

```python
# predict
candidate_labels = ['disaster']
test_df['target'] = [x['scores'][0] for x in classifier(test_df.text.to_list(), candidate_labels)]
```

```python
test_df["target"] = tf.round(test_df.target)
test_df["target"] = test_df["target"].astype(int)
```

```python
sub = test_df[["id", "target"]]
sub
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3258</th>
      <td>10861</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>10865</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>10868</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3261</th>
      <td>10874</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3262</th>
      <td>10875</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3263 rows × 2 columns</p>
</div>
</div>

```python
sub.to_csv("HF_ZS_Submission.csv", index = False)
```

![image.png](assets/d7631cbf-6fad-4494-a240-b3f579cdc252.png)

Make sense because is a zero shot classification!

## Fine-Tune

Now we will download a model from HuggingFace models and finetune with our data

```python
# Model name to fine tune
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
```

```python
from transformers import AutoTokenizer

# First we need to use the tokenizer (Is similar we did in the Tensorflow hub with the preprocessing layer)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])
```

```text
Downloading:   0%|          | 0.00/48.0 [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/629 [00:00<?, ?B/s]
```

```text
Downloading:   0%|          | 0.00/226k [00:00<?, ?B/s]
```

```python
# We need to load our data, for this we use HF datasets (read directly from the disk, so save us ram)
from datasets import load_dataset
data_files = {"train": "/kaggle/input/df-split/df_split/df_train.csv",
             "test":"/kaggle/input/df-split/df_split/df_test.csv"}
dataset = load_dataset("csv", data_files = data_files, usecols = ['text', 'target'])
```

```text
Downloading and preparing dataset csv/default to /root/.cache/huggingface/datasets/csv/default-1860883215d5579d/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519...
```

```text
Downloading data files:   0%|          | 0/2 [00:00<?, ?it/s]
```

```text
Extracting data files:   0%|          | 0/2 [00:00<?, ?it/s]
```

```text
Dataset csv downloaded and prepared to /root/.cache/huggingface/datasets/csv/default-1860883215d5579d/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519. Subsequent calls will reuse this data.
```

```text
  0%|          | 0/2 [00:00<?, ?it/s]
```

```python
dataset
```

```text
DatasetDict({
    train: Dataset({
        features: ['text', 'target'],
        num_rows: 6090
    })
    test: Dataset({
        features: ['text', 'target'],
        num_rows: 1523
    })
})
```

```python
# preprocess
dataset = dataset.map(tokenize_dataset)
```

```text
  0%|          | 0/6090 [00:00<?, ?ex/s]
```

```text
  0%|          | 0/1523 [00:00<?, ?ex/s]
```

```python
train_dataset = dataset['train']
test_dataset = dataset['test']
```

```python
# we transform the HF dataset into TF dataset to fine tune
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
tf_train_dataset = train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["target"],
    batch_size=16,
    collate_fn=data_collator,
    shuffle=True)

tf_val_dataset = test_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["target"],
    batch_size=16,
    collate_fn=data_collator,
    shuffle=True
    )
```

## Model

```python
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5),metrics = ['accuracy'])

early_stop_callback = EarlyStopping(patience = 5)
model.fit(tf_train_dataset,
         validation_data = tf_val_dataset,
          epochs = 20,
          callbacks = [early_stop_callback]
         )
```

```text
Downloading:   0%|          | 0.00/256M [00:00<?, ?B/s]
```

```text
2022-12-12 03:54:13.603300: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
Some layers from the model checkpoint at distilbert-base-uncased-finetuned-sst-2-english were not used when initializing TFDistilBertForSequenceClassification: ['dropout_19']
- This IS expected if you are initializing TFDistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFDistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some layers of TFDistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased-finetuned-sst-2-english and are newly initialized: ['dropout_20']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
No loss specified in compile() - the model's internal loss computation will be used as the loss. Don't panic - this is a common way to train TensorFlow models in Transformers! To disable this behaviour please pass a loss argument, or explicitly pass `loss=None` if you do not want your model to compute a loss.
Epoch 1/20
380/380 [==============================] - 44s 87ms/step - loss: 0.4815 - accuracy: 0.7913 - val_loss: 0.3999 - val_accuracy: 0.8270
Epoch 2/20
380/380 [==============================] - 31s 83ms/step - loss: 0.3179 - accuracy: 0.8750 - val_loss: 0.4339 - val_accuracy: 0.8164
Epoch 3/20
380/380 [==============================] - 31s 83ms/step - loss: 0.2035 - accuracy: 0.9238 - val_loss: 0.4884 - val_accuracy: 0.8053
Epoch 4/20
380/380 [==============================] - 31s 83ms/step - loss: 0.1147 - accuracy: 0.9572 - val_loss: 0.5929 - val_accuracy: 0.7967
Epoch 5/20
380/380 [==============================] - 32s 83ms/step - loss: 0.0829 - accuracy: 0.9729 - val_loss: 0.7295 - val_accuracy: 0.7717
Epoch 6/20
380/380 [==============================] - 31s 82ms/step - loss: 0.0579 - accuracy: 0.9760 - val_loss: 0.9569 - val_accuracy: 0.7855
```

```text
<keras.callbacks.History at 0x7f06a863c950>
```

### Predict Test

```python
pred_dataset = load_dataset("csv", data_files = "/kaggle/input/nlp-getting-started/test.csv", usecols = ['text'])
```

```text
Downloading and preparing dataset csv/default to /root/.cache/huggingface/datasets/csv/default-e12e02ba714b9048/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519...
```

```text
Downloading data files:   0%|          | 0/1 [00:00<?, ?it/s]
```

```text
Extracting data files:   0%|          | 0/1 [00:08<?, ?it/s]
```

```text
Dataset csv downloaded and prepared to /root/.cache/huggingface/datasets/csv/default-e12e02ba714b9048/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519. Subsequent calls will reuse this data.
```

```text
  0%|          | 0/1 [00:00<?, ?it/s]
```

```python
# preprocess
pred_dataset = pred_dataset.map(tokenize_dataset)
```

```text
  0%|          | 0/3263 [00:00<?, ?ex/s]
```

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
tf_pred_dataset = pred_dataset['train'].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    batch_size=16,
    collate_fn=data_collator,
    shuffle=False)
```

```python
preds = model.predict(tf_pred_dataset)
```

```python
preds = np.argmax(preds['logits'], axis = 1)
```

```python
test_df["target"] = preds
```

```python
sub = test_df[["id", "target"]]
sub
```

<div class="nb-output">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3258</th>
      <td>10861</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>10865</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>10868</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3261</th>
      <td>10874</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3262</th>
      <td>10875</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3263 rows × 2 columns</p>
</div>
</div>

```python
sub.to_csv("DistilBERT_FT_Submission.csv", index = False)
```

![image.png](assets/4e5219a2-c10d-4596-b616-3ddfebaa305d.png)

## Observations
Not bad! Note that we don't tune much this model, so probably we could improve it easily!

The important point is how easy is use hub models from huggingface and finetune. If you look the website, you can see that there are a lot of datasets and models made by de comunnity!

In the next notebook I will do my best combining all the stuff that we saw in this notebooks NLP series to get a better result. Also I will pre-process the data before feed it into the model
