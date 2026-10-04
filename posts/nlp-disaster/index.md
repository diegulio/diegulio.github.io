# Identificando desastres en Twitter con NLP 🌪️

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/diegomachado/seqclass-nn-embed-rnn-lstm-gru-bert-hf)

![Twitter Disaster](twitter_disaster.png)

# 🎯 Goal

Estoy aprendiendo NLP. Para hacerlo decidí pasar por diversos modelos de NLP, estudiar la teoría e implementarlos — desde redes neuronales simples hasta BERT. Creo que es un buen método para aprender Machine Learning: no sólo leer, sino ensuciarse las manos con el código.

**Disclaimer**: Todo el contenido en los notebooks es lo que entendí de diversas referencias. Si encuentras algún error por favor avísame para poder aprender de ello. 🙋🏽

# 📚 Tópico: Natural Language Processing

La competición [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) de Kaggle propone un problema de clasificación binaria: dado un tweet, determinar si está describiendo un **desastre real** o si usa el lenguaje de desastre de forma **metafórica/figurada**.

Por ejemplo:
- Tweet real de desastre: *"There's an emergency evacuation in my city due to wildfires 🔥"*
- Tweet metafórico: *"My exam was a total disaster lol"*

# 🔎 Motivación

Twitter (ahora X) es una de las fuentes de información en tiempo real más importantes del mundo. Organizaciones de respuesta a emergencias, periodistas y ciudadanos usan esta plataforma para reportar desastres. Sin embargo, distinguir automáticamente tweets reales de desastre es un desafío no trivial — el lenguaje humano es ambiguo y contextual.

🧠 **Solución: Aplicar técnicas de NLP progresivamente más complejas para clasificar tweets de desastre.**

# 🔨 Tool Path: Que utilizaremos

1. **TensorFlow / Keras**: Para modelos de redes neuronales
2. **HuggingFace Transformers**: Para modelos BERT
3. **PyTorch**: Para implementaciones más avanzadas
4. **NLTK / SpaCy**: Para preprocesamiento de texto

# 💭 Concept Path: Que aprenderemos

Este proyecto nos lleva por una progresión de técnicas NLP modernas:

1. **Simple Neural Network** — Baseline básico
2. **Word Embeddings** — Representación semántica del texto
3. **RNN / LSTM / GRU** — Modelos secuenciales
4. **BERT (HuggingFace)** — Transformers y la mejor solución

# 🧠 EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data

```python
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print(train_df.head())
print(f"Train shape: {train_df.shape}")
print(f"Disaster tweets: {train_df.target.sum()}")
print(f"Non-disaster tweets: {(train_df.target == 0).sum()}")
```

```
   id keyword location                                               text  target
0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...       1
1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada       1
2   5     NaN      NaN  All residents asked to 'shelter in place' are ...       1
```

El dataset está **bastante balanceado**: ~57% no-desastre, ~43% desastre.

## Pre-processing

```python
import re
import string

def clean_text(text):
    # Remover URLs
    text = re.sub(r'http\S+', '', text)
    # Remover menciones  
    text = re.sub(r'@\w+', '', text)
    # Remover hashtags (conservar la palabra)
    text = re.sub(r'#', '', text)
    # Remover puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower().strip()
    return text

train_df['clean_text'] = train_df['text'].apply(clean_text)
```

# ⚡️ Modelos

## 1. Simple Neural Network (Baseline)

El primer enfoque es una NN simple con representación **Bag of Words**:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenización
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean_text'])

X_train = tokenizer.texts_to_sequences(train_df['clean_text'])
X_train = pad_sequences(X_train, maxlen=100, padding='post', truncating='post')

# Modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_split=0.2)
```

## 2. Word Embeddings

Los embeddings representan palabras como vectores en un espacio semántico continuo. Palabras similares tienden a tener vectores cercanos:

```python
from tensorflow.keras.layers import Embedding

# Pre-trained GloVe Embeddings
def load_glove_embeddings(glove_file, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

embedding_matrix = load_glove_embeddings('glove.6B.100d.txt', tokenizer.word_index)

model = tf.keras.Sequential([
    Embedding(len(tokenizer.word_index) + 1, 100, 
              weights=[embedding_matrix], 
              input_length=100, 
              trainable=False),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## 3. RNN / LSTM / GRU

Los modelos recurrentes capturan el **orden secuencial** del texto, algo que los embeddings simples no hacen:

```python
# LSTM
model_lstm = tf.keras.Sequential([
    Embedding(10000, 64, input_length=100),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# GRU (más eficiente que LSTM)
model_gru = tf.keras.Sequential([
    Embedding(10000, 64, input_length=100),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.GRU(32),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

Diferencia clave entre LSTM y GRU:
- **LSTM**: Tiene "forget gate", "input gate" y "output gate" — más expresivo pero más lento
- **GRU**: Simplifica con solo "reset gate" y "update gate" — más rápido con resultados similares

## 4. BERT (La Solución Ganadora)

BERT (Bidirectional Encoder Representations from Transformers) es un modelo pre-entrenado de HuggingFace que logra el mejor performance:

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_examples(ds, limit=-1):
    input_ids_list = []
    attention_mask_list = []
    label_list = []
    
    if limit > 0:
        ds = ds.take(limit)
    
    for text, label in zip(ds['clean_text'], ds['target']):
        bert_input = tokenizer_bert(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    
    return (
        tf.squeeze(tf.stack(input_ids_list), axis=1),
        tf.squeeze(tf.stack(attention_mask_list), axis=1),
        tf.stack(label_list)
    )

model_bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model_bert.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

> [!NOTE]
> La diferencia clave de BERT vs los modelos anteriores es que es **bidireccional**: entiende el contexto de cada palabra mirando tanto lo que viene antes como lo que viene después. Esto es fundamental para entender el lenguaje humano con sus ambigüedades.

# 🎯 Resultados

| Modelo | Accuracy Validación | F1-Score |
|--------|-------------------|----------|
| Simple NN | ~78% | ~0.76 |
| GloVe Embeddings | ~80% | ~0.79 |
| LSTM | ~81% | ~0.80 |
| GRU | ~81% | ~0.80 |
| **BERT** | **~84%** | **~0.83** |

BERT supera significativamente a los modelos anteriores gracias a su representación contextual bidireccional y su preentrenamiento en grandes corpus de texto.

# 🚀 Próximos Pasos

- Fine-tuning de **RoBERTa** o **DeBERTa** para mayor performance
- Explorar **ensemble** de BERT + LSTM
- Aplicar técnicas de **data cleaning** más agresivas
- Usar **knowledge distillation** para comprimir BERT en un modelo más pequeño

# 🥳 Conclusión

Este proyecto fue un gran viaje por el ecosistema de NLP moderno. Comenzamos con redes neuronales simples y step by step llegamos a BERT, experimentando cómo cada técnica mejora nuestra capacidad de entender el lenguaje humano.

La progresión Simple NN → Embeddings → LSTM/GRU → BERT ilustra perfectamente la evolución del campo de NLP en los últimos años. Cada paso resuelve limitaciones del anterior:

- **Simple NN**: No entiende semántica
- **Embeddings**: Captura semántica pero no orden
- **LSTM/GRU**: Captura orden pero ve el texto en una sola dirección
- **BERT**: Contexto bidireccional completo con preentrenamiento masivo

¡El código completo está en Kaggle — te invito a explorar y mejorar la solución!
