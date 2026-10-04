# BirdClef 2023 🐦 — Bird Sound Classification

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/diegomachado/birdclef2023)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/diegulio/birdclef2023)

![BirdClef Competition](birdclef.png)

# 🐦 Tópico: Audio Classification con Machine Learning

En este post resolveremos un problema de **audio con Machine Learning**. Este es un tipo de problema muy interesante ya que el audio, al igual que las imágenes, se puede representar de formas muy ricas. La competición [BirdCLEF 2023](https://www.kaggle.com/competitions/birdclef-2023) de Kaggle nos desafía a identificar especies de aves a partir de grabaciones de sus cantos.

La metodología más común para clasificar audios con Deep Learning es convertir el audio a una representación visual llamada **espectrograma** y luego aplicar modelos de Computer Vision. ¡Los audios se convierten en imágenes!

# 🔎 Motivación: BirdCLEF 2023

Esta competencia en Kaggle propone identificar especies de aves nativas de Hawaii (muchas en peligro de extinción) mediante el análisis de sus cantos. La biodiversidad de aves en Hawaii está severamente amenazada, y sistemas automáticos de identificación pueden ayudar a conservacionistas a monitorear poblaciones en campo.

El objetivo es construir un modelo que, dado un fragmento de audio, prediga qué especie de ave está cantando de entre 264 clases posibles.

# 🔨 Tool Path: Que utilizaremos

1. **PyTorch + Pytorch Lightning**: Para el entrenamiento del modelo
2. **Torchaudio**: Para procesar audio con PyTorch
3. **Librosa**: Biblioteca de análisis de audio en Python
4. **EfficientNet (Timm)**: Modelo pre-entrenado de Computer Vision
5. **Kaggle**: Plataforma de la competencia

# 💭 Concept Path: Que aprenderemos

1. Procesamiento de señales de audio
2. Transformada de Fourier y STFT
3. Espectrogramas (Mel Spectrograms)
4. Data Augmentation para audio
5. Transfer Learning para clasificación de audio

# ♟️ Estrategia: Como abordamos

La estrategia principal es tratar el problema de audio como un problema de **Computer Vision**:

```
Audio (waveform) → MelSpectrogram (imagen) → EfficientNet → Predicción de especie
```

# 🧠 Prototyping

## Librerías

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import torchaudio
import torch
import timm
import lightning as L
```

## CFG (Configuración)

```python
class CFG:
    seed = 13
    
    # Audio Settings
    DURATION = 5         # seconds
    FRAME_SIZE = 1024
    HOP_SIZE = 512
    SR = 22050           # Sample rate
    N_MELS = 128         # Número de bandas mel
    
    # Model
    MODEL = 'efficientnet_b0'
    NUM_CLASSES = 264
    
    # Training
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 10
    ACCELERATOR = 'gpu'
```

## EDA: Entendiendo los audios

Para tener un espectrograma, necesitamos calcular las variaciones de frecuencias a través del tiempo. Una metodología muy conocida es la **Transformada de Fourier**. Pero para señales de audio necesitamos una versión que funcione en segmentos temporales: la **STFT (Short-Time Fourier Transform)**.

### De audio a espectrograma

```python
import librosa
import numpy as np

# Cargar audio
audio, sr = librosa.load(audio_path, sr=CFG.SR, duration=CFG.DURATION)

# STFT
stft = librosa.stft(audio, n_fft=CFG.FRAME_SIZE, hop_length=CFG.HOP_SIZE)

# Magnitud (los números complejos de la STFT los convertimos a magnitud)
magnitude = np.abs(stft)

# Log-Amplitude (convertir a decibeles)
log_magnitude = librosa.amplitude_to_db(magnitude)
```

### Mel Spectrogram

El **Mel Spectrogram** aplica una escala de frecuencia basada en la percepción humana del sonido, lo que lo hace especialmente útil para audio biológico:

```python
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=CFG.FRAME_SIZE,
    hop_length=CFG.HOP_SIZE,
    n_mels=CFG.N_MELS
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

## Data Augmentation

Para mejorar la generalización del modelo, aplicamos **Data Augmentation** tanto a nivel de señal de audio como a nivel de espectrograma:

### Signal Augmentations

**Gaussian Noise**: Agrega ruido aleatorio para simular condiciones ambientales:

```python
def add_gaussian_noise(audio, min_amplitude=0.001, max_amplitude=0.015):
    amplitude = np.random.uniform(min_amplitude, max_amplitude)
    noise = np.random.randn(len(audio))
    return audio + amplitude * noise
```

**Time Stretching**: Cambia la velocidad del audio sin afectar el pitch:

```python
def time_stretch(audio, rate=None):
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(audio, rate=rate)
```

### Spectrogram Augmentations

**Frequency Masking**: Enmascara rangos de frecuencia aleatoriamente:

```python
def frequency_masking(spec, freq_mask_param=30):
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, spec.shape[0] - f)
    spec[f0:f0+f, :] = 0
    return spec
```

**Time Masking**: Enmascara segmentos temporales aleatoriamente:

```python
def time_masking(spec, time_mask_param=40):
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, spec.shape[1] - t)
    spec[:, t0:t0+t] = 0
    return spec
```

## Load Data — Pytorch Dataset

```python
class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, df, audio_dir, transform=None, augment=False):
        self.df = df
        self.audio_dir = audio_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['filename'])
        
        # Leer audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze().numpy()
        
        # Resample si es necesario
        if sr != CFG.SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CFG.SR)
        
        # Data Augmentation (solo en train)
        if self.augment:
            audio = add_gaussian_noise(audio)
        
        # MelSpectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=CFG.SR,
            n_fft=CFG.FRAME_SIZE, hop_length=CFG.HOP_SIZE,
            n_mels=CFG.N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convertir a tensor (3 canales para EfficientNet)
        image = torch.tensor(mel_spec_db).unsqueeze(0).repeat(3, 1, 1)
        
        if self.transform:
            image = self.transform(image)
        
        label = row['label_id']
        return image, label
```

### Dataloader

```python
train_dataset = BirdDataset(train_df, AUDIO_DIR, augment=True)
val_dataset = BirdDataset(val_df, AUDIO_DIR, augment=False)

train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)
```

## Modeling — EfficientNet + Pytorch Lightning

```python
class BirdModel(L.LightningModule):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        return [optimizer], [scheduler]
```

## MixUp

Una técnica aprendida de la comunidad Kaggle es **MixUp**, que combina dos señales de audio para crear nueva data sintética:

```python
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## Entrenamiento

```python
trainer = L.Trainer(
    max_epochs=CFG.EPOCHS,
    accelerator=CFG.ACCELERATOR,
    devices=1,
    precision=16,
)

model = BirdModel(num_classes=CFG.NUM_CLASSES)
trainer.fit(model, train_loader, val_loader)
```

# 🎯 Resultados

El modelo logró un buen performance en la competencia considerando que:
- Se usó EfficientNet-B0 (modelo relativamente pequeño)
- Solo 10 epochs de entrenamiento
- Sin usar la metadata adicional disponible

La técnica de convertir audio a MelSpectrogramas demostró ser muy efectiva. Los modelos de Computer Vision pre-entrenados en ImageNet logran aprender patrones acústicos representados visualmente.

# 🚀 Próximos Pasos

- Usar **metadata** (duración, tipo de grabación) como features adicionales
- Probar modelos más grandes: EfficientNet-B3, B5
- Implementar **pseudo-labeling** con datos sin etiquetar
- Experimentar con más técnicas de Data Augmentation para audio
- Usar **ensemble** de múltiples modelos

# 🥳 Conclusión

Este post nos llevó a explorar un área emocionante: **clasificación de audio con Deep Learning**. La clave fue convertir el problema de audio en uno de visión mediante MelSpectrogramas, permitiéndonos aprovechar el poder de los modelos pre-entrenados como EfficientNet.

Técnicas como Data Augmentation (Gaussian Noise, Frequency/Time Masking) y MixUp nos permiten mejorar la robustez del modelo con datos limitados. ¡La creatividad en preprocesamiento hace la diferencia en Kaggle!
