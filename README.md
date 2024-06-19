Importing Libraries
import pandas as pd
import numpy as np

import os
import sys

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from IPython.display import Audio

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
Dataset
Crema = "/kaggle/input/cremad/AudioWAV/"
crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()
Emotions	Path
0	disgust	/kaggle/input/cremad/AudioWAV/1028_TSI_DIS_XX.wav
1	happy	/kaggle/input/cremad/AudioWAV/1075_IEO_HAP_LO.wav
2	happy	/kaggle/input/cremad/AudioWAV/1084_ITS_HAP_XX.wav
3	disgust	/kaggle/input/cremad/AudioWAV/1067_IWW_DIS_XX.wav
4	disgust	/kaggle/input/cremad/AudioWAV/1066_TIE_DIS_XX.wav
plt.title('Count of Emotions', size=16)
sns.countplot(Crema_df.Emotions)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()

Visualization
def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for {} emotion'.format(e), size=15)
    librosa.display.waveplot(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    plt.colorbar()
emotion='angry'
path = np.array(Crema_df.Path[Crema_df.Emotions==emotion])[0]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
Audio(path)


MFCC Extraction
labels = {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
Crema_df.replace({'Emotions':labels},inplace=True)
num_mfcc=13
n_fft=2048
hop_length=512
SAMPLE_RATE = 22050
data = {
        "labels": [],
        "mfcc": []
    }
for i in range(7442):
    data['labels'].append(Crema_df.iloc[i,0])
    signal, sample_rate = librosa.load(Crema_df.iloc[i,1], sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    data["mfcc"].append(np.asarray(mfcc))
    if i%500==0:
        print(i)
0
500
1000
1500
2000
2500
3000
3500
4000
4500
5000
5500
6000
6500
7000
Padding MFCC to make them of equal length
X = np.asarray(data['mfcc'])
y = np.asarray(data["labels"])
X = tf.keras.preprocessing.sequence.pad_sequences(X)
X.shape
(7442, 216, 13)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
print(X_train.shape,y_train.shape,X_validation.shape,y_validation.shape,X_test.shape,y_test.shape)
(5357, 216, 13) (5357,) (1340, 216, 13) (1340,) (745, 216, 13) (745,)
Model
def build_model(input_shape):
    model = tf.keras.Sequential()

    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(6, activation='softmax'))

    return model
# create network
input_shape = (None,13)
model = build_model(input_shape)

# compile model
optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, None, 128)         72704     
_________________________________________________________________
lstm_1 (LSTM)                (None, 64)                49408     
_________________________________________________________________
dense (Dense)                (None, 64)                4160      
_________________________________________________________________
dropout (Dropout)            (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 390       
=================================================================
Total params: 126,662
Trainable params: 126,662
Non-trainable params: 0
_________________________________________________________________
Training
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
Epoch 1/30
168/168 [==============================] - 9s 29ms/step - loss: 1.6471 - accuracy: 0.2998 - val_loss: 1.4173 - val_accuracy: 0.4373
Epoch 2/30
168/168 [==============================] - 4s 23ms/step - loss: 1.5058 - accuracy: 0.3737 - val_loss: 1.4070 - val_accuracy: 0.4388
Epoch 3/30
168/168 [==============================] - 4s 24ms/step - loss: 1.4647 - accuracy: 0.4006 - val_loss: 1.3801 - val_accuracy: 0.4515
Epoch 4/30
168/168 [==============================] - 4s 22ms/step - loss: 1.4572 - accuracy: 0.4115 - val_loss: 1.3795 - val_accuracy: 0.4507
Epoch 5/30
168/168 [==============================] - 4s 23ms/step - loss: 1.4235 - accuracy: 0.4205 - val_loss: 1.3509 - val_accuracy: 0.4485
Epoch 6/30
168/168 [==============================] - 4s 22ms/step - loss: 1.3981 - accuracy: 0.4334 - val_loss: 1.3364 - val_accuracy: 0.4582
Epoch 7/30
168/168 [==============================] - 4s 22ms/step - loss: 1.3942 - accuracy: 0.4391 - val_loss: 1.3362 - val_accuracy: 0.4642
Epoch 8/30
168/168 [==============================] - 4s 23ms/step - loss: 1.3706 - accuracy: 0.4609 - val_loss: 1.2877 - val_accuracy: 0.4761
Epoch 9/30
168/168 [==============================] - 4s 22ms/step - loss: 1.3394 - accuracy: 0.4601 - val_loss: 1.3043 - val_accuracy: 0.4664
Epoch 10/30
168/168 [==============================] - 4s 22ms/step - loss: 1.3195 - accuracy: 0.4828 - val_loss: 1.2969 - val_accuracy: 0.4679
Epoch 11/30
168/168 [==============================] - 4s 23ms/step - loss: 1.3135 - accuracy: 0.4834 - val_loss: 1.2508 - val_accuracy: 0.5090
Epoch 12/30
168/168 [==============================] - 4s 23ms/step - loss: 1.2816 - accuracy: 0.5153 - val_loss: 1.2401 - val_accuracy: 0.5164
Epoch 13/30
168/168 [==============================] - 4s 22ms/step - loss: 1.2723 - accuracy: 0.5016 - val_loss: 1.2301 - val_accuracy: 0.5201
Epoch 14/30
168/168 [==============================] - 4s 23ms/step - loss: 1.2353 - accuracy: 0.5257 - val_loss: 1.2099 - val_accuracy: 0.5194
Epoch 15/30
168/168 [==============================] - 4s 22ms/step - loss: 1.2186 - accuracy: 0.5198 - val_loss: 1.1667 - val_accuracy: 0.5530
Epoch 16/30
168/168 [==============================] - 4s 23ms/step - loss: 1.1892 - accuracy: 0.5366 - val_loss: 1.1892 - val_accuracy: 0.5381
Epoch 17/30
168/168 [==============================] - 4s 23ms/step - loss: 1.1847 - accuracy: 0.5515 - val_loss: 1.1699 - val_accuracy: 0.5254
Epoch 18/30
168/168 [==============================] - 4s 22ms/step - loss: 1.1964 - accuracy: 0.5474 - val_loss: 1.1640 - val_accuracy: 0.5396
Epoch 19/30
168/168 [==============================] - 4s 23ms/step - loss: 1.1418 - accuracy: 0.5676 - val_loss: 1.1774 - val_accuracy: 0.5515
Epoch 20/30
168/168 [==============================] - 4s 24ms/step - loss: 1.0904 - accuracy: 0.5906 - val_loss: 1.1691 - val_accuracy: 0.5582
Epoch 21/30
168/168 [==============================] - 4s 22ms/step - loss: 1.0924 - accuracy: 0.5863 - val_loss: 1.1475 - val_accuracy: 0.5612
Epoch 22/30
168/168 [==============================] - 4s 23ms/step - loss: 1.0912 - accuracy: 0.5859 - val_loss: 1.1340 - val_accuracy: 0.5627
Epoch 23/30
168/168 [==============================] - 4s 23ms/step - loss: 1.0430 - accuracy: 0.6085 - val_loss: 1.1479 - val_accuracy: 0.5493
Epoch 24/30
168/168 [==============================] - 4s 22ms/step - loss: 1.0264 - accuracy: 0.6117 - val_loss: 1.0858 - val_accuracy: 0.5851
Epoch 25/30
168/168 [==============================] - 4s 23ms/step - loss: 1.0153 - accuracy: 0.6116 - val_loss: 1.0996 - val_accuracy: 0.5687
Epoch 26/30
168/168 [==============================] - 4s 22ms/step - loss: 0.9915 - accuracy: 0.6299 - val_loss: 1.1131 - val_accuracy: 0.5821
Epoch 27/30
168/168 [==============================] - 4s 22ms/step - loss: 0.9779 - accuracy: 0.6221 - val_loss: 1.1090 - val_accuracy: 0.5627
Epoch 28/30
168/168 [==============================] - 4s 23ms/step - loss: 0.9417 - accuracy: 0.6419 - val_loss: 1.0965 - val_accuracy: 0.5858
Epoch 29/30
168/168 [==============================] - 4s 24ms/step - loss: 0.9348 - accuracy: 0.6440 - val_loss: 1.1353 - val_accuracy: 0.5709
Epoch 30/30
168/168 [==============================] - 4s 23ms/step - loss: 0.9239 - accuracy: 0.6497 - val_loss: 1.1182 - val_accuracy: 0.5903
Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: ",test_acc)
Test Accuracy:  0.563758373260498
model.save('Speech-Emotion-Recognition-Model.h5')
