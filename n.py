from keras.models import Sequential
import numpy as np
from keras.layers import Embedding
from keras.models import Sequential
import numpy as np
from tensorflow.keras.callbacks import Callback

from sklearn.model_selection import StratifiedKFold
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras import backend as K

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, Input, Lambda, concatenate, ReLU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences

def comparison(tensor):

    x, y = tensor[0], tensor[1]
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):

        print(f'Эпоха [{epoch}], Потеря: {logs["loss"]:.4f}')

    def set_model(self, model):
        self.model = model
        # ваш код
        return


model1 = Sequential([
    tf.keras.layers.Embedding(input_dim=10, output_dim=512, input_length=1024, trainable=False, name='input_dim1'),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(256)),
    Dense(32, activation='sigmoid', name='output_dim1')
])



model2 = Sequential([
    tf.keras.layers.Embedding(input_dim=10, output_dim=512, input_length=1024, trainable=False, name='input_dim2'),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(256)),
    Dense(32, activation='sigmoid', name='output_dim2')
])

combined = Lambda(comparison)([model1.output, model2.output])





model = Model(inputs=[model1.input, model2.input], outputs=combined)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = [[[4], [2], [1], [2]]]
x_train1 = [[[4], [7], [1], [2]]]
x_train_padded = pad_sequences(x_train, maxlen=1024)
x_train1_padded = pad_sequences(x_train1, maxlen=1024)
y_train = [[1]]
x_train_padded = np.array(x_train_padded)
x_train1_padded = np.array(x_train1_padded)
y_train = np.array(y_train)
print([x_train_padded, x_train1_padded])
model.fit([x_train_padded, x_train1_padded], y_train)

model.save('model1.h5')
