from keras.models import Sequential
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
import numpy as np
from tensorflow.keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.text import Tokenizer



import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model1.h5')
x_train = [[[4], [2], [1], [2]]]
x_train1 = [[[4], [7], [1], [2]]]
x_train_padded = pad_sequences(x_train, maxlen=1024)
x_train1_padded = pad_sequences(x_train1, maxlen=1024)

pro = model.predict([x_train1_padded, x_train_padded])
print(pro)

