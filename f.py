from keras.models import Sequential
import numpy as np
from keras.layers import Embedding
from keras.models import Sequential, Model
import numpy as np
from tensorflow.keras.callbacks import Callback

from sklearn.model_selection import StratifiedKFold
import numpy as np
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
from keras.preprocessing.text import Tokenizer
from keras import backend as K





# Загрузка модели
model = load_model('model1.h5')

# Создание новой модели с отдельной частью
new_model = Model(inputs=model.input, outputs=model.get_layer('output_dim1').output)
x_train = [[[4], [2], [1], [2]]]
x_train1 = [[]]
x_train_padded = pad_sequences(x_train, maxlen=1024)
x_train1_padded = pad_sequences(x_train1, maxlen=1024)

pro = new_model.predict([x_train1_padded, x_train_padded])
print(pro)
