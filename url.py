import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

df = pd.read_csv('urldata.csv')
df = df.drop(df['result'])
label_encoder = LabelEncoder()
#df['type_encoded'] = label_encoder.fit_transform(df['type'])
df['type_encoded'] = label_encoder.fit_transform(df['label'])

# Tokenize and pad the URLs
max_len = 100  # adjust based on your dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['url'])
X = tokenizer.texts_to_sequences(df['url'])
X = pad_sequences(X, maxlen=max_len)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['type_encoded'], test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax'))  # Output layer

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

epochs = 5  # You can adjust this based on the training performance
batch_size = 32

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,callbacks=early_stopping)

accuracy = model.evaluate(X_test, y_test)[1]
print(f'Model Accuracy: {accuracy * 100:.2f}%')

#model.save("url_detector.h5")
#np.save('url_checker.npy', tokenizer.word_index)

model.save("url_detection.h5")
np.save('url_check.npy', tokenizer.word_index)
