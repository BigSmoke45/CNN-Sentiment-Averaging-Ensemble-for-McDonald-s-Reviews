import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# Завантаження даних
data = pd.read_csv("McDonald_s_Reviews.csv", encoding='latin1')

# Попередня обробка даних
data['rating'] = data['rating'].apply(lambda x: 1 if x.startswith('5') else 0)  # Позитивний: 1, Негативний: 0

# Розподіл даних на тренувальний та тестовий набори
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Підготовка тексту для моделі
max_words = 10000  # Максимальна кількість слів для токенізатора
max_len = 100  # Максимальна довжина відгуку
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train['review'])
X_train = pad_sequences(tokenizer.texts_to_sequences(train['review']), maxlen=max_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(test['review']), maxlen=max_len)

# Перетворення міток класів
encoder = LabelEncoder()
encoder.fit(train['rating'])
y_train = encoder.transform(train['rating'])
y_test = encoder.transform(test['rating'])

# Параметри моделі
embedding_dim = 100
num_filters = 128
kernel_size = 5
dropout_rate = 0.5

# Створення ансамблю моделей
num_models = 3  # Кількість моделей в ансамблі
models = []
for i in range(num_models):
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Conv1D(num_filters, kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stop])
    models.append(model)

# Створення ансамблю методом усереднення
ensemble_predictions = np.zeros((len(X_test), 1))
for model in models:
    ensemble_predictions += model.predict(X_test)
ensemble_predictions /= num_models

# Визначення класів на основі усереднених прогнозів
ensemble_predictions = np.where(ensemble_predictions > 0.5, 1, 0)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print("Точність ансамблю на тестовій вибірці:", ensemble_accuracy)

# Збереження об'єкта токенізатора
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
