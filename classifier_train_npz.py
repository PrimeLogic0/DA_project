import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Valori globali per la normalizzazione
min_global = -1460
max_global = 1460

# Funzione per caricare i dati da una cartella e prendere solo le diagonali principali
def load_data(data_dir):
    X = []
    y = []
    class_labels = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_labels):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npz'):
                    file_path = os.path.join(class_path, file_name)
                    with np.load(file_path) as data:
                        # Caricamento della matrice
                        key = list(data.keys())[0]  # Supponendo che la matrice sia la prima chiave
                        matrix = data[key]
                        # Estrai la diagonale principale
                        diagonal = np.diag(matrix.squeeze())  # Rimuove dimensioni extra e prende la diagonale
                        X.append(diagonal)
                        y.append(label)
    return np.array(X), np.array(y)

# Funzione per normalizzare i dati
def normalize_data(data):
    """
    Normalizza i dati dall'intervallo originale [0, 255] a [-1, 1] usando i limiti globali.
    """
    normalized_data = ((data / 255.0) * 2) - 1
    denormalized_data = normalized_data * (max_global - min_global) + min_global
    return denormalized_data

# Percorso ai dati di training
data_dir = "./distribuzioni/npz_image_folder"

# Caricamento dei dati
X, y = load_data(data_dir)

# Normalizzazione dei dati
X = normalize_data(X)

# Aggiunta del canale ai dati (necessario per Conv1D)
X = np.expand_dims(X, axis=-1)

# Conversione delle etichette in one-hot encoding
y = to_categorical(y, num_classes=16)

# Suddivisione in training e validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcolo dei pesi delle classi
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
class_weights = dict(enumerate(class_weights))

# Definizione del modello
model = Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(10, 1)),  # Conv1D per sequenze
    BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(16, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
batch_size = 32
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    class_weight=class_weights
)

# Salvataggio del modello
model.save("best_npz_classifier_model.keras")
print("Modello salvato come 'best_npz_classifier_model.keras'")
