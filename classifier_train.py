import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Suddivisione in train e validation
from sklearn.model_selection import train_test_split

# Percorso ai dati
data_dir = "npz_image_folder/"

# Parametri
img_height = 10
img_width = 10
batch_size = 32
epochs = 100  # Numero massimo di epoche
num_classes = 16  # Numero di classi

# Funzione per caricare i dati da file .npz
def load_npz_dataset(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Assumendo che ogni cartella corrisponda a una classe
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npz'):
                    file_path = os.path.join(class_path, file_name)
                    data = np.load(file_path)
                    if 'immagine.npy' in data:  # Caricamento del primo array salvato nel file .npz
                        image = data['immagine.npy']
                        if image.shape == (img_height, img_width, 1):
                            images.append(image)
                            labels.append(class_to_idx[class_name])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels, class_names

# Caricamento del dataset
images, labels, class_names = load_npz_dataset(data_dir)

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=123)

# Creazione dei dataset TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Preprocessing e ottimizzazione
autotune = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1)  # Aggiunge il canale (da 10x10 a 10x10x1)
    return image, label

train_dataset = (train_dataset
    .map(preprocess, num_parallel_calls=autotune)
    .shuffle(buffer_size=1000)
    .batch(batch_size)
    .prefetch(buffer_size=autotune))

val_dataset = (val_dataset
    .map(preprocess, num_parallel_calls=autotune)
    .batch(batch_size)
    .prefetch(buffer_size=autotune))

# Creazione del modello
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_loss', save_best_only=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint]
)
