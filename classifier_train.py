import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

# Percorso ai dati
data_dir = "training_10/"

# Parametri
img_height = 10
img_width = 10
batch_size = 32
epochs = 100  # Numero massimo di epoche

# Caricamento del dataset
train_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 80% training, 20% validation
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Ottimizzazione del caricamento
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
val_dataset = val_dataset.cache().prefetch(buffer_size=autotune)

# Creazione del modello
num_classes = 16

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
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
# Utilizzo di EarlyStopping e ModelCheckpoint per salvare il modello migliore
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_loss', save_best_only=True
)

# Addestramento del modello
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,  # Utilizzo della variabile `epochs`
    callbacks=[early_stopping, model_checkpoint]
)

# Il modello migliore è già salvato come 'best_model.keras'

