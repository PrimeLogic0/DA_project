import os
import csv
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Parametri del modello e delle immagini
model_path = "best_model.keras"  # Path del modello salvato
image_dir = "testing/"        # Directory con le nuove immagini
output_csv = "predictions.csv"  # Nome del file CSV di output
img_height = 10
img_width = 10

# Caricamento del modello
model = tf.keras.models.load_model(model_path)

# Recupero dei nomi delle classi dal modello addestrato
# Supponendo che train_dataset.class_names fosse salvato in fase di training
class_names = sorted(os.listdir("model_generated_image/"))  # Ordina le sottodirectory per ricavare i nomi delle classi

# Funzione per elaborare e classificare le immagini
def classify_images(image_dir, model, class_names):
    predictions = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        # Caricamento e preprocessing dell'immagine
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Creazione batch

        # Predizione
        preds = model.predict(img_array)
        scores = tf.nn.softmax(preds[0])

        # Ottieni le prime 3 classi con confidenza
        top_indices = tf.argsort(scores, direction='DESCENDING')[:3]
        top_classes = [(class_names[i], float(scores[i]) * 100) for i in top_indices]

        # Salva il risultato
        predictions.append({
            "image": img_name,
            "top_classes": top_classes
        })

    return predictions

# Funzione per salvare i risultati in un file CSV
def save_predictions_to_csv(predictions, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Class 1", "Confidence 1 (%)", "Class 2", "Confidence 2 (%)", "Class 3", "Confidence 3 (%)"])

        for pred in predictions:
            img_name = pred["image"]
            top_classes = pred["top_classes"]
            row = [img_name]
            for class_name, confidence in top_classes:
                row.append(class_name)
                row.append(f"{confidence:.2f}")

            writer.writerow(row)

# Esecuzione del processo
predictions = classify_images(image_dir, model, class_names)
save_predictions_to_csv(predictions, output_csv)

print(f"Predizioni salvate in {output_csv}")

