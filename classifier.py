import os
import csv
import numpy as np
import tensorflow as tf

# Disabilita l'uso della GPU per risolvere i problemi CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Parametri del modello e delle immagini
model_path = "best_model.keras"  # Path del modello salvato
image_dir = "testing/"          # Directory con le nuove immagini
output_csv = "predictions.csv"  # Nome del file CSV di output
img_height = 10
img_width = 10

# Caricamento del modello
model = tf.keras.models.load_model(model_path)

# Recupero dei nomi delle classi dal modello addestrato
class_names = sorted(os.listdir("model_generated_image/"))  # Ordina le sottodirectory per ricavare i nomi delle classi

# Funzione per elaborare e classificare le immagini
def classify_images(image_dir, model, class_names):
    predictions = []

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith('.npz'):
            continue

        # Caricamento e preprocessing del file .npz
        data = np.load(file_path)
        if 'gasf_img.npy' not in data:
            continue

        img_array = data['gasf_img.npy']  # Caricamento dell'immagine dalla matrice
        if img_array.shape[:2] != (img_height, img_width):  # Controllo dimensioni immagine
            continue

        img_array = np.expand_dims(img_array, axis=0)  # Creazione batch (aggiunge dimensione batch)

        # Predizione
        preds = model.predict(img_array)
        scores = tf.nn.softmax(preds[0])

        # Ottieni le prime 3 classi con confidenza
        top_indices = tf.argsort(scores, direction='DESCENDING')[:3]
        top_classes = [(class_names[i], float(scores[i]) * 100) for i in top_indices]

        # Salva il risultato
        predictions.append({
            "image": file_name,
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
