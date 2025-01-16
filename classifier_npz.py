import os
import numpy as np
import tensorflow as tf
import csv

# Valori globali per la normalizzazione
min_global = -1460
max_global = 1460

# Funzione per normalizzare i dati
def normalize_data(data):
    """
    Normalizza i dati dall'intervallo originale [0, 255] a [-1, 1] usando i limiti globali.
    """
    denormalized_data = data * (max_global - min_global) + min_global
    return denormalized_data

# Caricamento del modello
model_path = "best_npz_classifier_model.keras"
model = tf.keras.models.load_model(model_path)

# Recupero dei nomi delle classi
class_names = sorted(os.listdir("model_generated_image/"))

# Funzione per classificare le immagini
def classify_images(image_dir, model, class_names):
    predictions = []
    i = 0
    for file_name in os.listdir(image_dir):
        print(i)
        file_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith('.npz'):
            continue

        # Caricamento e preprocessing
        data = np.load(file_path)
        key = list(data.keys())[0]  # Supponendo che la matrice sia la prima chiave
        matrix = data[key]
        diagonal = np.diag(matrix.squeeze())  # Estrarre la diagonale principale

        # Normalizzazione
        diagonal = normalize_data(diagonal)

        # Predizione
        diagonal = np.expand_dims(diagonal, axis=(0, -1))  # Aggiungi batch e canale
        preds = model.predict(diagonal)
        scores = tf.nn.softmax(preds[0])

        # Top 3 classi
        top_indices = tf.argsort(scores, direction='DESCENDING')[:3]
        top_classes = [(class_names[i], float(scores[i]) * 100) for i in top_indices]

        predictions.append({
            "image": file_name,
            "top_classes": top_classes
        })
        i++

    return predictions

# Funzione per salvare i risultati
def save_predictions_to_csv(predictions, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Class 1", "Confidence 1 (%)", "Class 2", "Confidence 2 (%)", "Class 3", "Confidence 3 (%)"])
        for pred in predictions:
            img_name = pred["image"]
            row = [img_name]
            for class_name, confidence in pred["top_classes"]:
                row.extend([class_name, f"{confidence:.2f}"])
            writer.writerow(row)

# Esecuzione
image_dir = "testing/"
output_csv = "predictions_npz.csv"
predictions = classify_images(image_dir, model, class_names)
save_predictions_to_csv(predictions, output_csv)
print(f"Predizioni salvate in {output_csv}")
