import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Percorsi delle directory e del file CSV
base_dir = "Output_images_npz/"  # Directory con le immagini organizzate in sottodirectory per classe
csv_path = "predictions_npz.csv"  # CSV generato con le predizioni
output_image = "confusion_matrix.png"  # File di output per l'immagine della matrice di confusione

# Caricamento del CSV
def load_predictions_from_csv(csv_path):
    predictions = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row["Image"]
            predictions[image_name] = row["Class 1"]  # Classe predetta come prima classe
    return predictions

# Costruzione della matrice di confusione
def compute_confusion_matrix(base_dir, predictions):
    y_true = []
    y_pred = []

    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            if img_name in predictions:
                y_true.append(class_name)
                y_pred.append(predictions[img_name])

    return y_true, y_pred

# Funzione per creare e salvare la matrice di confusione
def save_confusion_matrix_image(y_true, y_pred, output_image):
    labels = sorted(set(y_true + y_pred))  # Etichette ordinate
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(10, 10))  # Aumenta la dimensione della figura
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)

    # Salvataggio dell'immagine
    plt.title("Confusion Matrix")
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()

# Main
predictions = load_predictions_from_csv(csv_path)
y_true, y_pred = compute_confusion_matrix(base_dir, predictions)

# Salva la matrice di confusione come immagine
save_confusion_matrix_image(y_true, y_pred, output_image)
print(f"La matrice di confusione è stata salvata in {output_image}")
