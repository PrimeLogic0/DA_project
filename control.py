import os
import csv
from collections import defaultdict

# Percorsi delle directory e del file CSV
base_dir = "Output_images_npz/"  # Directory con le immagini organizzate in sottodirectory per classe
csv_path = "predictions_npz.csv"  # CSV generato con le predizioni
output_file = "evaluation_results.csv"  # File di output per i risultati

# Caricamento del CSV
def load_predictions_from_csv(csv_path):
    predictions = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row["Image"]
            predictions[image_name] = [
                (row["Class 1"], float(row["Confidence 1 (%)"])),
                (row["Class 2"], float(row["Confidence 2 (%)"])),
                (row["Class 3"], float(row["Confidence 3 (%)"]))
            ]
    return predictions

# Verifica delle immagini nella directory e calcolo delle statistiche
def evaluate_predictions(base_dir, predictions):
    stats = {
        "first_place": 0,
        "second_place": 0,
        "third_place": 0,
        "wrong": 0,
        "total_images": 0,
    }
    class_wise_counts = defaultdict(int)
    total_images_per_class = defaultdict(int)

    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Conta il totale delle immagini per classe
        total_images_per_class[class_name] = len([img for img in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, img))])

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            if img_name in predictions:
                stats["total_images"] += 1
                # Verifica se la classe corretta è nelle top 3 predette
                top_classes = [cls for cls, _ in predictions[img_name]]
                if class_name == top_classes[0]:
                    stats["first_place"] += 1
                    class_wise_counts[class_name] += 1
                elif class_name == top_classes[1]:
                    stats["second_place"] += 1
                elif class_name == top_classes[2]:
                    stats["third_place"] += 1
                else:
                    stats["wrong"] += 1

    return stats, class_wise_counts, total_images_per_class

# Funzione per salvare i risultati in un file CSV
def save_results_to_csv(stats, class_wise_counts, total_images_per_class, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Classe", "Immagini corrette come prima classe", "Totale immagini per classe"])

        # Scriviamo i risultati per ogni classe
        for class_name in class_wise_counts:
            writer.writerow([class_name, class_wise_counts[class_name], total_images_per_class[class_name]])

        # Scriviamo le statistiche globali
        writer.writerow([])
        writer.writerow(["Statistiche globali"])
        writer.writerow(["Totale immagini analizzate", stats["total_images"]])
        writer.writerow(["Immagini classificate correttamente come prima classe", stats["first_place"]])
        writer.writerow(["Immagini classificate correttamente come seconda classe", stats["second_place"]])
        writer.writerow(["Immagini classificate correttamente come terza classe", stats["third_place"]])
        writer.writerow(["Immagini sbagliate", stats["wrong"]])

# Main
predictions = load_predictions_from_csv(csv_path)
stats, class_wise_counts, total_images_per_class = evaluate_predictions(base_dir, predictions)

# Output su console
print("Risultati delle predizioni:")
print(f"Immagini classificate correttamente come prima classe: {stats['first_place']}")
print(f"Immagini classificate correttamente come seconda classe: {stats['second_place']}")
print(f"Immagini classificate correttamente come terza classe: {stats['third_place']}")
print(f"Immagini sbagliate: {stats['wrong']}")
print(f"Totale immagini analizzate: {stats['total_images']}")

print("\nConteggio per classe:")
for class_name, count in class_wise_counts.items():
    print(f"Classe {class_name}: {count} immagini corrette come prima classe")

# Salva i risultati nel file CSV
save_results_to_csv(stats, class_wise_counts, total_images_per_class, output_file)
print(f"\nI risultati sono stati salvati in {output_file}")
