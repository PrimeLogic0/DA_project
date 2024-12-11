import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

def create_csv_from_images(folder_path, output_csv):
    """
    Crea un file CSV contenente il nome delle immagini e il nome della sottocartella che le contiene.

    :param folder_path: Percorso della cartella principale contenente le sottocartelle di immagini.
    :param output_csv: Percorso del file CSV da generare.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label'])  # Intestazione del CSV

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):  # Verifica che sia una sottocartella
                for image in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image)
                    if os.path.isfile(image_path):  # Verifica che sia un file
                        writer.writerow([image, subfolder])


def join_csv_files(csv_file1, csv_file2, output_csv, join_field="Image", join_type="inner"):
    """
    Esegue una join tra due file CSV basandosi su un campo comune.

    :param csv_file1: Percorso del primo file CSV.
    :param csv_file2: Percorso del secondo file CSV.
    :param output_csv: Percorso del file CSV risultante.
    :param join_field: Nome del campo su cui effettuare la join (default: "Image").
    :param join_type: Tipo di join: "inner", "outer", "left", "right" (default: "inner").
    """
    # Carica i file CSV in DataFrame
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Esegui la join
    merged_df = pd.merge(df1, df2, on=join_field, how=join_type)

    # Salva il risultato in un nuovo file CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"File CSV risultante creato con successo: {output_csv}")


def add_verification_column(csv_file, output_csv, label_field="Label", class_field="Class 1"):
    """
    Aggiunge una colonna "Results" a un file CSV, che verifica se Label == Class 1.

    :param csv_file: Percorso del file CSV da elaborare.
    :param output_csv: Percorso del file CSV risultante.
    :param label_field: Nome del campo che contiene le etichette (default: "Label").
    :param class_field: Nome del campo che contiene la classe da confrontare (default: "Class 1").
    """
    # Leggi il file CSV
    df = pd.read_csv(csv_file)

    # Aggiungi la colonna "Results" con T o F
    df['Results'] = df[label_field] == df[class_field]
    df['Results'] = df['Results'].map({True: 'T', False: 'F'})

    # Salva il DataFrame modificato in un nuovo file CSV
    df.to_csv(output_csv, index=False)
    print(f"File CSV con colonna 'Results' creato con successo: {output_csv}")


def plot_results_piechart(csv_file, results_field="Results"):
    """
    Crea un pie chart con la percentuale di T e F nella colonna "Results" di un file CSV.

    :param csv_file: Percorso del file CSV contenente i risultati.
    :param results_field: Nome della colonna che contiene i risultati (default: "Results").
    """
    # Leggi il file CSV
    df = pd.read_csv(csv_file)

    # Conta le occorrenze di T e F
    results_count = df[results_field].value_counts()

    # Etichette e valori per il pie chart
    labels = results_count.index
    sizes = results_count.values
    colors = ['#4CAF50', '#FF5733']  # Colori per T e F
    explode = [0.1 if label == 'T' else 0 for label in labels]  # Esplosione per T

    # Crea il pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
    plt.title('Distribuzione di T e F nei risultati')
    plt.axis('equal')  # Assicura che il grafico sia un cerchio
    plt.show()

csv_file1 = "oracolo.csv"
csv_file2 = "./Classifier_Results/Blue_line/predictions_real.csv"
csv_file3 = "joined_output2.csv"
output_csv = "verified_output2.csv"
output_csv2 = "joined_output2.csv"

plot_results_piechart(output_csv)

#add_verification_column(csv_file3, output_csv, label_field="Label", class_field="Class 1")
#join_csv_files(csv_file1, csv_file2, output_csv2, join_field="Image", join_type="inner")


