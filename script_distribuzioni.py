import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def calcola_distribuzione_per_cartelle(directory_principale):
    """
    Calcola la distribuzione media dei pixel per ciascuna sotto-directory contenente immagini 10x10
    e salva i grafici con il nome della sotto-directory.

    Parametri:
        directory_principale (str): Percorso della directory principale contenente le sotto-directory.
    """
    # Verifica se la directory principale esiste
    if not os.path.exists(directory_principale):
        print(f"Directory {directory_principale} non trovata!")
        return

    # Leggi tutte le sotto-directory
    sotto_directory = [os.path.join(directory_principale, d) for d in os.listdir(directory_principale) if
                       os.path.isdir(os.path.join(directory_principale, d))]

    # Controlla se ci sono sotto-directory
    if not sotto_directory:
        print(f"Nessuna sotto-directory trovata nella directory {directory_principale}!")
        return

    # Processa ogni sotto-directory
    for sotto_dir in sotto_directory:
        # Nome della sotto-directory
        nome_cartella = os.path.basename(sotto_dir)

        # Lista per contenere i pixel di tutte le immagini nella sotto-directory
        tutti_i_pixel = []

        # Leggi tutte le immagini nella sotto-directory
        for file in os.listdir(sotto_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Filtra i formati di immagine
                percorso_file = os.path.join(sotto_dir, file)
                # Apri l'immagine
                immagine = Image.open(percorso_file)
                # Verifica se l'immagine è 10x10 pixel
                if immagine.size == (10, 10):
                    # Converti l'immagine in scala di grigi
                    immagine = immagine.convert('L')  # 'L' = scala di grigi
                    # Aggiungi i pixel dell'immagine alla lista
                    tutti_i_pixel.extend(np.array(immagine).flatten())

        # Calcola la distribuzione media
        if len(tutti_i_pixel) > 0:
            # Creare un istogramma della distribuzione dei pixel, normalizzato
            plt.figure(figsize=(10, 6))
            plt.hist(tutti_i_pixel, bins=256, range=(0, 255), color='blue', alpha=0.7, edgecolor='black', density=True)
            plt.title(f"Distribuzione media dei pixel - {nome_cartella}")
            plt.xlabel("Valore dei pixel (0-255)")
            plt.ylabel("Densità di frequenza")
            plt.grid(axis='y', alpha=0.75)

            # Salva il grafico come immagine
            nome_file_grafico = os.path.join(directory_principale, f"{nome_cartella}.png")
            plt.savefig(nome_file_grafico)
            plt.close()
            print(f"Grafico salvato: {nome_file_grafico}")
        else:
            print(f"Nessuna immagine 10x10 trovata nella cartella {nome_cartella}!")


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def confronta_distribuzioni(directory_principale1, directory_principale2, directory_output):
    """
    Confronta i grafici delle distribuzioni generate da due directory principali (con sottocartelle omonime)
    e salva i grafici sovrapposti in una directory di output.

    Parametri:
        directory_principale1 (str): Percorso della prima directory principale.
        directory_principale2 (str): Percorso della seconda directory principale.
        directory_output (str): Percorso della directory in cui salvare i grafici sovrapposti.
    """
    # Verifica se le directory principali esistono
    if not os.path.exists(directory_principale1):
        print(f"Directory {directory_principale1} non trovata!")
        return
    if not os.path.exists(directory_principale2):
        print(f"Directory {directory_principale2} non trovata!")
        return
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
        print(f"Directory di output {directory_output} creata!")

    # Trova le sottocartelle comuni
    sottocartelle1 = {d for d in os.listdir(directory_principale1) if os.path.isdir(os.path.join(directory_principale1, d))}
    sottocartelle2 = {d for d in os.listdir(directory_principale2) if os.path.isdir(os.path.join(directory_principale2, d))}
    sottocartelle_comuni = sottocartelle1 & sottocartelle2

    if not sottocartelle_comuni:
        print("Non ci sono sottocartelle comuni tra le due directory principali!")
        return

    for sottocartella in sottocartelle_comuni:
        percorso_dir1 = os.path.join(directory_principale1, sottocartella)
        percorso_dir2 = os.path.join(directory_principale2, sottocartella)

        # Lista per contenere tutti i pixel delle immagini di entrambe le directory
        pixel_dir1 = []
        pixel_dir2 = []

        # Carica i pixel da tutte le immagini della prima directory
        for file in os.listdir(percorso_dir1):
            if file.endswith(".png"):
                img = Image.open(os.path.join(percorso_dir1, file)).convert('L')
                pixel_dir1.extend(np.array(img).flatten())

        # Carica i pixel da tutte le immagini della seconda directory
        for file in os.listdir(percorso_dir2):
            if file.endswith(".png"):
                img = Image.open(os.path.join(percorso_dir2, file)).convert('L')
                pixel_dir2.extend(np.array(img).flatten())

        # Verifica se sono stati trovati pixel
        if pixel_dir1 and pixel_dir2:
            # Crea gli istogrammi sovrapposti (normalizzati)
            plt.figure(figsize=(10, 6))
            plt.hist(pixel_dir1, bins=256, range=(0, 255), color='blue', alpha=0.5, label=f"{sottocartella} - Dir1", density=True)
            plt.hist(pixel_dir2, bins=256, range=(0, 255), color='red', alpha=0.5, label=f"{sottocartella} - Dir2", density=True)
            plt.title(f"Confronto distribuzioni - {sottocartella}")
            plt.xlabel("Valore dei pixel (0-255)")
            plt.ylabel("Densità di frequenza")
            plt.legend()
            plt.grid(axis='y', alpha=0.75)

            # Salva il grafico
            output_path = os.path.join(directory_output, f"confronto_{sottocartella}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Grafico salvato: {output_path}")
        else:
            print(f"Nessun dato trovato per la sottocartella {sottocartella}.")

#calcola_distribuzione_per_cartelle("/Users/matteospavone/Desktop/model_generated_image")
confronta_distribuzioni("/Users/matteospavone/Desktop/model_generated_image", "/Users/matteospavone/Desktop/Output_images_10_minmax", "/Users/matteospavone/Desktop/prova")
