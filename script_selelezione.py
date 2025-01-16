import os
import random
import shutil

def select_and_copy_images(source_dirs, dest_dir, images_per_folder=100):
    """
    Seleziona un numero specifico di immagini da ciascuna directory sorgente
    e le copia in una nuova directory organizzata in sottocartelle.

    :param source_dirs: Lista di percorsi delle directory sorgenti.
    :param dest_dir: Percorso della directory di destinazione.
    :param images_per_folder: Numero di immagini da selezionare per ciascuna cartella.
    """
    # Crea la directory di destinazione se non esiste
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Directory non trovata: {source_dir}")
            continue

        # Ottieni il nome della cartella sorgente
        folder_name = os.path.basename(source_dir.rstrip("/\\"))

        # Crea una sottocartella nella directory di destinazione
        sub_dest_dir = os.path.join(dest_dir)
        os.makedirs(sub_dest_dir, exist_ok=True)

        # Elenca tutte le immagini nella directory sorgente
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('npz'))]

        # Seleziona un numero specifico di immagini casuali
        selected_images = random.sample(images, min(images_per_folder, len(images)))

        # Copia le immagini nella sottocartella di destinazione
        for image in selected_images:
            src_path = os.path.join(source_dir, image)
            dest_path = os.path.join(sub_dest_dir, image)
            shutil.copy(src_path, dest_path)

        print(f"Copiate {len(selected_images)} immagini da '{source_dir}' a '{sub_dest_dir}'.")

# Configurazione
source_directories = [
    "./Output_images_npz/DiscoveryVR",
    "./Output_images_npz/Facebook",
    "./Output_images_npz/FulldiveVR",
    "./Output_images_npz/Instagram",
    "./Output_images_npz/Messenger",
    "./Output_images_npz/Netflix",
    "./Output_images_npz/PrimeVideo",
    "./Output_images_npz/Skype",
    "./Output_images_npz/Snapchat",
    "./Output_images_npz/TikTok",
    "./Output_images_npz/Vimeo",
    "./Output_images_npz/VRRollercoaster",
    "./Output_images_npz/Whatsapp",
    "./Output_images_npz/Within",
    "./Output_images_npz/Youtube",
    "./Output_images_npz/Zoom"
]  # Sostituisci con i percorsi delle tue directory sorgenti
destination_directory = "./testing"  # Sostituisci con il percorso della directory di destinazione
images_per_directory = 100000

# Esegui la funzione
select_and_copy_images(source_directories, destination_directory, images_per_directory)
