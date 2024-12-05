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
        target_subfolder = os.path.join(dest_dir, folder_name)
        os.makedirs(target_subfolder, exist_ok=True)

        # Elenca tutte le immagini nella directory sorgente
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # Seleziona un numero specifico di immagini casuali
        selected_images = random.sample(images, min(images_per_folder, len(images)))

        # Copia le immagini nella sottocartella di destinazione
        for image in selected_images:
            src_path = os.path.join(source_dir, image)
            dest_path = os.path.join(target_subfolder, image)
            shutil.copy(src_path, dest_path)

        print(f"Copiate {len(selected_images)} immagini da '{source_dir}' a '{target_subfolder}'.")

# Configurazione
source_directories = [
    "/Output_images_10_minmax/DiscoveryVR",
    "/Output_images_10_minmax/Facebook",
    "/Output_images_10_minmax/FulldiveVR",
    "/Output_images_10_minmax/Instagram",
    "/Output_images_10_minmax/Messenger",
    "/Output_images_10_minmax/Netflix",
    "/Output_images_10_minmax/PrimeVideo",
    "/Output_images_10_minmax/Skype",
    "/Output_images_10_minmax/Snapchat",
    "/Output_images_10_minmax/TikTok",
    "/Output_images_10_minmax/Vimeo",
    "/Output_images_10_minmax/VRRollercoaster",
    "/Output_images_10_minmax/Whatsapp",
    "/Output_images_10_minmax/Within",
    "/Output_images_10_minmax/Youtube",
    "/Output_images_10_minmax/Zoom"
]  # Sostituisci con i percorsi delle tue directory sorgenti
destination_directory = "/training_10"  # Sostituisci con il percorso della directory di destinazione
images_per_directory = 100

# Esegui la funzione
select_and_copy_images(source_directories, destination_directory, images_per_directory)
