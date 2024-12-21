import os

def delete_matching_images(source_dir, reference_dir):
    """
    Elimina immagini in `reference_dir` se i loro nomi corrispondono a quelli trovati ricorsivamente in `source_dir`.

    :param source_dir: Directory con immagini organizzate in sottocartelle.
    :param reference_dir: Directory contenente tutte le immagini in un'unica cartella.
    """
    if not os.path.isdir(source_dir):
        print(f"Directory sorgente non trovata: {source_dir}")
        return
    if not os.path.isdir(reference_dir):
        print(f"Directory di riferimento non trovata: {reference_dir}")
        return

    # Ottieni un set di nomi immagine dalla struttura ricorsiva di `source_dir`
    source_images = set()
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                source_images.add(file)

    # Itera su tutti i file nella directory di riferimento
    for image in os.listdir(reference_dir):
        if image in source_images:
            image_path = os.path.join(reference_dir, image)
            try:
                os.remove(image_path)
                print(f"Eliminata immagine: {image_path}")
            except Exception as e:
                print(f"Errore nell'eliminazione di {image_path}: {e}")

# Configurazione
source_directory = "./training_10"  # Directory con immagini divise in sottocartelle
reference_directory = "./testing"  # Directory con tutte le immagini in un'unica cartella

# Esegui la funzione
delete_matching_images(source_directory, reference_directory)
