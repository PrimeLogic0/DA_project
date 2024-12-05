import math
import random
import os
import csv
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=True,
    random_crop=False,
    random_flip=False,
):
    """
    Carica immagini preprocessate e, se specificato, le classi corrispondenti.

    :param data_dir: Directory contenente immagini organizzate per classi.
    :param batch_size: Dimensione del batch per il DataLoader.
    :param image_size: Dimensione a cui ridimensionare le immagini.
    :param class_cond: Se True, include le etichette delle classi.
    :param random_crop: Se True, applica crop casuali per augmentation.
    :param random_flip: Se True, applica flip orizzontali casuali per augmentation.
    :return: Generatore infinito che produce batch di immagini e, opzionalmente, etichette.
    """
    all_files, labels = _list_image_files_recursively(data_dir)

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=labels if class_cond else None,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Elenca tutti i file immagine all'interno di una directory, suddivisi per classi.
    Ogni sottodirectory rappresenta una classe, convertita in un indice numerico.
    """
    all_files = []
    labels = []

    # Mappa per assegnare un indice univoco alle directory
    label_map = {}

    for label_idx, label in enumerate(sorted(os.listdir(data_dir))):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Considera solo directory
            # Mappa la directory a un indice univoco
            label_map[label] = label_idx
            for file in os.listdir(label_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Considera solo immagini
                    all_files.append(os.path.join(label_dir, file))
                    labels.append(label_idx)  # Usa l'indice numerico associato alla directory

    csv_filename = "./128/iterate/df/synth_models/label_map.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Scrivi l'intestazione (opzionale)
        writer.writerow(["label", "index"])

        # Scrivi i dati della mappatura
        for label, index in label_map.items():
            writer.writerow([label, index])

    return all_files, labels



class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
