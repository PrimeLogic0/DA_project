import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from functools import partial
from pathlib import Path

sys.path.append('/media/data/facil-tc/src/')
from evaluator.micro_quality_evaluator import BiflowEvaluator


def compute_distances(real_histograms, synthetic_histograms, n_features):
    """
    Compute distances between real and synthetic histograms.
    
    Parameters:
        real_histograms (np.ndarray): Real data histograms ((n_features,) + (n_bins,) * ngram_size ).
        synthetic_histograms (np.ndarray): Synthetic data histograms ((n_features,) + (n_bins,) * ngram_size ).
        n_features (int): Number of features.
    
    Returns:
        dict: Distances (JSD, Hellinger, TVD) for each feature.
    """
    jsd = np.zeros(n_features)
    hd = np.zeros(n_features)
    tvd = np.zeros(n_features)
    
    for f in range(n_features):
        real_pdf = real_histograms[f, :]
        synth_pdf = synthetic_histograms[f, :]
        
        jsd[f] = BiflowEvaluator.compute_jsd(real_pdf, synth_pdf)
        hd[f] = BiflowEvaluator.compute_hd(real_pdf, synth_pdf)
        tvd[f] = BiflowEvaluator.compute_tvd(real_pdf, synth_pdf)

    return {
        "JSD": jsd,
        "Hellinger": hd,
        "TVD": tvd
    }


def load_parquet_as_array(file_path, n_pkts):
    """
    Load a parquet file and convert it into a 3D numpy array.

    Parameters:
        file_path (str): Path to the parquet file.

    Returns:
        tuple:
            - np.ndarray: 3D array with shape (samples, time_steps, features).
            - np.ndarray: Labels associated with each sample.
    """
    df = pd.read_parquet(file_path)
    labels = df["LABEL"].values  # Updated to use the correct column name

    data = []

    for _, row in df.iterrows():
        pl = row['PL']  # Lista delle dimensioni dei pacchetti
        dir_ = list(row['DIR'])  # Convertire esplicitamente in lista

        # Creazione dei valori positivi/negativi per PL
        pl_adjusted = [p if d == 1 else -p for p, d in zip(pl, dir_)]

        if not pl_adjusted:  # Verifica se la lista è vuota
            data.append([np.zeros(n_pkts), np.zeros(n_pkts)])  # Valori di default
            continue

        # Riempimento o troncamento a n_pkts valori
        pl_adjusted = (pl_adjusted + [np.int64(0)] * n_pkts)[:n_pkts]
        dir_adjusted = (list(dir_) + [np.int64(0)] * n_pkts)[:n_pkts]

        data.append([pl_adjusted, dir_adjusted])

    # Convertire la lista in un array NumPy con la forma (samples, time_steps, features)
    data = np.transpose(np.array(data), (0, 2, 1))

    return data, labels


def extract_ngrams(data, n):
    """
    Extract n-grams from time-series data for each feature.
    
    Parameters:
        data (np.ndarray): Input data with shape (samples, time_steps, features).
        n (int): Size of the n-grams.
    
    Returns:
        np.ndarray: Flattened n-grams for histogram computation with shape (samples * (time_steps - n + 1), features * n).
    """
    samples, time_steps, features = data.shape
    if time_steps < n:
        raise ValueError(f"Time steps ({time_steps}) must be >= n-gram size ({n}).")

    # Extract n-grams
    ngrams = np.array([
        data[:, i:i + n, :]
        for i in range(time_steps - n + 1)
    ])

    return ngrams.reshape((np.prod(ngrams.shape[:2]),) + ngrams.shape[2:])


def compute_histograms(real_data, synthetic_data, n_bins, ngram_size, n_features):
    """
    Compute histograms for real and synthetic data using the HistogramEstimator.
    
    Parameters:
        real_data (np.ndarray): Real dataset (samples, time_steps, features).
        synthetic_data (np.ndarray): Synthetic dataset (samples, time_steps, features).
        n_bins (int): Number of bins for the histogram.
        ngram_size (int): Size of n-grams for each feature.
    
    Returns:
        tuple: Normalized histograms for real and synthetic datasets.
    """
    # Extract n-grams
    real_ngrams = extract_ngrams(real_data, ngram_size)
    synthetic_ngrams = extract_ngrams(synthetic_data, ngram_size)

    # Filter padding samples
    no_pad_index = np.where(real_ngrams[:, :, 0] != 0)[0]
    no_pad_index, counts = np.unique(no_pad_index, return_counts=True)
    no_pad_index = no_pad_index[np.where(counts == ngram_size)]
    real_ngrams = real_ngrams[no_pad_index]

    no_pad_index = np.where(synthetic_ngrams[:, :, 0] != 0)[0]
    no_pad_index, counts = np.unique(no_pad_index, return_counts=True)
    no_pad_index = no_pad_index[np.where(counts == ngram_size)]
    synthetic_ngrams = synthetic_ngrams[no_pad_index]

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', subsample=1000, strategy='uniform')
    flat_real_ngrams = real_ngrams.reshape(np.prod(real_ngrams.shape[:-1]), real_ngrams.shape[-1])
    discretizer.fit(flat_real_ngrams)
    discr_real_ngrams = np.array(list(map(discretizer.transform, real_ngrams)), dtype=int).transpose(0, 2, 1)
    discr_synthetic_ngrams = np.array(list(map(discretizer.transform, synthetic_ngrams)), dtype=int).transpose(0, 2, 1)

    real_histograms = np.zeros((n_features,) + (n_bins,) * ngram_size, dtype=float)
    for gram_feats in discr_real_ngrams:
        for i, gram in enumerate(gram_feats):
            index = tuple((i, *gram))
            real_histograms[index] += 1
    
    synthetic_histograms = np.zeros((n_features,) + (n_bins,) * ngram_size, dtype=float)
    for gram_feats in discr_synthetic_ngrams:
        for i, gram in enumerate(gram_feats):
            index = tuple((i, *gram))
            synthetic_histograms[index] += 1

    # The normalization should involve all the ngram axes (NB. there are axes for feature level)
    axes = tuple(range(0, ngram_size))

    # Normalize histograms
    real_histograms /= np.array(list(map(partial(np.sum, axis=axes, keepdims=True), real_histograms)), dtype=float)
    synthetic_histograms /= np.array(list(map(partial(np.sum, axis=axes, keepdims=True), synthetic_histograms)), dtype=float)
    
    return real_histograms, synthetic_histograms


def compute_per_class_distances(real_data, real_labels, synthetic_data, synthetic_labels, n_bins, ngram_size):
    """
    Compute per-class distances for the real and synthetic datasets.
    
    Parameters:
        real_data (np.ndarray): Real dataset (samples, time_steps, features).
        real_labels (np.ndarray): Labels for the real dataset.
        synthetic_data (np.ndarray): Synthetic dataset (samples, time_steps, features).
        synthetic_labels (np.ndarray): Labels for the synthetic dataset.
        n_bins (int): Number of bins for histogram computation.
        ngram_size (int): Size of n-grams for each feature.
    
    Returns:
        dict: Per-class distances (JSD, Hellinger, TVD) for each class.
    """
    n_features = real_data.shape[-1]
    classes = np.unique(real_labels)
    per_class_distances = {}

    for cls in classes:
        # Filter real and synthetic data for the current class
        real_cls_data = real_data[real_labels == cls]
        synthetic_cls_data = synthetic_data[synthetic_labels == cls]

        # Compute histograms
        real_histograms, synthetic_histograms = compute_histograms(
            real_cls_data, synthetic_cls_data, n_bins, ngram_size, n_features)

        # Compute distances
        distances = compute_distances(real_histograms, synthetic_histograms, n_features)

        per_class_distances[cls] = distances

    return per_class_distances


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate similarity between real and synthetic datasets using histogram-based distances.")
    parser.add_argument("--real_path", type=str, required=True, help="Path to the real dataset (parquet file).")
    parser.add_argument("--synthetic_path", type=str, required=True, help="Path to the synthetic dataset (parquet file).")
    parser.add_argument("--n_pkts", type=int, default=500, help="Number of packets to consider (default: 10).")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for histogram computation (default: 10).")
    parser.add_argument("--ngram_size", type=int, default=1, help="Size of n-grams for each feature (default: 1, no n-grams).")
    args = parser.parse_args()

    # Load datasets
    real_data, real_labels = load_parquet_as_array(args.real_path, args.n_pkts)
    synthetic_data, synthetic_labels = load_parquet_as_array(args.synthetic_path, args.n_pkts)

    # Validate dataset features
    if real_data.shape[2] != synthetic_data.shape[2]:
        raise ValueError("Real and synthetic datasets must have the same number of features (columns).")

    # Compute per-class distances
    per_class_distances = compute_per_class_distances(real_data, real_labels, synthetic_data, synthetic_labels,
                                                      args.n_bins, args.ngram_size)

    # Print results
    Path(f"./evaluation_results").mkdir(parents=True, exist_ok=True)
    with open(f"./evaluation_results/{args.real_path.split('/')[-1]}_{args.ngram_size}_{args.n_bins}.txt", "w") as file:
        for cls, distances in per_class_distances.items():
            print(f"Class {cls}:")
            print("  Jensen-Shannon Divergence (per feature):", distances["JSD"])
            print("  Hellinger Distance (per feature):", distances["Hellinger"])
            print("  Total Variation Distance (per feature):", distances["TVD"])
            file.write(f"Class {cls}:\n")
            file.write(f"   Jensen-Shannon Divergence (per feature): {distances['Hellinger']}\n")
            file.write(f"   Hellinger Distance (per feature): {distances['Hellinger']}\n")
            file.write(f"   Total Variation Distance (per feature): {distances['TVD']}\n")
