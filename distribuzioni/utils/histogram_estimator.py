from argparse import ArgumentParser
import numpy as np

class DensityEstimator():
    def __init__(self, real_features:np.ndarray, syn_features:np.ndarray, n_bins: int, n_pkt:int):
        self.n_pkt = n_pkt
        if real_features.ndim != 2:
            self.n_features = real_features.shape[3]
            self.real_features = np.reshape(self.real_features, (real_features.shape[0] * real_features.shape[1] * real_features.shape[2], self.n_features))
            self.syn_features = np.reshape(self.syn_features, (syn_features.shape[0] * syn_features.shape[1] * syn_features.shape[2], self.n_features))            
        else:
            self.n_features = real_features.shape[1]//n_pkt
            self.real_features = real_features.reshape((len(real_features)*n_pkt, self.n_features))
        self.real_pdf = np.zeros(self.n_features)
        self.syn_pdf = np.zeros(self.n_features)
        for i in range (self.n_features):
            self.real_pdf[i] = DensityEstimator._get_pdf(self.real_features[:, i],n_bins)
            self.syn_pdf[i] = DensityEstimator._get_pdf(self.syn_features[:, i],n_bins)

    def get_pdfs(self):
        return self.real_pdf, self.syn_pdf
    
    @staticmethod
    def _get_pdf(features:np.ndarray, n_bins: int) -> np.ndarray:
        print(features.shape)
        counts, bin_edges = np.histogram(features, bins=n_bins)
        pdf = counts / features.shape[0]
        print("pdf:",pdf)
        return pdf
    @staticmethod
    def get_pdfs(real_features:np.ndarray, syn_features:np.ndarray, n_bins: int, n_pkt:int):
        if real_features.ndim != 2:
            n_features = real_features.shape[3]
            real_features = np.reshape(real_features, (real_features.shape[0] * real_features.shape[1] * real_features.shape[2], n_features))
            syn_features = np.reshape(syn_features, (syn_features.shape[0] * syn_features.shape[1] * syn_features.shape[2], n_features))
        else:
            n_features = real_features.shape[1]/n_pkt
            real_features = real_features.reshape((len(real_features)*n_pkt, n_features))
        real_pdf = np.zeros((n_bins, n_features))
        syn_pdf = np.zeros((n_bins, n_features))
        for i in range (n_features):
            real_pdf[:, i] = DensityEstimator._get_pdf(real_features[:, i],n_bins)
            syn_pdf[:, i] = DensityEstimator._get_pdf(syn_features[:, i],n_bins)
        return real_pdf, syn_pdf
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the Generator specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--n-bin', default=100, type=int, required=False,  help='Bin number for feature binarization (default=%(default)s)')
        parser.add_argument('--n-pkt', default=10, type=int, required=False,  help='Packte number for each biflow (default=%(default)s)')
        return parser.parse_known_args(args)
 