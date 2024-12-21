from argparse import ArgumentParser
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class DensityEstimator():
    def __init__(self):
        pass

    def train(self, features:np.ndarray, args, to_optimized=True):      
        *_, pkt_num, feature_num = features.shape
        self.kdes = np.zeros((pkt_num, feature_num), dtype=object)
        if to_optimized:
            params = {"bandwidth": np.logspace(-2, 0, 40)}
            grid = GridSearchCV(KernelDensity(kernel=args.kernel,metric=args.metric), params, cv=10)
        for p in range(pkt_num):
            for f in range(feature_num):
                    print(f"Kernel Density Processing packet {p}, feature {f}")
                    X = features[:, 0, p, f].reshape(-1, 1)
                    if to_optimized:
                        grid.fit(X)
                        kde = grid.best_estimator_
                    else:
                        kde = KernelDensity(bandwidth=args.bandwidth,kernel=args.kernel,metric=args.metric).fit(X)
                    self.kdes[ p, f] = kde
        return self.kdes
    
    def generate(self, syn_sample_num:int):
        pkt_num, feature_num = self.kdes.shape
        synthetic_samples = np.zeros((syn_sample_num, 1, pkt_num, feature_num))
        
        for f in range(feature_num):
            for p in range(pkt_num):
                synthetic_samples[:, 0, p, f] = self.kdes[p, f].sample(syn_sample_num)[0]
        return synthetic_samples

    
    @staticmethod
    def generate_by_estimators(density_estimators, syn_sample_num:int):
        pkt_num, feature_num = density_estimators.shape
        synthetic_samples = np.zeros((syn_sample_num, 1, pkt_num, feature_num))
        
        for f in range(feature_num):
            for p in range(pkt_num):
                synthetic_samples[:, 0, p, f] = density_estimators[p, f].sample(syn_sample_num)[0]
        return synthetic_samples
    
    @staticmethod
    def log_likelihood(des:np.ndarray, features:np.ndarray):
        #Compute the log-likelihood (normalized to be probability densities) of each sample
        datasize, _, pkt_num, features_num = features.shape
        scores = np.zeros((datasize, pkt_num, features_num))
        for p in range(pkt_num):
            for f in range(features_num):
                kde = des[p, f]
                scores[:, p, f] = kde.score_samples(features[:, 0, p, f].reshape(-1, 1))
        return scores
    
    @staticmethod
    def get_pdf(kdes:np.ndarray, features:np.ndarray):
        pkt_num, features_num = kdes.shape
        datasize = features.shape[0]
        estimated_pdfs = np.zeros((datasize,pkt_num, features_num))
        for p in range(pkt_num):
            for f in range(features_num):
                kde = kdes[p, f]
                estimated_pdfs[:, p, f] = np.exp(kde.score_samples(features[:, 0, p, f].reshape(-1, 1)))
        return estimated_pdfs
    
    

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the Generator specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--kernel', default='gaussian', type=str,required=False, dest='kernel',choices=['gaussian','epanechnikov','exponential','cosine'],
            help='The kernel to use (default=%(default)s)')
        parser.add_argument('--bandwidth', default=1.0, type=float,required=False, dest='bandwidth', help='The bandwidth of the kernel as float (default=%(default)s)')
        parser.add_argument('--metric', default="euclidean", type=str,required=False, dest='metric',
            help='Metric to use for distance computation. See the documentation of scipy.spatial.distance and the metrics listed in distance_metrics for valid metric values. (default=%(default)s)')
        parser.add_argument('--seed', default=0, type=int,required=False, 
            help='Seed to use for distance computation. (default=%(default)s)')
        return parser.parse_known_args(args)
 