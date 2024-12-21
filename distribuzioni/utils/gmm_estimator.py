from argparse import ArgumentParser
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from math import sqrt
class DensityEstimator():
    def __init__(self):
        pass
    
    def _prepare_data(self, pkt_idx,feature_idx, features):
        X = []
        for sample in features:
            prev = sample[0, :pkt_idx, feature_idx].flatten()
            cur = sample[0, pkt_idx, feature_idx]
            X.append(np.hstack([prev, cur]))
        return np.array(X)
    
    def train(self, features:np.ndarray, args, to_optimized:bool=True):    
        datasize, _, pkt_num, feature_num = features.shape
        self.cgmms = np.zeros((pkt_num, feature_num), dtype=object)
        self.n_components = np.zeros((pkt_num, feature_num), dtype=object)
        self.features = features
        if to_optimized:
            gmm = GaussianMixture(random_state=args.seed,init_params=args.init_method,covariance_type=args.cov_type)
            param_grid = {'n_components': np.arange(1, 11)}
            grid = GridSearchCV(gmm, param_grid, cv=10)# 10CV with log-likelihood of model vs prepared input 
        else:
            gmm = GaussianMixture(n_components=sqrt(datasize/2), random_state=args.seed,init_params=args.init_method, 
                                          covariance_type=args.cov_type)
        for f in range(feature_num):
            for p in range(pkt_num): 
                X = self._prepare_data(p,f,features)
                if to_optimized:
                    grid.fit(X)
                    self.cgmms[p,f] = grid.best_estimator_
                    self.n_components[p,f] = grid.best_params_
                else:
                    self.cgmms[p,f] = gmm.fit(X)      
        return self.cgmms, self.n_components
    def pdfs(self):
        datasize, _, pkt_num, features_num = self.features.shape
        estimated_pdfs = np.zeros((pkt_num, features_num, datasize))
        for p in range(pkt_num):
            for f in range(features_num):
                gmm = self.cgmms[p, f]
                estimated_pdfs[:, p, f] = np.exp(gmm.score_samples(self._prepare_data(p,f,self.features)))
        return estimated_pdfs
    
    def generate(self, syn_sample_num:int):
        pkt_num, feature_num = self.cgmms.shape
        synthetic_samples = np.zeros((syn_sample_num, 1, pkt_num, feature_num))        
        for f in range(feature_num):
            synthetic_samples[:, 0, :, f] = self.cgmms[pkt_num-1, f].sample(syn_sample_num)[0]
        return synthetic_samples
    
    @staticmethod
    def generate_by_estimators(self, density_estimators, syn_sample_num:int):
        pkt_num, feature_num = density_estimators.shape
        synthetic_samples = np.zeros((syn_sample_num, 1, pkt_num, feature_num))
        
        for f in range(feature_num):
            synthetic_samples[:, 0, :, f] = density_estimators[pkt_num-1, f].sample(syn_sample_num)[0]
        return synthetic_samples
    
    @staticmethod
    def get_pdf(self, gmms, features):
        datasize, _, pkt_num, features_num = features.shape
        estimated_pdfs = np.zeros((datasize, pkt_num, features_num))
        for p in range(pkt_num):
            for f in range(features_num):
                gmm = gmms[p, f]
                estimated_pdfs[:, p, f] = np.exp(gmm.score_samples(self._prepare_data(p,f,features)))
        return estimated_pdfs
    
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the Generator specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--init-method', default='k-means++', type=str,required=False, dest='init_method',choices=['kmeans', 'k-means++', 'random', 'random_from_data'],
            help='The method used to initialize the weights, the means and the precisions. Responsibilities are initialized randomly if random.\n'+
            'selects initial cluster centroids using sampling based on an empirical probability distribution of the points contribution to the overall inertia if k-means++.'+ 
            'Algorithm: greedy k-means++ (several trials at each sampling step and choosing the best centroid among them)\n (default=%(default)s)')
        parser.add_argument('--covariance-type', default='full',  type=str,required=False, dest='cov_type',choices=[ 'full', 'tied', 'diag', 'spherical'],
                             help='full: each component has its own general covariance matrix. tied: all components share the same general covariance matrix.\ndiag: each component has its own diagonal covariance matrix.+'
                                    'spherical: each component has its own single variance. (default=%(default)s)')
        parser.add_argument('--seed', default=0, type=int,required=False, 
            help='Seed to use for distance computation. (default=%(default)s)')
        return parser.parse_known_args(args)
 