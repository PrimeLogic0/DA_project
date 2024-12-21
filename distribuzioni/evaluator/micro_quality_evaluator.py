from argparse import ArgumentParser
import importlib
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import *
from tqdm import tqdm
from scipy.stats.mstats import gmean
from scipy.spatial.distance import jensenshannon
from utils.kernel_density_estimator import DensityEstimator as kd
from utils.histogram_estimator import DensityEstimator as histogram
from utils import gmm_estimator as gmm
from utils.transformers import get_images, get_labels
from scipy.special import kl_div
from copy import deepcopy
from types import SimpleNamespace

class BiflowEvaluator():
    def __init__(self, augmented_dataloader, syn_indexes, args) :
        if isinstance(augmented_dataloader, str):  #is the dataset path
            df = pd.read_parquet(augmented_dataloader)
            syn_indexes = df[df['Type']== 'syn'].index.tolist()
            images = np.array([row.reshape(1, 10, 4) for row in df['Features'].values])
            labels = df['Targets'].values
            self.augmented_dataloader = SimpleNamespace()
            self.augmented_dataloader.dataset = SimpleNamespace()
            self.augmented_dataloader.images = images
            self.augmented_dataloader.labels = labels
            self.syn_indexes = syn_indexes
            class_order = df['Order'].values[0]
        else:
            self.augmented_dataloader = deepcopy(augmented_dataloader)
            images = get_images(augmented_dataloader)
            labels = get_labels (augmented_dataloader)
            self.syn_indexes = syn_indexes

        self.real_indexes = [i for i in range(len(labels)) if i not in syn_indexes]
        self.real_features = images[self.real_indexes]
        self.real_target = labels[self.real_indexes]
        self.syn_features = images[syn_indexes]
        self.syn_targets =  labels[syn_indexes]
        self.real_datasize, _, self.pkt_num, self.features_num = self.real_features.shape
        self.syn_datasize = len(self.syn_targets)
        unique_classes=sorted(set(self.real_target))
        class_num = len(unique_classes)
        self.estimators=np.zeros( (class_num, self.pkt_num, self.features_num),  dtype=object)
        self.real_pdfs = np.empty( class_num,  dtype=object)
        self.syn_pdfs = np.empty(class_num, dtype=object)
        
        self.jsd = np.zeros( (self.syn_datasize, self.pkt_num, self.features_num),  dtype=float)
        self.log_likeh = np.zeros( (self.syn_datasize, self.pkt_num, self.features_num),  dtype=float)
        self.hellinger_dist = np.zeros( (self.syn_datasize, self.pkt_num, self.features_num),  dtype=float)
        self.tot_var_dist = np.zeros( (self.syn_datasize, self.pkt_num, self.features_num),  dtype=float)
        
        class_jsd = np.empty( (class_num, ),  dtype=object)
        class_log_likeh = np.empty( (class_num, ),  dtype=object)
        class_hellinger_dist = np.empty( (class_num, ),  dtype=object)
        class_tot_var_dist = np.empty( (class_num, ),  dtype=object)

        unique_targets, counts = np.unique(self.real_target, return_counts=True)
        majority_class = counts.argmax()
        for idx, c in enumerate(unique_classes):
            c_indices = np.where(self.real_target == c)[0]
            c_syn_indices = np.where(self.syn_targets == c)[0]
            r_features = self.real_features[c_indices]
            s_features = self.syn_features[c_syn_indices]           
            #class estimation pdf
            match args.estimator_type:
                case "kd":
                    DensityEstimator = kd
                case "gmm":
                    DensityEstimator = gmm
            
            estimator = DensityEstimator()
            self.estimators[c] = estimator.train(features=r_features,args=args, to_optimized=True)
            r_pdfs = DensityEstimator.get_pdf(self.estimators[c], r_features)
            if c == majority_class:
                self.syn_pdfs[c] = np.NAN
                continue
            else:
                s_pdfs = DensityEstimator.get_pdf(self.estimators[c], s_features)
            self.real_pdfs[c] = r_pdfs
            self.syn_pdfs[c] = s_pdfs
            #evaluation for each syn sample
            for p in range(self.pkt_num):
                for f in range(self.features_num):
                    pdfs_avg = np.mean(r_pdfs[:,p,f],axis=0)
                    for i in range(s_pdfs.shape[0]):
                        pm = (pdfs_avg + s_pdfs[i,p,f]) / 2
                        real_kld = np.average([kl_div(p, pm) for p in r_pdfs[:,p,f]])
                        syn_kld = kl_div(s_pdfs[i,p,f], pm)
                        
                        self.jsd[c_syn_indices[i],p,f] =  np.sqrt(0.5 * (real_kld + syn_kld)) / np.log(2) #jensenshannon(pdfs_avg,s_pdfs[i,p,f])**2
                        self.hellinger_dist[c_syn_indices[i],p,f] = np.sqrt(np.sum((np.sqrt(pdfs_avg) - np.sqrt(s_pdfs[i,p,f])) ** 2)) / np.sqrt(2)
                        self.tot_var_dist[c_syn_indices[i],p,f] = 0.5 * np.sum(np.abs(pdfs_avg - s_pdfs[i,p,f]))
            
            self.log_likeh[c_syn_indices,:,:] = DensityEstimator.log_likelihood(self.estimators[c],s_features)
            class_jsd[class_order[c]] = self.jsd[c_syn_indices]
            class_log_likeh[class_order[c]] = self.log_likeh[c_syn_indices]
            class_hellinger_dist[class_order[c]] = self.hellinger_dist[c_syn_indices]
            class_tot_var_dist[class_order[c]] = self.tot_var_dist[c_syn_indices]

        self.metrix_indexes_to_aug_dataset_indexes = {i:self.syn_indexes[i] for i in range(self.syn_datasize)}
        self.simil_metric_by_class_df = pd.DataFrame({
            'log_likeh': class_log_likeh.tolist(),
            'hellinger_dist' : class_hellinger_dist.tolist(),
            "tvd" : class_tot_var_dist.tolist(),
            'jsd' : class_jsd.tolist()
         })

    def index_to_original_dataset_index (self, i:int):
        return self.metrix_indexes_to_aug_dataset_indexes[i]
    
    def get_micro_quality_valutation(self):
        return self.log_likeh ,self.hellinger_dist, self.tot_var_dist, self.jsd, self.metrix_indexes_to_aug_dataset_indexes, self.simil_metric_by_class_df

    def get_metric(self,metric_name:str):
        return getattr(self,metric_name), self.metrix_indexes_to_aug_dataset_indexes
    
    def get_metrics_names(self):
        return ["log_likeh" , "hellinger_dist", "tot_var_dist", "jsd"]
    
    def get_best_syn_samples(self,target_to_syn_num:dict, metric_name:str="jsd", feature_weights:list=[.25,.25,.25,.25]):
        # featue_weights example [.25,.25,.25,.25] => uniform means;[0,5,0,5] => uniform mean between second and fourth features
        # Less is better - [0,1]:
            # JSD : 0.1 ==> very simil to real dataset distributions
            # Hellinger Distance : 0.05 ==> very simil to real dataset distributions
            # Total Variation Distance : 0.2 ==> simil (not perfect) to real dataset distribution 
        # Higher is better:
            # Log-Likelihood 
        if metric_name not in self.get_metrics_names:
          raise RuntimeError("Unsupported metric name: {} compared to available metrics {}".format(metric_name,self.get_metrics_names))    
        
        if len(feature_weights) != self.features_num:
            raise RuntimeError("Unsupported feature weights: {} must have {} weights".format(feature_weights,self.features_num))
        aug_features = deepcopy(self.real_features)
        aug_labels = deepcopy(self.real_target)
        class_to_best_indexes = {}
        for t in target_to_syn_num:
            syn_num = target_to_syn_num[t]
            c_syn_indices = np.where(self.syn_targets == t)[0]
            val = self.get_metric(metric_name)[c_syn_indices]
            val_avg_features =  np.mean(val, axis=1) # out-shape: (syn_t_datasize,feature_num)
            weighted_avg_val = np.dot(val_avg_features, feature_weights) 
            if metric_name == "log_likeh":
                sorted_indices = np.argsort(-weighted_avg_val)#desc
            else:
                sorted_indices = np.argsort(weighted_avg_val)
            class_to_best_indexes[t] = [c_syn_indices[i] for i in sorted_indices]
            best_syn_num_t_indexes = [i for i in class_to_best_indexes[t][:syn_num]]
            aug_features = np.concatenate((aug_features,self.syn_features[best_syn_num_t_indexes]),axis=0)
            aug_labels = np.concatenate((aug_labels,self.syn_targets[best_syn_num_t_indexes]),axis=0)
        self.augmented_dataloader.dataset.images = aug_features
        self.augmented_dataloader.dataset.labels = aug_labels
        return self.augmented_dataloader, list(range(self.real_datasize,len(aug_labels))), class_to_best_indexes

    @staticmethod
    def ml_to_dl_format(images, features_num=4, pkt_num=10):
        if images.shape[1] != features_num * pkt_num:
            raise ValueError(f"Each Biflow Input in images must have {features_num * pkt_num} elements to convert to DL format.")
        return images.reshape(-1, features_num, pkt_num).transpose(0, 2, 1).reshape(-1, 1, pkt_num, features_num)
    
    @staticmethod
    def check_distributions(P, Q):
        assert P.shape == Q.shape
        assert np.isclose(P.sum(), 1) and np.isclose(Q.sum(), 1)

    @staticmethod
    def compute_jsd(real_distribution, synth_distribution):
        BiflowEvaluator.check_distributions(real_distribution, synth_distribution)

        M = 0.5 * (real_distribution + synth_distribution)

        def kl_divergence(P, Q):
            # Evita log(0) e divisioni per zero usando np.where
            mask = (P != 0) & (Q != 0)
            if not np.any(mask):
                return np.log(2)  # The two distributions are completely not overlapping
            return np.sum(P[mask] * np.log(P[mask] / Q[mask]))    
        
        jsd_value = 0.5 * kl_divergence(real_distribution, M) + 0.5 * kl_divergence(synth_distribution, M)
        return jsd_value / np.log(2)
    
    @staticmethod
    def compute_hd(real_distribution, synth_distribution):
        BiflowEvaluator.check_distributions(real_distribution, synth_distribution)
        return np.sqrt(np.sum((np.sqrt(real_distribution) - np.sqrt(synth_distribution)) ** 2)) / np.sqrt(2)

    @staticmethod
    def compute_tvd(real_distribution, synth_distribution):
        BiflowEvaluator.check_distributions(real_distribution, synth_distribution)
        return 0.5 * np.sum(np.abs(real_distribution - synth_distribution))
    
    @staticmethod
    def get_valutation_by_estimators(augmented_data_path, estimator_path, estimator_class_order, args) :
        df = pd.read_parquet(augmented_data_path)
        if args.estimator_type != "histogram":
            estimators = np.load(estimator_path,allow_pickle=True) 
        
        syn_indexes = df[df['Type']== 'syn'].index.tolist()
        
        images = np.stack(df['Features'].values)
        print(images[0])
        if images.ndim != 4:
            print("ok")
            images = BiflowEvaluator.ml_to_dl_format(images)
            print(images[0])
       
        labels = df['Targets'].values
        
        class_order = df['Order'].values[0]
        real_indexes = np.setdiff1d(np.arange(len(labels)), syn_indexes)
        print(len(labels),len(syn_indexes),len(real_indexes))
        real_features = images[real_indexes]
        real_target = labels[real_indexes]
        syn_features = images[syn_indexes]
        syn_targets =  labels[syn_indexes]
        real_datasize, _, pkt_num, features_num = real_features.shape
        syn_datasize = len(syn_targets)
        unique_classes, counts = np.unique(real_target, return_counts=True)
        majority_class = counts.argmax()
        
        class_num = len(unique_classes)
        
        real_pdfs =  np.empty( (class_num,),  dtype=object)
        syn_pdfs =  np.empty( (class_num,),  dtype=object)
        
        jsd = np.zeros( (syn_datasize, pkt_num, features_num),  dtype=float)
        log_likeh = np.zeros( (syn_datasize, pkt_num, features_num),  dtype=float)
        hellinger_dist = np.zeros( (syn_datasize, pkt_num, features_num),  dtype=float)
        tot_var_dist = np.zeros( (syn_datasize, pkt_num, features_num),  dtype=float)
        class_to_jsd = {}
        class_to_tvd = {}
        class_to_hd = {}
        class_jsd = np.empty( (class_num, features_num),  dtype=float)
        class_log_likeh = np.empty( (class_num,  features_num),  dtype=float)
        class_hellinger_dist = np.empty( (class_num,  features_num),  dtype=float)
        class_tot_var_dist = np.empty( (class_num, features_num),  dtype=float)
        class_jsd_std = np.empty( (class_num, features_num),  dtype=float)
        class_log_likeh_std = np.empty( (class_num,  features_num),  dtype=float)
        class_hellinger_dist_std = np.empty( (class_num,  features_num),  dtype=float)
        class_tot_var_dist_std = np.empty( (class_num, features_num),  dtype=float)
        metrix_indexes_to_aug_dataset_indexes = {}

        for idx, c in enumerate(unique_classes):
            if c == majority_class:
                continue
            c_indices = np.where(real_target == c)[0]
            c_syn_indices = np.where(syn_targets == c)[0]
            metrix_indexes_to_aug_dataset_indexes[c]= c_syn_indices
            r_features = real_features[c_indices]
            s_features = syn_features[c_syn_indices]
            # pdf estimations for class c
            match args.estimator_type:
                case "kd":
                    r_pdfs = kd.get_pdf(estimators[c], r_features)
                    s_pdfs =kd.get_pdf(estimators[c], s_features)
                case "gmm":
                    r_pdfs = gmm.get_pdf(estimators[c], r_features)
                    s_pdfs = gmm.get_pdf(estimators[c], s_features)
                case "histogram":
                    r_pdfs, s_pdfs = histogram.get_pdfs(real_features=r_features,syn_features=s_features,
                                                         n_bins=args.n_bin,n_pkt = pkt_num)
            real_pdfs[c] = r_pdfs
            syn_pdfs[c] = s_pdfs
            
            if args.estimator_type != "histogram": # evaluation for each syn sample
                for p in range(pkt_num):
                    for f in range(features_num):
                        pdfs_avg = np.mean(r_pdfs[:,p,f],axis=0)
                        for i in range(len(s_pdfs)):
                            pm = (pdfs_avg + s_pdfs[i,p,f]) / 2
                            real_kld = np.average([kl_div(p, pm) for p in r_pdfs[:,p,f]])
                            syn_kld = kl_div(s_pdfs[i,p,f], pm)
                            jsd[c_syn_indices[i],p,f] = BiflowEvaluator.compute_jsd(real_kld, syn_kld) 
                            hellinger_dist[c_syn_indices[i],p,f] = BiflowEvaluator.compute_hd(pdfs_avg, s_pdfs[i, p, f])
                            tot_var_dist[c_syn_indices[i],p,f] = BiflowEvaluator.compute_tvd(pdfs_avg, s_pdfs[i, p, f])
                
                match args.estimator_type:
                    case "kd":
                        log_likeh[c_syn_indices,:,:] = kd.log_likelihood(estimators[c], s_features)
                    case "gmm":
                        log_likeh[c_syn_indices,:,:] = gmm.log_likelihood(estimators[c], s_features)          
                        
                class_jsd[class_order[c]] = np.mean(jsd[c_syn_indices], axis=(0, 1))
                class_log_likeh[class_order[c]] = np.mean(log_likeh[c_syn_indices],axis=(0, 1))
                class_hellinger_dist[class_order[c]] = np.mean(hellinger_dist[c_syn_indices],axis=(0, 1))
                class_tot_var_dist[class_order[c]] = np.mean(tot_var_dist[c_syn_indices],axis=(0, 1))
                class_jsd_std[class_order[c]] = np.std(jsd[c_syn_indices], axis=(0, 1))
                class_log_likeh_std[class_order[c]] = np.std(log_likeh[c_syn_indices],axis=(0, 1))
                class_hellinger_dist_std[class_order[c]] = np.std(hellinger_dist[c_syn_indices],axis=(0, 1))
                class_tot_var_dist_std[class_order[c]] = np.std(tot_var_dist[c_syn_indices],axis=(0, 1))
            else: # histogram estimation: evaluation for each features by class
                    n_bin = len(s_pdfs)
                    temp_jsd,temp_hellinger_dist, temp_tot_var_dist = 0.0,0.0,0.0
                    for f in range(features_num): 
                        pdfs = r_pdfs[:, f]
                        js, hd, tvd = 0.0,0.0,0.0
                        non_zero_mask = pdfs != 0.0
                        pdfs_avg = np.mean(r_pdfs[non_zero_mask,f])
                        j=0
                        for i in range(n_bin):# histogram bin number
                            if s_pdfs[i,f] != 0.0:
                                pm = (pdfs_avg + s_pdfs[i,f]) / 2
                                real_kld = np.average([kl_div(p, pm) for p in r_pdfs[non_zero_mask,f]])
                                syn_kld = kl_div(s_pdfs[i,f], pm)
                                js += BiflowEvaluator.compute_jsd(real_kld, syn_kld)
                                hd += BiflowEvaluator.compute_hd(pdfs_avg, s_pdfs[i, f])
                                tvd += BiflowEvaluator.compute_tvd(pdfs_avg, s_pdfs[i, f])
                                j+=1
                        temp_jsd += js / (j-1)
                        temp_hellinger_dist += hd /(j-1)
                        temp_tot_var_dist += tvd / (j-1)
                  
                    class_to_jsd[class_order[c]] = temp_jsd / features_num
                    if class_to_jsd[class_order[c]] is None:
                         print(f"Errore: Valore mancante per la classe {c}, j={j-1} jsd={js}/j")
                    class_to_hd[class_order[c]] = temp_hellinger_dist / features_num
                    class_to_tvd[class_order[c]] = temp_tot_var_dist / features_num

        if args.estimator_type == "histogram":
            classes = list(class_to_jsd.keys())
            if class_order[majority_class] in classes:
                classes.remove(class_order[majority_class])
            print("Class to jsd:",class_to_jsd)
            result = {}         
            jsd,hd,tvd = 0.0,0.0,0.0
            for c in classes:
                result[f'jsd-{c}'] = [class_to_jsd[c]]
                result[f'hd-{c}'] = [class_to_hd[c]]
                result[f'tvd-{c}'] = [class_to_tvd[c]]
                jsd += class_to_jsd[c]
                hd += class_to_hd[c]
                tvd += class_to_tvd[c]
            result['jsd-mean'] = [jsd/len(classes)]
            result['hd-mean'] = [hd/len(classes)]
            result['tvd-mean'] = [tvd/len(classes)]
            return pd.DataFrame(result)
        
        simil_metric_by_class_df = pd.DataFrame({
            'hellinger_dist' : class_hellinger_dist.tolist(),
            "tvd" : class_tot_var_dist.tolist(),
            'jsd' : class_jsd.tolist(),
            'hellinger_dist_sd' : class_hellinger_dist_std.tolist(),
            "tvd_sd" : class_tot_var_dist_std.tolist(),
            'jsd_sd' : class_jsd_std.tolist()
         })
        return log_likeh, hellinger_dist, tot_var_dist, jsd, metrix_indexes_to_aug_dataset_indexes,simil_metric_by_class_df

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the Generator specific parameters"""
        parser = ArgumentParser(add_help=False)
        parser.add_argument('--exp-name', dest="exp_name", help='The experimental name', required=True)
        parser.add_argument('--estimator-approach', default='per-class', type=str,required=False, dest='estimator_approach',choices=['class-packet','class-features','packet','features'],
            help='Specifies the density estimator approach.  [default=%(default)s  means learn estimators for each features separately (learn class_num * pkt_num * feature_num estimators)]')
        parser.add_argument('--estimator-type', default='kd', type=str,required=False, dest='estimator_type',choices=['kd','gmm','histogram'],
            help='Specifies the density estimator method [default=%(default)s means kernel density estimation (KernelDensity) ]')
        temp_args,_ = parser.parse_known_args(args)
        match temp_args.estimator_type:
            case "kd":
                parser.add_argument('--kernel', default='gaussian', type=str,required=False, dest='kernel',choices=['gaussian','epanechnikov','exponential','cosine'],
                    help='The kernel to use (default=%(default)s)')
                parser.add_argument('--bandwidth', default=1.0, type=float,required=False, dest='bandwidth', help='The bandwidth of the kernel as float (default=%(default)s)')
                parser.add_argument('--metric', default="euclidean", type=str,required=False, dest='metric',
                    help='Metric to use for distance computation. See the documentation of scipy.spatial.distance and the metrics listed in distance_metrics for valid metric values. (default=%(default)s)')
            case "gmm":
                parser.add_argument('--init-method', default='k-means++', type=str,required=False, dest='init_method',choices=['kmeans', 'k-means++', 'random', 'random_from_data'],
                    help='The method used to initialize the weights, the means and the precisions. Responsibilities are initialized randomly if random.\n'+
                    'selects initial cluster centroids using sampling based on an empirical probability distribution of the points contribution to the overall inertia if k-means++.'+ 
                    'Algorithm: greedy k-means++ (several trials at each sampling step and choosing the best centroid among them)\n (default=%(default)s)')
                parser.add_argument('--covariance-type', default='full',  type=str,required=False, dest='cov_type',choices=[ 'full', 'tied', 'diag', 'spherical'],
                                    help='full: each component has its own general covariance matrix. tied: all components share the same general covariance matrix.\ndiag: each component has its own diagonal covariance matrix.+'
                                            'spherical: each component has its own single variance. (default=%(default)s)')
                parser.add_argument('--seed', default=0, type=int,required=False, 
                    help='Seed to use for distance computation.')
            case "histogram":
                parser.add_argument('--n-bin', default=100, type=int, required=False,  help='Bin number for feature binarization (default=%(default)s)')

        return parser.parse_known_args(args)
