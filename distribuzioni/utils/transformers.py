import numpy as np
from copy import deepcopy
from sklearn.impute import SimpleImputer
from statistics import stdev
import random

"""
Some has been implemented from https://arxiv.org/abs/2007.15951
"""

def duplicating(x):
    """
    It simply duplicates x.
    """
    return deepcopy(x)

def jittering(x, u:float=0, s:float=.03, features=[0,1,3], random_state=None):
    """
    Adds different random values rho_i~N(u,s) to the selected features.
    x: the sample (normalized) to transform.
    u, s: parameters for ~N(u,s)
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    """
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    mask = np.zeros_like(x)
    jittering_matrix = execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=x.shape), random_state)
    mask[..., features] = jittering_matrix[..., features]
    
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 0

    # Forcing padding-mask values to 0
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 0 
    else:
        mask[:, idx, features] = 0
        
    
    x_tr = np.abs(x + mask).astype(x.dtype)
    for x in x_tr:
        x[np.where(x > 1)] = 1

    return x_tr

def translating(x, u:float=0, s:float=.03, features=[0,1,3], random_state=None):
    """
    Adds the same random value rho~N(u,s) to the selected features.
    x: the sample (normalized) to transform.
    u, s: parameters for ~N(u,s)
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    """
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))

    mask = np.zeros_like(x)
    traslation_matrix = np.ones_like(x) * execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=None), random_state)
    mask[..., features] = traslation_matrix[..., features]
    
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 0
    
    # Forcing padding-mask values to 0
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 0 
    else:
        mask[:, idx, features] = 0
        
    
    if not ((x + mask) >= 0).all():
        mask[..., features] = np.abs(mask[..., features])

    x_tr = (x + mask).astype(x.dtype)
    for x in x_tr:
        x[np.where(x > 1)] = 1

    return x_tr

def scaling(x, u:float=1, s:float=.2, features=[0,1,3], random_state=None):
    """
    Scales the selected features using the same random value rho~N(u,s).
    x: the sample (normalized) to transform.
    u, s: parameters for ~N(u,s)
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    """
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    mask = np.ones_like(x)
    scaling_matrix = np.ones_like(x) * execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=None), random_state)
    mask[..., features] = scaling_matrix[..., features]

    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 1

    # Forcing padding-mask values to 1
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 1 
    else:
        mask[:, idx, features] = 1
        
    
    x_tr = (x * mask).astype(x.dtype)
    for x in x_tr:
        x[np.where(x > 1)] = 1

    return x_tr 

def warping(x, u:float=1, s:float=.2, n_knot:int=4, features=[0,1,3], random_state=None):
    """
    Scales the selected features using a spline with n_knot values rho_i~N(u,s).
    x: the sample (normalized) to transform.
    u, s: parameters for ~N(u,s)
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    """
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    knots = execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=n_knot), random_state)
    n_points = x.shape[1]
    mask = np.ones_like(x)
    warping_matrix = np.interp(
        np.arange(n_points), np.linspace(0, n_points - 1, n_knot), knots
        ).repeat(x.shape[-1], axis=-1).reshape(x.shape)
    mask[..., features] = warping_matrix[..., features]

    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 1

    # Forcing padding-mask values to 1
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 1
    else:
        mask[:, idx, features] = 1
        

    x_tr = (x * mask).astype(x.dtype)
    for x in x_tr:
        x[np.where(x > 1)] = 1
    return x_tr

def slicing(x, wr:float=.9, pr=None, fill_value:float=.0, features=[0,1,3]):
    """
    Randomly hides a portion of time-series.
    x: the sample (normalized) to transform.
    wr: ratio of non hidden packets. E.g., given a length 10 ts and wr=.6, a window of size 6 is not hidden.
    pr: ratio of starting position. E.g., given a length 10 ts and pr=.1, the hiding window starts from position 1.
        If None (default), a random starting point is uniformely selected.
    fill_value: value used to mask. Can be a real (one for all the features) or a list of values (one per feature).
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    """
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    if not isinstance(fill_value, (np.ndarray, list, set)):
        fill_value = [fill_value] * x.shape[-1]
    n_points = x.shape[1]
    w = max(int(np.floor(n_points * wr)), 1)
    pos = np.random.randint(n_points) if pr is None else int(pr * n_points)
    sliced_x = deepcopy(x)
    if w + pos <= n_points:
        sliced_x[..., :pos, features] = np.nan
        sliced_x[..., pos + w:, features] = np.nan
    else:
        sliced_x[..., pos + w - n_points:pos, features] = np.nan
    fill_matrix = np.repeat(fill_value, n_points).reshape(x.shape[1:][::-1]).T.reshape(x.shape)
    sliced_x[np.isnan(sliced_x)] = fill_matrix[np.isnan(sliced_x)]
    return sliced_x

def features_hiding(x, features='all', fill_value:float=0):
    """
    Hides the selected feature.
    x: the sample (normalized) to transform.
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    fill_value: value used to mask. Can be a real (one for all the features) or a list of values (one per feature).
    """
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        return (np.ones_like(x) * fill_value).astype(x.dtype)
    mask = np.ones_like(x)
    mask[..., features] = 0
    return (x * mask + fill_value * (1 - mask)).astype(x.dtype)


#AMPLITUDE AUGMENTATIONS

def gaussian_noise(x, features=[0,1,3], random_state=None, a:str='1'):
    """
    Add independently sampled Gaussian noise to the selected features.
    x(d,t)= x(d,t) +ε(t) where εt∼N(0, α{σy(d,t)}^2)
    x: the sample (normalized) to transform.
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    a: magnitude parameter
    """
    check_features(features)
    a = eval(a)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    mask = np.zeros_like(x)

    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
        temp=[execute_revert_random_state(np.random.normal, dict(loc=0, scale=(get_standard_deviation(x,d)**2)*a, size=(1,10,1)), random_state) for d in features ]      
        mask = np.concatenate(np.asarray(temp),2) 
    else:
      for d in features:
        mask[0,:,d] = execute_revert_random_state(
            np.random.normal, dict(loc=0, scale=(get_standard_deviation(x,d)**2)*a, size=(1,1,10)), random_state)    
    
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 0

    # Forcing padding-mask values to 0
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 0 
    else:
        mask[:, idx, features] = 0
        

    x_tr = np.abs(x + mask).astype(x.dtype)
    for x in x_tr:
        x[np.where(x > 1)] = 1.  
    return x_tr
def spike_noise(x, features=[0,1,3], random_state=None, a:str='1'):
    """
    Add independently sampled Gaussian noise to the selected features.
    x(d,t)= x(d,t) + |ε(t)| where ε(t)∼N(0, α{σy(d,t)}^2)
    x: the sample (normalized) to transform.
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    a: magnitude parameter
    """
    check_features(features)
    a = eval(a)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    mask = np.zeros_like(x)

    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
        temp = [execute_revert_random_state(np.random.normal, dict(loc=0, scale=(get_standard_deviation(x,d)**2)*a, size=(1,10,1)), random_state) 
                for d in features]
        mask = np.concatenate(np.asarray(temp),2)      
    else:
      for d in features:
        mask[0,:,d] = execute_revert_random_state(
            np.random.normal, dict(loc=0, scale=(get_standard_deviation(x,d)**2)*a, size=(1,1,10)), random_state)    
    
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 0

    # Forcing padding-mask values to 0
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 0 
    else:
        mask[:, idx, features] = 0


    mask = np.abs(mask)
    
    x_tr = (x + mask).astype(x.dtype)

    for x in x_tr:
        x[np.where(x > 1)] = 1.  
    return x_tr

print('Warning: IAT is not considered for gaussian_wrap transformation.')

def gaussian_wrap(x, features=[0, 1, 3], random_state=None, a:str='1'):
    """
    Scale the selected features by independently sampled Gaussian values.
    Sample a feature and multiply Gaussian noise to its values:
    x(d,t)= x(d,t) * ε(t) where ε(t)∼N(1+0.01*α, 0.02*α{σy(d,t)}^2)
    x: the sample (normalized) to transform.
    a: magnitude parameter
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    """
    check_features(features)
    a = eval(a)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)

    avg = (a * 0.01) + 1
    mask = np.ones_like(x)

    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
        temp = [execute_revert_random_state(
            np.random.normal, 
            dict(loc=avg, scale=(get_standard_deviation(x, d)**2) * a * 0.02, size=(1, 10, 1)), random_state)
            for d in features]
        mask = np.concatenate(np.asarray(temp),2)      
    else:
        for d in features:
            stand = (get_standard_deviation(x, d)**2) * a * 0.02
            mask[0, :, d] = execute_revert_random_state(
                np.random.normal, dict(loc=avg, scale=stand, size=(1, 1, 10)), random_state)    
    # Forcing the first IAT to the original value
    # mask[:, 0, 1] = 1

    # Forcing padding-mask values to 1
    idx = np.where(x[0, :, 0] == 0)[0]
    mask[:, idx, features] = 1
    
    x_tr = (x * mask).astype(x.dtype)
    for x in x_tr:
        x[np.where(x > 1)] = 1.

    return x_tr


def constant_wrap(label, x, dev_dict, features=[0,1,3], a:str='1', random_state=None):
    """
    Scale the selected features by a single randomly sampled value
    Sample a single uniformly sampled value ϵ∼U[c, b] and perform xi· ϵ to all xi of selected feature with
    c=1+σy(d,:)*(0.06-0.02*alpha); b = 1+σy(d,:)*(0.14+0.02*alpha)
    dev_dict: the dict containing pairs (label,d):σy(d,:)
    label: the sample label
    x: the sample (normalized) to transform.
    a: magnitude parameter
    features: if 'all' (default) is applied to each feature.
        Otherwise should be a list of indexes that indicates the features
        on which the transformation is applied. e.g. if --fields is PL IAT,
        features=[0] enforces the transformation only on the PL.
    random_state: seed for the generation of random variables.
    """
    
    check_features(features)
    a = eval(a)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    
    mask = np.ones_like(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
        temp=[ execute_revert_random_state(np.random.uniform, 
                                           dict(low=dev_dict[(label,d)]*(0.06-0.02*a)+1, high=dev_dict[(label,d)]*(0.14+0.02*a)+1, 
                                                size=(1,10,1)),random_state) 
               for d in features ] 
        mask = np.concatenate(np.asarray(temp),2)
    else:
      for d in features:
        s = dev_dict[(label,d)]
        start = s*(0.06-0.02*a)+1
        end = s*(0.14+0.02*a)+1
        mask[0,:,d] = execute_revert_random_state(
            np.random.uniform, dict(low=start, high=end, size=(1,1,10)), random_state)  
    
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 1

    # Forcing padding-mask values to 1
    idx = np.where(x[0, :, 0] == 0)[0]
    if len(features) > 1:
      mask[:, idx[:, np.newaxis], features] = 1
    else:
        mask[:, idx, features] = 1
        
    
    x_tr = (x * mask).astype(x.dtype)

    for x in x_tr:
        x[np.where(x > 1)] = 1.    
    return x_tr


#MASKING AUGMENTATIONS


def bernulli_mask(x, a:str='1',random_state=None):
    """
    Random masking values
    Details: Independently set to zero feature values by sampling a Bernoulli(p = 0.6α)
    x: the sample (normalized) to transform.
    a: magnitude parameter.
    random_state: seed for the generation of random variables.
    """
    a = eval(a)
    mask = np.zeros_like(x)
    mask = execute_revert_random_state(
        np.random.binomial, dict(size=x.shape,n=1, p=0.6*a), random_state)

    x_out = np.abs(x * mask).astype(x.dtype)
    for x in x_out:
        x[np.where(x > 1)] = 1.     
    return x_out

def window_mask(x, a:str='1', random_state=None):
    """
    Masking the same sequences across all features
    Details: Given a configured maximum size W=⌊1+2.5α⌉, sample a window length w∼U[1, W]
    and a random starting point t=U[0, T − w] and set to zero all x(:,t) falling in the sampled window
    x: the sample (normalized) to transform.
    a: magnitude parameter.
    random_state: seed for the generation of random variables.
    """
    a = eval(a)
    x_out = deepcopy(x)
    _, T, *_= np.shape(x)
    W = round(2.5*a+1)
    w = execute_revert_random_state(
        np.random.randint, dict(low=1,high=W+1, size=None), random_state)
    start = execute_revert_random_state(
        np.random.randint, dict(low=0,high=T-w+1, size=None), random_state)

    x_out[:,start:start+w,0] = 0  # PL
    x_out[:,start:start+w,1] = 0.5  # DIR

    return x_out

#PACKET ORDER AUGMENTATIONS

# def flip(x,axis=1):
#   """
#   Reverse the order of elements in an array along the given axis
#   x: the sample (normalized) to transform.
#   axis: Axis or axes along which to flip over. The default, axis=1, will flip over the packet axis of the input array (precisely axis 1)
#   If axis is negative it counts from the last to the first axis.
#   If axis is a tuple of ints, flipping is performed on all of the axes specified in the tuple.
#   """
#   return np.flip(x,axis)

def interpolate(x, random_state=None):
  """
  Densify time series by injecting average values and then sample a new sequence of length T
  Details: Expand each feature by inserting the average 0.5(x(d,t) +x(d,t+1)) in-between each pair of values.
  Then randomly select a starting point t=U[0, T − 1] and extract the following T values for all features x(:,t:t+T )
  x: the sample (normalized) to transform.
  random_state: seed for the generation of random variables.
  """
  _, T, _ = np.shape(x)
  x_out = deepcopy(x)
  i = 0
  while True:
    if i >= T+(T-2):
      break
    x_out = np.insert(x_out, i+1, np.mean(x_out[:,i:i+2,:],1), axis=1)
    i=i+2
 
  start = execute_revert_random_state(
          np.random.randint, dict(low=0, high=T, size=None), random_state)

  return x_out[:,start:start+T,:]

def cut_mix(x, X, random_state=None):
  """
  Swap segments of two different samples
  Details: Given a training mini-batch, define a sample s1 by sampling without replacement.
  Then sample a segment of length w∼U[0, T − 1] starting at t∼U[0, T − 1 − w] and swap the segment of
  each feature between x and s1 (no magnitude needed).
  https://arxiv.org/pdf/:1905.04899.pdf
  x: the sample (normalized) to transform.
  random_state: seed for the generation of random variables.
  """
  x_out = deepcopy(x)
  _, T, *_ = np.shape(x)
  rng = np.random.default_rng()
  s1 = execute_revert_random_state(
        rng.choice, dict(a=X, size=1, axis=0, replace=False), random_state)
  w = execute_revert_random_state(
        np.random.randint, dict(low=0,high=T, size=None), random_state)
  start = execute_revert_random_state(
            np.random.randint, dict(low=0,high=T-w, size=None), random_state)
  x_out[:,start:start+w,:] = s1[0,:,start:start+w,:]

  return x_out

def pkt_translating(x, a:str='1', random_state=None):
    """
    Move a segment to left or the right
    Details: Define N =1+arg maxi{ai ≤ α} where ai ∈ {0.15, 0.3, 0.5, 0.8} and sample
    n ∼ U[1, N]. Then, sample a direction b ∈ {left, right} and a starting point t∼U[0, T]:
    If b = left, left shift each feature values n times starting from t  else right shift each
    feature values n times starting from t.(replace shifted values with the single value x(d,t))
    x: the sample (normalized) to transform.
    random_state: seed for the generation of random variables.
    a: magnitude parameter
    """
    a = eval(a)
    weights = [0.15, 0.3, 0.5, 0.8]

    try:
       N = 1 + np.argmax([w for w in weights if w <= a])
    except ValueError:
       N = 1
    
    _, T, *_ = np.shape(x)
    n = execute_revert_random_state( np.random.randint, dict(low=1,high=N+1, size=None), random_state)
    rng = np.random.default_rng()
    b = execute_revert_random_state(rng.choice, dict(a=['L','R'],size=1), random_state)
    t = execute_revert_random_state( np.random.randint, dict(low=0,high=T+1, size=None), random_state)
    x_out = deepcopy(x)
    if b == 'R':
      for i in range(t,T):
        if (i+n) >= T:
          break
        else:
          x_out[0,i+n,:] = x[0,i,:]
    else:
      for i in range(t, 0, -1):
        if (i-n)<0:
          break
        else:
          x_out[0,i-n,:] = x[0,i,:]
    return x_out

def single_interpolate(x, i):
  """
  x:sample
  i: packet index
  """
  _, T, D = np.shape(x)
  x_out = np.zeros_like(x)
  for j in range(D):
    if(i >= T-1):
      break
    x_out[0,i,j] = np.mean([x[0,i,j],x[0,i+1,j]])
  return x_out[0,i,:]

def wrap(x,a:str='1'):
  """
  Mixing interpolation, drop and no change
  Details: Compose a new sample x′ by manipulating each x(:,t) based on three options with
  probabilities P[interpolate] = P[discard] = 0.5α and P[nochange] = 1−α.
  If “nochange” then keep x(:,t); if “interpolate” then keep x(:,t) and x(:,t) = 0.5(x(:,t) + x(:,t+1));
  if “nochange” then do nothing.
  Stop when |x'| = (packet num per features) or apply tail padding (if needed).
  x: the sample (normalized) to transform.
  random_state: seed for the generation of random variables.
  a: magnitude parameter
  """
  a = eval(a)
  _,T, *_ = np.shape(x)
  x_out = np.zeros_like(x)
  for i in range(T):
    func = np.random.choice([single_interpolate,lambda x,t: np.zeros_like(x[0,t,:]),lambda x,t: x[0,t,:] ],p=[0.5*a, 0.5*a, 1-a])
    x_out[0,i,:] = func(x,i)
  return x_out

def permutation(x, packet_axis=1, random_state=None, a:str='1'):
    """
    Segment the time series and reorder the segments
    Details: Define N=2+arg maxi{ai ≤ α} where ai ∈{0.15, 0.45, 0.75, 0.9}, a sample
    n∼U[2, N] and split the range [0:T-1] into n segments of random length. Compose a
    new sample x' by concatenating  x(:,t) from a random order of segments
    x: the sample (normalized) to transform.
    packet_axis: packet axis of x
    random_state: seed for the generation of random variables.
    a: magnitude parameter
    """
    a = eval(a)
    weights = [0.15, 0.45, 0.75, 0.9]
    
    try:
       N = 2 + np.argmax([w for w in weights if w <= a])
    except ValueError:
       N = 2

    if N == 2:#split into one segment
       n = 1
    else:
       n = execute_revert_random_state( np.random.randint, dict(low=2,high=N+1, size=None), random_state)
    rng = np.random.default_rng()
    temp = np.array_split(x,n,1) #split into n segments of random length
    rng.shuffle(temp) #get random order of segments
    out = np.concatenate(temp,packet_axis)
    return out

def dup_rto(x,p:float=0.1,a:str='1', random_state=None,Lmin:int=1,Lmax:int=9):
  """
  Packet Subsequence Duplication Augmentation via RTO  Mimic TCP pkt retrans due to 
  timeout by duplicating values  Details: Duplicating a range of packets according to a 
  Bernoulli(p = 0.1α)  https://cloud.tsinghua.edu.cn/f/7f250d2ffce8404b845e/?dl=1. Algorithm 1
  p: packet loss rate
  x: the sample (normalized) to transform.
  random_state: seed for the generation of random variables.
  a: magnitude parameter
  [Lmin,Lmax]:Range of Lost Packets
  """
  a = eval(a)
  p = p*a
  _, T, D  = x.shape
  i=0
  temp_list = []
  x_out = []
  rng = np.random.default_rng(seed=random_state)
  rand_values = execute_revert_random_state(rng.uniform, dict(low=0.0, high=1.0,size=T), random_state)
  while i < T:
    rand = rand_values[i]
    if rand < p: #if packets are lost
        if i == T-1:
          temp_list.append(x[:,i,:])
          x_out.append(x[:,i,:])
          break
        L = execute_revert_random_state( np.random.randint, dict(low=Lmin,high=Lmax+1, size=None), random_state)
        upper = min(i + L + 1, T)
        temp_list.append(x[:,i:upper,:].tolist())        
        x_out.append(x[:,i:upper,:].tolist())
        i = upper
    else:
        x_out.append(x[:,i,:])
        if temp_list:
          x_out.append(temp_list)
          temp_list = []
        i +=1
  if temp_list:
    x_out.append(temp_list)

  out = np.zeros_like(x)
  output_len = 0
  for item in x_out:
      if output_len >= T:
        break
      elif hasattr(item, 'ndim'):
        out[:,output_len,:] = item
        output_len += 1
      else:
        if len(item) > 1:
          temp =  np.concatenate( [l.reshape((1,1,D)) if isinstance(l,np.ndarray) 
                                   else np.asarray(l) for l in item], 
                                  axis=1 )
        else:
          temp = np.array(item)
          if temp.ndim > 3:
           temp = temp[0]
        item_pkt_num = temp.shape[1]        
        upper = T - output_len
        if item_pkt_num > upper:
          out[:,output_len:output_len+upper,:] = temp[:,:upper,:]
          break
        else:
          out[:,output_len:output_len+item_pkt_num,:] = temp
          output_len += item_pkt_num

  return out

def dup_fast_retransmit(x, p: float = 0.1, a: float = 1, random_state=None):
    """
    Packet Subsequence Duplication Augmentation via Fast Retransmit
    Mimic TCP fast retrans by duplicating values
    Details: Duplicating one packet according to a Bernoulli(p = 0.1α)
    https://cloud.tsinghua.edu.cn/f/7f250d2ffce8404b845e/?dl=1. Algorithm 2
    x: the sample (normalized) to transform.
    random_state: seed for the generation of random variables.
    p: packet loss rate
    a: magnitude parameter
    """
    T = x.shape[1]
    p = p * a
    rng = np.random.default_rng(random_state)
    flags = np.zeros(T, dtype=bool)
    x_out = np.zeros_like(x)
    rand_values = execute_revert_random_state(rng.uniform, dict(low=0.0, high=1.0, size=T), random_state)
    j = 0
    for i in range(T):        
        if j >= T:
            break
        x_out[:, j, :] = x[:, i, :]
        j += 1
        rand = rand_values[i]
        if rand <= p : #the pkt is lost
          flags[i] = True
        else:
          for index in np.flatnonzero(flags):
            if j < T:
              x_out[:, j, :] = x[:, index, :]
              flags[index] = False
              j += 1
            else:
              break        
    return x_out

def rto_shift(x,p:float=0.1,a:str='1', random_state=None,Lmin:int=1,Lmax:int=9):
  """
  Packet Subsequence Shift Augmentation via RTO
  Shows packet subsequence shift augmentation via the RTO mechanism.
  it's similar to dup_rto except considers the capture operation after the packet loss
  (dup_rto considers the capture operation before the packet loss)
  https://cloud.tsinghua.edu.cn/f/7f250d2ffce8404b845e/?dl=1. Algorithm 3
  x: the sample (normalized) to transform.
  random_state: seed for the generation of random variables.
  p: packet loss rate
  a: magnitude parameter
  [Lmin,Lmax]:Range of Lost Packets
  """
  a = eval(a)
  p = p*a
  _, T, D  = x.shape
  i=0
  temp_list = []
  x_out = []
  rng = np.random.default_rng(seed=random_state)
  rand_values = execute_revert_random_state(rng.uniform, dict(low=0.0, high=1.0,size=T), random_state)
  while i < T:
    rand = rand_values[i]
    if rand < p: # packets are lost
        if i == T-1:
          temp_list.append(x[:,i,:])
          x_out.append(x[:,i,:])
          break
        L = execute_revert_random_state( np.random.randint, dict(low=Lmin,high=Lmax, size=None), random_state)
        upper = min(i + L + 1, T)
        temp_list.append(x[:,i:upper,:].tolist()) 
        i = upper
    else:
        x_out.append(x[:,i,:])
        if temp_list:
          x_out.append(temp_list)
          temp_list = []
        i +=1
  if temp_list:
    x_out.append(temp_list)

  out = np.zeros_like(x)
  output_len = 0
  for item in x_out:
      if output_len >= T:
        break
      elif hasattr(item, 'ndim'):
        out[:,output_len,:] = item
        output_len += 1
      else:
        if len(item) > 1:
          temp = np.concatenate([l.reshape((1,1,D)) if isinstance(l,np.ndarray) 
                                 else np.asarray(l) for l in item],
                                 axis=1 )
        else:
          temp = np.array(item)
          if temp.ndim > 3:
           temp = temp[0]
        item_pkt_num = temp.shape[1]        
        upper = T - output_len
        if item_pkt_num > upper:
          out[:,output_len:output_len+upper,:] = temp[:,:upper,:]
          break
        else:
          out[:,output_len:output_len+item_pkt_num,:] = temp
          output_len += item_pkt_num

  return out

def fast_retransmit_shift(x,p:float=0.1,a:str='1', random_state=None):
    """
    Shows packet subsequence shift augmentation via the fast retransmit mechanism.
    It's similar to dup_fast_retransmit except that retransmit_shift considers the
    capture operation after the packet loss (dup_fast_retransmit considers the
    capture operation before the packet loss)
    https://cloud.tsinghua.edu.cn/f/7f250d2ffce8404b845e/?dl=1. Algorithm 4
    x: the sample (normalized) to transform.
    random_state: seed for the generation of random variables.
    p: packet loss rate
    a: magnitude parameter
    """
    a = eval(a)
    T = x.shape[1]
    p = p * a
    rng = np.random.default_rng(random_state)
    flags = np.zeros(T, dtype=bool)
    x_out = np.zeros_like(x)
    rand_values = execute_revert_random_state(rng.uniform, dict(low=0.0, high=1.0, size=T), random_state)
    j = 0
    for i in range(T):        
        if j >= T:
            break
        rand = rand_values[i]
        if rand <= p : #the pkt is lost
          flags[i] = True
        else:
          x_out[:, j, :] = x[:, i, :]
          j += 1
          for index in np.flatnonzero(flags):
            if j < T:
              x_out[:, j, :] = x[:, index, :]
              flags[index] = False
              j += 1
            else:
              break        
    return x_out

def check_features(features):
    if isinstance(features, str):
        assert features in ['none', 'all']
    else:
        assert isinstance(features, (np.ndarray, list, set))

def execute_revert_random_state(fn, fn_kwargs=None, new_random_state=None):
    """
    Execute fn(**fn_kwargs) without impacting the external random_state behavior.
    """
    old_random_state = np.random.get_state()
    np.random.seed(new_random_state)
    ret = fn(**fn_kwargs)
    np.random.set_state(old_random_state)
    return ret

def get_avarage_standard_deviation(images, labels, y, feature_index):
    """
    Return the average standard deviation for feature and class passed
    y: the class for which you want to calculate the standard deviation
    feature_index: the index of the feature for which to calculate the standard deviation
    """
    indexes = [i for i, l in enumerate(labels) if l == y]
    return np.std(images[indexes,0,:,feature_index])

def get_standard_deviation(image, feature_index):
    """
    Returns the image standard deviation for passed feature
    feature_index: the index of the feature for which to calculate the standard deviation
    """
    return np.std(image[0,:,feature_index])

def get_images(sel_loader):
    if hasattr(sel_loader.dataset, 'labels'): 
        images = np.asarray(sel_loader.dataset.images)
    elif hasattr(sel_loader.dataset, 'datasets'):
        images = []
        for ds in sel_loader.dataset.datasets:
            images.extend(ds.images)
        images = np.array(images)
    else:
        raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
    return images

def get_labels(sel_loader):
    if hasattr(sel_loader.dataset, 'labels'):  
        labels = np.asarray(sel_loader.dataset.labels)
    elif hasattr(sel_loader.dataset, 'datasets'):
        labels = []
        for ds in sel_loader.dataset.datasets:
            labels.extend(ds.labels)
        labels = np.array(labels)
    else:
        raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
    return labels

