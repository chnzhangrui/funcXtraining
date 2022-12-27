import numpy as np
from pdb import set_trace

def preprocessing(X_train, kin, name=None, reverse=False):
    if not reverse: # train
        if name == 'log10':
            X_train = np.log10(X_train) + 6 # 1keV => 0
            X_train /= kin
        elif name == 'neglog10plus1':
            X_train = - np.log10((X_train + 1) / kin)
        elif name is None:
            X_train /= kin
        else:
            raise NotImplementedError
    else: # evaluate
        if name == 'log10':
            X_train = np.power(10, X_train - 6)
            X_train *= kin
        elif name == 'neglog10plus1':
             X_train = np.power(10, -X_train) * kin - 1
        elif name is None:
            X_train *= kin
        else:
            raise NotImplementedError
    return X_train

