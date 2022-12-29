import numpy as np
import re
from pdb import set_trace

def preprocessing(X_train, kin, name=None, reverse=False, input_file=None):
    if not reverse: # train
        if name == 'neglog10plus1':
            X_train = - np.log10((X_train + 1) / kin)
        elif re.compile("^log10.([0-9.]+)+$").match(name): # log10.x
            from common import split_energy, get_energies
            X_train = np.log10((X_train / kin) + 1)
            _, xtrain_list = split_energy(input_file, X_train)
            _, kin_list = split_energy(input_file, kin)
            high = float(re.compile("^log10.([0-9.]+)+$").match(name).groups()[0])
            print('scale to', high)
            scale = []
            for k, v in zip(kin_list, xtrain_list):
                scale.append((k[0].item(), int(high/np.sort(v.flatten())[-3])))
            scale = dict(scale)
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] *= s
            return X_train, scale
        elif name is None:
            X_train /= kin
        else:
            raise NotImplementedError
    else: # evaluate
        if name == 'neglog10plus1':
             X_train = np.power(10, -X_train) * kin - 1
        elif re.compile("^log10.([0-9.]+)+$").match(name): # log10.x
            import json, tensorflow as tf
            with open(input_file, 'r') as fp:
                scale = json.load(fp)
            scale = dict([(float(k), v) for k,v in scale.items()])
            X_train = X_train.numpy()
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] /= s
            X_train = (np.power(10, X_train) - 1) * kin
            X_train = tf.convert_to_tensor(X_train)
        elif name is None:
            X_train *= kin
        else:
            raise NotImplementedError
    return X_train

