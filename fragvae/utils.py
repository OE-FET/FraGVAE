# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:07:31 2019

@author: ja550
"""

       
       


from .hyperparameters import load_params, calc_num_atom_features,save_params
from .FHO_models import *
from .F1_models import *
import numpy as np
from .convert_mol_smile_tensor import *
from .generators import *

'''       
params = load_params()
g = FHO_data_generator(params,return_smiles = False, cross_vald = False)
#g = data_generator(params)
t = time.time()
data = next(g)  
print( time.time() - t)
     

'''

    
def reset_weights(model):
    session=tf.Session()
    for layer in model.layers: 
        print(layer)
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                weight.initializer.run(session=session)
    return model
    
def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass
    

def gen_dropout_fun(model):
    # for some model with dropout ...
    inputs = []
    for input_layer in model._input_layers:
        inputs.append(input_layer.input)
    inputs = inputs+[ tf.keras.backend.learning_phase()]
    
    outputs = []
    for output_layer in model._output_layers:
        outputs.append(output_layer.output)
    
    f = tf.keras.backend.function(inputs,  outputs)
    return f



     
def predict_with_uncertainty(f, x, num_class, n_iter=100):
    result = np.zeros((n_iter,) + (x[0].shape[0], num_class) )
    
    
    X = tuple(x+[1])
    for i in range(n_iter):
        #result[i,:, :] = f((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11], 1))[0]
        result[i,:, :] = f(X)[0]

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty


            
def calc_Tanimoto(z1,z2,params):
    
    z1 = z1[0]
    z2 = z2[0]
    
    if(params['FHO_decoder_activation']=='elu'or params['FHO_decoder_activation']=='tanh'):
        z1 = np.maximum(z1 + np.ones_like(z1),np.zeros_like(z1))
        z2 = np.maximum((z2 + np.ones_like(z2)),np.zeros_like(z2))
    
    

    tanimoto_max = np.maximum(z1,z2)
    tanimoto_min = np.minimum(z1,z2)
    
    tanimoto_max = np.sum(tanimoto_max,axis=-1)
    tanimoto_min = np.sum(tanimoto_min,axis=-1)
    
    tanimoto_val = tanimoto_min/(tanimoto_max+1E-20)
    
    return tanimoto_val

    
    
def is_valid_substructure(mol_subs,mol,NN_Tensor_summed,NN_Tensor_summed_recon,map_edmol_Idx_to_mol):
    match_found = False
    if(mol!=-1):
        matches = mol.GetSubstructMatches(mol_subs,uniquify=False)
        valid_matches = []
       
        for match in matches:
            match_found = True
            for sub_Idx in range(0,len(match)):
                if(not(all(NN_Tensor_summed_recon[match[sub_Idx]] - NN_Tensor_summed[map_edmol_Idx_to_mol[sub_Idx]]==0))):
                    match_found = False
                    break
            
            if(match_found):
                break
 
    
    
    
    return match_found    
    
    
    

       






