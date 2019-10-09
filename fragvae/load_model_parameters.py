"""
Created on Sun Jan 13 12:55:33 2019
@author: ja550
"""

import json
from rdkit.Chem import AllChem as Chem

def load_params(param_file=None,hyper_p={}, verbose=True):
    # Parameters from params.json and exp.json loaded here to override parameters set below
    hyper_p2 ={}
    if param_file is not None:
        try:
            hyper_p2.update( json.loads(open(param_file).read()))
            if verbose:
                print('Using hyper-parameters:')
                for key, value in hyper_p2.items():
                    print('{:25s} - {:12}'.format(key, str(value)))
                print('rest of parameters are set as default')
        except:
            print('')
            
                    
            
            
    parameters = {

        # Package paramaters
        "printMe":False,
        "excess_space":'                                                             ',
        
        #Master Model Paramters
        'FHO_interpretable':False, 
            
        #Training Params 
        'batch_size':30,
        'epoch_start': 0,
        'cont_atoms':False,
        'max_epoch':200,
        'expt_dataset': True,
        'train_dataset': 'Zinc15_No_BrI',
        'CV_dataset': 'Zinc15_No_BrI',
        
        # F1 Encoder paramaters
        'F1_encoder_activation':'relu',
        'F1_convo_width':10,
        'F1_convo_msg':10,
        'F1_encode_dropout':0,
        'finger_print': 10,
        'Node_decoder_len':4,
        'kl_loss_weight':0.001,
        'F1_msg_hid_dim':2,
        'F1_inner_hdd_dim':2,
        'NGF_reg':10,
        'NGF_F1_Sparse_layers':2,
        
        #F1 decoder paramaters
        'F1_decoder_activation':'tanh',
        'F1_decode_dropout':0.03,
        'F1_decode_reg':0.0001,
        'NN_decoder_len':5,
        'rand_sample':False,
        'sample_with_replacement':False,
        'decoder_0_expand':5,
        'decoder_1_expand':12,
        
        
        #'F1_decode_reg':0.05, is a reasonable value
        
        
        
        # FHO Encoder paramaters
        'FHO_encoder_activation':'relu',
        'FHO_convo_msg':10,
        'FHO_convo_width':10,
        'FHO_encode_dropout':0,
        'FHO_finger_print': 20,
        'max_rings':25,
        'FHO_hidden_atoms':0,
        'FHO_hidden_bonds':0,
        'Num_Graph_Convolutions':6,
        'FHO_num_sparse_layers':2,
        'z_bond_out_width':10,
        'bond_hid_dim':1,
        'update_bonds':True,
        'use_update_bonds_in_NGC':True,
        'inner_hdd_dim':2,
        'NGH_inner_hdd_dim':2, 
        'NGH_msg_hid_dim': 2,
        'Ring_F1_convo_width':7,
        'Ring_F1_convo_msg':7,
        'FHO_encode_reg':10,
        'Ring_bias_sparse':False,
        

        # FHO Decoder paramaters
        'FHO_decoder_activation':'tanh',
        'FHO_decode_dropout':0.05,
        'FHO_Input_NN_Shape':60,
        'FHO_depth':5,
        'FHO_decode_reg':0.0001,
        'FHO_max_P': True,
        'no_replacement':True,
        
        
        
        #FHO training Data
        'Leaf_decay':0.4,
        'FHO_attempts_per_batch':3,
        'error_correction':3,
        
        

        #FHO Varational layer
        'FHO_anneal_sigmod_slope':1,
        'FHO_kl_loss_weight':0.001,
        'FHO_anneal_epoch_start': 15,
        'FHO_anneal_sigmod_slope':1,
        
        #F1 Varational layer
        'F1_anneal_sigmod_slope':1,
        'F1_kl_loss_weight':0.001,
        'F1_anneal_epoch_start': 7,
        'F1_anneal_sigmod_slope':1,
        
        
        #RND forrest predict paramters Theoritical Examples
        
        'n_estimators':200,
        'min_samples_leaf':2,
        'max_features':0.5,
        'max_depth':None, #was 5
        
        #RND forrest prediction parameters Experimental Examples
        
        'n_estimators_expt': [300],
        'max_depth_expt': [4], 
        'max_features_expt': [0.25],
        'rnd_boostrap':[True],
        'num_features_expt': [60],
        'min_samples_leaf_expt':[1],
        
        
        #RND forrest prediction parameters Experimental Examples
            
        'Pred_NN_layers': [2,1,0],
        
        'Pred_NN_Dropout': [0.1,0.2],
        'Pred_NN_regl2':[0.01,0.02,0.04,0.1],
        'Pred_NN_Dense_layer_size':[60],
        'Pred_NN_growth': [1.2], 

        #Chemcical Vector Description
        'num_bond_features': 4,
        'KekulizeBonds': False,
        'max_atoms': 60,
        'max_dangle_atoms':60,
        'max_bonds' : 60, 
        'max_degree': 4,

        "atoms" : ["S", "O", "H", "C","N",  "Cl", "F" ],
        "charges":  [-1,0,1],
        "degrees":  [1,2,3,4,5,6],
        "valence": [1,2,3,4,5,6],
        "include_charges":False,
        "include_degrees":False,
        "include_valence":False,
        "random_shuffle":True,

        "bonds" : [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]   , 
        
        #Will  be depricated parameters
        'True_Test':False,
        'Zinc15_dir':'//np-nobelium/OE_Personal/ja550/Desktop/Molecular Catalogs/data'
        
        #'NGF_reg':1
        }
    # overwrite parameters
    parameters.update(hyper_p2)
    parameters.update(hyper_p)
    
    
    parameters["bonds"] = [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]    
    if(parameters['KekulizeBonds']):
        parameters["bonds"] = [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE]    
        parameters['num_bond_features'] = len(parameters["bonds"])
        

    parameters['num_atom_features'] = calc_num_atom_features(parameters)

    return parameters

def save_model_params(parameters,param_file):
    
    try:
        json.dump(parameters, open(param_file,'x'))
        print('Generate new File '+param_file)
    except:
        json.dump(parameters, open(param_file,'w'))
        print('Overwite file '+param_file)

def calc_num_atom_features(params):
    num_atom_features = len(params['atoms'])
    if(params['include_charges']):
        num_atom_features = num_atom_features+len(params['charges'])
    if(params['include_degrees']):
        num_atom_features = num_atom_features+len(params['degrees'])
    if(params['include_valence']):
        num_atom_features = num_atom_features+len(params['valence'])
    return num_atom_features