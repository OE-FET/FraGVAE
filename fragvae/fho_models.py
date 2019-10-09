# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:26:50 2019

@author: ja550
"""


from .layers import  NeuralGraphSparsify, NeuralGraphHidden,Variational,Find_Ring_Bonds,Find_Ring_Atoms,Hide_N_Drop,FHO_Error,Tanimoto,Ring_Edge_Mask,Mask_DB_atoms
from .convert_mol_smile_tensor import *
from .display_funs import*

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout,TimeDistributed,Lambda, GRU, Activation, RepeatVector
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import sys
import PIL
from io import BytesIO


#rdkit 
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem as Chem


# =============================
# Encoder functions
# =============================

def gen_FHO_encoder(params,name):
    # generate all higher order fragments, F2, F3 .... FN , FR  
    atoms = Input(name='atoms', shape=(params['max_dangle_atoms'], params['num_atom_features'] ))
    Dangling_atoms = Input(name='Dangling_atoms', shape=(params['max_dangle_atoms'],))
    bonds  = Input(name='bonds', shape=(params['max_dangle_atoms'], params['max_degree'], params['num_bond_features']+2))
    edge = Input(name='edges', shape=(params['max_dangle_atoms'], params['max_degree']))
    MSA = Input(name='MSA', shape=(params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1))
    MSB = Input(name='MSB', shape=(params['Num_Graph_Convolutions'],params['max_dangle_atoms']* params['max_degree'],1))
    
    MSA_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='MSA_unstack'))(MSA)
    MSB_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='MSB_unstack'))(MSB)

    
    '''
    #################################################################################
    MOL ENCODER
    
            self.conv_width ,self.msg_width 
    '''
    
    sys.stdout.write("\r" + 'Encoder Model Generation: Adding Variational_layer '+params['excess_space'])
    sys.stdout.flush()
    var_layer = Variational( name = 'Variational_layer', init_seed = params['kl_loss_weight'])
    
    '''
    sys.stdout.write("\r" + 'Encoder Model Generation: Adding Find_Ring_Atoms '+params['excess_space'])
    sys.stdout.flush()
    FRA_outputs = Find_Ring_Atoms(params['max_dangle_atoms'],params['max_degree'],params['max_rings'])(edge)
    
    sys.stdout.write("\r" + 'Encoder Model Generation: Adding Find_Ring_Bonds '+params['excess_space'])
    sys.stdout.flush()
    Bonds_in_ring = Find_Ring_Bonds(params['max_dangle_atoms'],params['max_degree'],params['max_rings'])([edge,FRA_outputs[2]])
    
    sys.stdout.write("\r" + 'Encoder Model Generation: Hide_N_Drop'+params['excess_space'])
    sys.stdout.flush()  
    '''
    Hide_N_Drop_layer = Hide_N_Drop(params['FHO_hidden_bonds'],params['FHO_hidden_atoms'],params['FHO_encode_dropout'],name = 'Hide_N_Drop_A_N_B')
    add_layer = Lambda(lambda x: tf.math.add_n(x),name='add_layer')
    mult_layer = Lambda(lambda x: tf.math.multiply(x[0],x[1]),name='mult_layer')
    reduce_layer = Lambda(lambda x: tf.math.reduce_sum(x,axis=1),name='reduce_layer')
    R_Dangling_atoms = Lambda(lambda x: K.reshape(x, shape=(K.shape(x)[0], x.shape[1], 1)),name='reshape_dangling_atoms')(Dangling_atoms)
    DB_Mask_layer = Mask_DB_atoms(name='DB_Mask')
    masks_for_rings =   Ring_Edge_Mask(with_aromatic=not(params['KekulizeBonds']),name='Ring_edge_Mask')
    
    for HO_idx in range(0,params['Num_Graph_Convolutions']):
        sys.stdout.write("\r" + 'Encoder Model Generation: Adding NeuralGraphHidden F' + str(HO_idx+2)+params['excess_space'] )
        sys.stdout.flush()
        
        
        
        
        if(HO_idx==0):
            bonds_types,Dangling_bonds,bonds_in_rings = Lambda(lambda x:  tf.split(x, [params['num_bond_features'],1,1], 3),name='split_bonds')(bonds)
            
            Dangling_bonds_R = Lambda(lambda x: K.reshape(x, shape=(K.shape(x)[0], params['max_dangle_atoms']*params['max_degree'], 1)),name='reshape_dangling_bonds')([Dangling_bonds])
            
            [FHO_atoms,updated_bonds,FHO_bonds] = NeuralGraphHidden(params['FHO_convo_width'] ,params['FHO_convo_msg'],
                                    inner_hdd_dim = params['NGH_inner_hdd_dim'],  msg_hid_dim =  params['NGH_msg_hid_dim'],
                                    update_bonds=params['update_bonds'],bond_hid_dim = params['bond_hid_dim'],
                                    kernel_regularizer= regularizers.l2(params['FHO_encode_reg']),
                                    activation = params['FHO_encoder_activation'],
                                    z_bond_out_width=params['z_bond_out_width'],
                                    name = 'NGH_'+str(HO_idx))([atoms, bonds_types, edge])
            
            Atoms_to_sparse = DB_Mask_layer([FHO_atoms , Dangling_atoms])
            FHO_bonds_sparse = DB_Mask_layer([FHO_bonds , Dangling_bonds_R])
            
            #bond_type_ring = tf.keras.layers.concatenate([bonds_types,bonds_in_rings], axis=-1)
            [edge_in_rings,atoms_not_in_rings] = masks_for_rings([edge,bonds_in_rings,bonds_types,R_Dangling_atoms])
            
            F1_rings = NeuralGraphHidden(params['FHO_convo_width'] ,params['FHO_convo_msg'], kernel_regularizer= regularizers.l2(params['FHO_encode_reg']),activation = params['FHO_encoder_activation'],name = 'NGH_Ring'+str(HO_idx))([atoms,bonds_types, edge])
            
                        
            Ring_Atoms_to_sparse =  DB_Mask_layer([F1_rings , atoms_not_in_rings])
            
        else:
            FHO_atoms_with_DB =tf.keras.layers.concatenate([FHO_atoms,R_Dangling_atoms], axis=-1,name = 'Concatenate_atoms_'+str(HO_idx))
            if(params['use_update_bonds_in_NGC']):
                updated_bonds_with_ring = tf.keras.layers.concatenate([bonds,updated_bonds], axis=-1,name = 'Concatenate_bonds_'+str(HO_idx))
            else:
                updated_bonds_with_ring = bonds
            
            [FHO_atoms,updated_bonds,FHO_bonds] = NeuralGraphHidden(params['FHO_convo_width'] ,params['FHO_convo_msg'],
                                                inner_hdd_dim = params['NGH_inner_hdd_dim'],  msg_hid_dim =  params['NGH_msg_hid_dim'],
                                                update_bonds=params['update_bonds'],bond_hid_dim = params['bond_hid_dim'],
                                                kernel_regularizer= regularizers.l2(params['FHO_encode_reg']),
                                                activation = params['FHO_encoder_activation'],
                                                z_bond_out_width=params['z_bond_out_width'],
                                                name = 'NGH_'+str(HO_idx))([FHO_atoms_with_DB, updated_bonds_with_ring, edge])  
            Atoms_to_sparse = FHO_atoms
            FHO_bonds_sparse = FHO_bonds
            Ring_Atoms_to_sparse = NeuralGraphHidden(params['Ring_F1_convo_width'] ,params['Ring_F1_convo_msg'], kernel_regularizer= regularizers.l2(params['FHO_encode_reg']),activation = params['FHO_encoder_activation'],name = 'NGH_Ring'+str(HO_idx))([Ring_Atoms_to_sparse, bonds, edge_in_rings])
            Ring_Atoms_to_sparse =  DB_Mask_layer([Ring_Atoms_to_sparse , atoms_not_in_rings])


        #[FHO_bonds_sparse,FHO_atoms_sparse] = Hide_N_Drop_layer([FHO_bonds,FHO_atoms])
        
        FHO_bonds_sparse = FHO_bonds_sparse
        for i in range(0, params['FHO_num_sparse_layers']):
            sys.stdout.write("\r" + 'Encoder Model Generation: Adding F2 ' + str(HO_idx)+'  Sparsify'+params['excess_space'] )
            sys.stdout.flush()
            if(i == params['FHO_num_sparse_layers']-1):
                Z_log_var = NeuralGraphSparsify(params['FHO_finger_print'],activation = params['FHO_encoder_activation'], use_bias=False , 
                                                name = 'Bond_NGS_var'+str(HO_idx)+'_'+str(i))(FHO_bonds_sparse)
                FHO_bonds_z_error = var_layer([Z_log_var,FHO_bonds_sparse])
                
            FHO_bonds_sparse = NeuralGraphSparsify(params['FHO_finger_print'],activation = params['FHO_encoder_activation'], use_bias=False,
                                                   name =  'Bond_NGS_'+str(HO_idx)+'_'+str(i))(FHO_bonds_sparse)
            
            
        
        FHO_atoms_sparse = Atoms_to_sparse
        for i in range(0, params['FHO_num_sparse_layers']):
            
            sys.stdout.write("\r" + 'Encoder Model Generation: Adding NeuralGraphHidden ' + str(i)+' Sparsify'+params['excess_space'])
            sys.stdout.flush()
            if(i == params['FHO_num_sparse_layers']-1):
                Z_log_var = NeuralGraphSparsify(params['FHO_finger_print'],activation = params['FHO_encoder_activation'], use_bias=False ,
                                                name = 'Atom_NGS_var'+str(HO_idx)+'_'+str(i))(FHO_atoms_sparse)
                FHO_atoms_z_error = var_layer([Z_log_var,FHO_atoms_sparse])
                
            FHO_atoms_sparse = NeuralGraphSparsify(params['FHO_finger_print'],activation = params['FHO_encoder_activation'], use_bias=False,
                                                   name =  'Atom_NGS_'+str(HO_idx)+'_'+str(i))(FHO_atoms_sparse)
            
            
            
            
                
        Rings_sparse = Ring_Atoms_to_sparse
        for i in range(0, params['FHO_num_sparse_layers']):
            
            sys.stdout.write("\r" + 'Encoder Model Generation: Adding RingGraphHidden ' + str(i)+' Sparsify'+params['excess_space'])
            sys.stdout.flush()
            if(i == params['FHO_num_sparse_layers']-1):
                Z_log_var = NeuralGraphSparsify(params['FHO_finger_print'],activation = params['FHO_encoder_activation'], use_bias=params['Ring_bias_sparse'] ,
                                                name = 'Ring_NGS_var'+str(HO_idx)+'_'+str(i))(Rings_sparse)
                Rings_z_error = var_layer([Z_log_var,Rings_sparse])
                
            Rings_sparse = NeuralGraphSparsify(params['FHO_finger_print'],activation = params['FHO_encoder_activation'], use_bias=params['Ring_bias_sparse'],
                                                   name =  'Ring_NGS_'+str(HO_idx)+'_'+str(i))(Rings_sparse)
            
      

            
            
        if(params['FHO_interpretable']):
            if(HO_idx == 0):    
                z1 = add_layer([FHO_atoms_sparse,Rings_sparse])
                z2 = FHO_bonds_sparse
                z =  add_layer([ FHO_atoms_sparse, FHO_bonds_sparse, Rings_sparse])
                z_error =  add_layer([ FHO_atoms_z_error, FHO_bonds_z_error, Rings_z_error])
                
            else:
                z =  add_layer([z, FHO_atoms_sparse, Rings_sparse])
                z2 = add_layer([z2,FHO_bonds_sparse])
                z_error =  add_layer([ z_error, FHO_atoms_z_error, FHO_bonds_z_error, Rings_z_error])
                if(HO_idx == 1):
                    zR =  Rings_sparse
                else:
                    zR =  add_layer([ zR, Rings_sparse])            
        else:
            if(HO_idx == 0):    
                z1 = add_layer([reduce_layer(FHO_atoms_sparse),reduce_layer(Rings_sparse)])
                z2 = reduce_layer(FHO_bonds_sparse)
                z =  add_layer([ reduce_layer(FHO_atoms_sparse), reduce_layer(FHO_bonds_sparse), reduce_layer(Rings_sparse)])
                z_error =  add_layer([ FHO_atoms_z_error, FHO_bonds_z_error, Rings_z_error])
                
            else:
                z =  add_layer([z, reduce_layer(FHO_atoms_sparse), reduce_layer(FHO_bonds_sparse),reduce_layer(Rings_sparse)])
                z_error =  add_layer([ z_error, FHO_atoms_z_error, FHO_bonds_z_error, Rings_z_error])
                if(HO_idx == 1):
                    zR =  reduce_layer(Rings_sparse)
                    zS = add_layer([ reduce_layer(mult_layer([MSA_unstack[HO_idx],FHO_atoms_sparse])),reduce_layer(mult_layer([MSB_unstack[HO_idx],FHO_bonds_sparse]))])

                else:
                    zR =  add_layer([zR, reduce_layer(Rings_sparse)])
                    zS =  add_layer([zS,  reduce_layer(mult_layer([MSA_unstack[HO_idx],FHO_atoms_sparse])),reduce_layer(mult_layer([MSB_unstack[HO_idx],FHO_bonds_sparse]))])

    
    sys.stdout.write("\r" + 'Encoder Model Generation: Complete :D' +params['excess_space'])
    sys.stdout.flush()
    
    outputs =[z,z_error,z1,z2,zR,zS]
    return models.Model(inputs =[atoms,Dangling_atoms, bonds, edge,MSA,MSB],outputs=outputs,name = name)


def gen_FHO_decoder(params,name):
    
    
    G_z = Input(name='G_z', shape=(params['FHO_finger_print'], ))
    G_z1 = Input(name='G_z1', shape=(params['FHO_finger_print'], ))
    
    z_valid_start = Input(name='z_valid_start', shape=(params['FHO_finger_print'], ))
    z1_valid_start = Input(name='z1_valid_start', shape=(params['FHO_finger_print'], ))
    z2_valid_start = Input(name='z2_valid_start', shape=(params['FHO_finger_print'], ))
    zR_valid_start = Input(name='zR_valid_start', shape=(params['FHO_finger_print'], ))
    zS_valid_start = Input(name='zS_valid_start', shape=(params['FHO_finger_print'], ))
    
    z_end = Input(name='z_end', shape=(params['FHO_finger_print'], ))
    z1_end = Input(name='z1_end', shape=(params['FHO_finger_print'], ))
    z2_end = Input(name='z2_end', shape=(params['FHO_finger_print'], ))
    zR_end = Input(name='zR_end', shape=(params['FHO_finger_print'], ))
    zS_end = Input(name='zS_end', shape=(params['FHO_finger_print'], ))

    inputs = [G_z,G_z1,z_valid_start,z1_valid_start,z2_valid_start,zR_valid_start,zS_valid_start,z_end,z1_end,z2_end,zR_end,zS_end]
    
    subtract_layer = tf.keras.layers.subtract
    
    Z_G_remain = subtract_layer([subtract_layer([subtract_layer([G_z,z2_valid_start]),G_z1]),zS_valid_start])
    Z_end_partial = subtract_layer([subtract_layer([subtract_layer([z_end,z2_end]),z1_end]),zS_end])
    Z_cur_partial = subtract_layer([subtract_layer([subtract_layer([z_valid_start,z2_valid_start]),z1_valid_start]),zS_valid_start])
    Z1_add = subtract_layer([z1_end,z1_valid_start])
    Z2_add = subtract_layer([z2_end ,z2_valid_start])
    ZS_add = subtract_layer([zS_end ,zS_valid_start])
    Z1_remain = subtract_layer([G_z1 , z1_valid_start])


    
    

    
    Tanimoto_layer = Tanimoto(params['FHO_encoder_activation'])
    tanimoto_values = Tanimoto_layer([G_z,z_end,z_valid_start])
        
    
    

    input_dense =  tf.keras.layers.concatenate([Z_G_remain,Z_end_partial,Z_cur_partial,Z1_add,Z2_add,ZS_add,zR_end,z1_valid_start,z2_valid_start,zR_valid_start,zS_valid_start,Z1_remain,tanimoto_values], axis=-1)
    star_dim = params['FHO_Input_NN_Shape']
    end_dim = 2
    increase_dim = int(( end_dim-star_dim)/params['FHO_depth'])
    
    sys.stdout.write("\r" + 'FHO Decoder Model Generation: '+params['excess_space'])
    sys.stdout.flush()
    
    FHO_decoder = Dense(star_dim, name = 'FHO_decoder_-1',activation = params['FHO_decoder_activation'], kernel_regularizer= regularizers.l2(params['FHO_decode_reg']))(input_dense)
    FHO_decoder = Dropout(params['FHO_decode_dropout'],name = 'FHO_decoder_dropout_'+str(0))(FHO_decoder)
    sys.stdout.write("\r" + 'FHO Decoder Model Generation: Layer 0'+params['excess_space'])
    sys.stdout.flush()
    FHO_decoder = Dense(star_dim+increase_dim, name = 'FHO_decoder_0',activation = params['FHO_decoder_activation'], kernel_regularizer= regularizers.l2(params['FHO_decode_reg']))(FHO_decoder)
    for i in range(1,params['FHO_depth']-2):
        sys.stdout.write("\r" + 'FHO Decoder Model Generation: Layer '+str(i)+params['excess_space'])
        sys.stdout.flush()
        FHO_decoder = Dropout(params['FHO_decode_dropout'],name = 'F1_N_decoder_dropout_'+str(i))(FHO_decoder)
        FHO_decoder = Dense(star_dim+increase_dim*i,name = 'FHO_decoder_'+str(i),activation = params['FHO_decoder_activation'], kernel_regularizer= regularizers.l2(params['FHO_decode_reg']))(FHO_decoder)
    FHO_decoder = Dense(star_dim+increase_dim*(i+1),name = 'FHO_decoder_'+str(i+1),activation = 'tanh')(FHO_decoder)
    FHO_decoder = Dense(end_dim, name = 'valid_mol',activation = 'softmax', kernel_regularizer= regularizers.l2(params['FHO_decode_reg']))(FHO_decoder)
    sys.stdout.write("\r" + 'FHO Decoder Model Generation: Complete '+params['excess_space'])
    sys.stdout.flush()
    return models.Model(inputs = inputs ,outputs= [FHO_decoder],name =name)
    


def gen_train_FHO_VAE(params,name,FHO_encoder,FHO_decoder):
        
    
    sys.stdout.write("\r" + 'Training Model Generation: '+params['excess_space'])
    sys.stdout.flush()

    
    G_atoms = Input(name='G_atoms', shape=(params['max_dangle_atoms'], params['num_atom_features'] ))
    G_bonds  = Input(name='G_bonds', shape=(params['max_dangle_atoms'], params['max_degree'], params['num_bond_features']+2))
    G_edges = Input(name='G_edges', shape=(params['max_dangle_atoms'], params['max_degree']))
    G_DG_atoms = Input(name='G_DG_atoms', shape=(params['max_dangle_atoms'],))
    G_MSA = Input(name='G_MSA', shape=(params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1))
    G_MSB = Input(name='G_MSB', shape=(params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1))
    
    
    G_valid_atoms = Input(name='G_valid_atoms', shape=(params['FHO_attempts_per_batch']+1, params['max_dangle_atoms'], params['num_atom_features'] ))
    G_valid_bonds  = Input(name='G_valid_bonds', shape=(params['FHO_attempts_per_batch']+1 ,params['max_dangle_atoms'], params['max_degree'], params['num_bond_features']+2))
    G_valid_edges = Input(name='G_valid_edges', shape=(params['FHO_attempts_per_batch']+1, params['max_dangle_atoms'], params['max_degree']))
    G_valid_DG_atoms = Input(name='G_valid_DG_atoms', shape=(params['FHO_attempts_per_batch']+1, params['max_dangle_atoms']))
    G_valid_MSA = Input(name='G_valid_MSA', shape=(params['FHO_attempts_per_batch']+1,params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1))
    G_valid_MSB = Input(name='G_valid_MSB', shape=(params['FHO_attempts_per_batch']+1,params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1))
    
    
    G_valid_atoms_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_valid_atoms_unstack'))(G_valid_atoms)
    G_valid_bonds_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_valid_bonds_unstack'))(G_valid_bonds)
    G_valid_edges_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_valid_edges_unstack'))(G_valid_edges)
    G_valid_DG_atoms_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_valid_DG_atoms_unstack'))(G_valid_DG_atoms)     
    G_valid_MSA_unstack  = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='batch_Gvalids_MSA_unstack'))(G_valid_MSA)     
    G_valid_MSB_unstack  = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='batch_Gvalids_MSB_unstack'))(G_valid_MSB)     
    
    G_INV1_atoms = Input(name='G_INV1_atoms', shape=(params['FHO_attempts_per_batch'], params['max_dangle_atoms'], params['num_atom_features'] ))
    G_INV1_bonds  = Input(name='G_INV1_bonds', shape=(params['FHO_attempts_per_batch'] ,params['max_dangle_atoms'], params['max_degree'], params['num_bond_features']+2))
    G_INV1_edges = Input(name='G_INV1_edges', shape=(params['FHO_attempts_per_batch'], params['max_dangle_atoms'], params['max_degree']))
    G_INV1_DG_atoms = Input(name='G_INV1_DG_atoms', shape=(params['FHO_attempts_per_batch'], params['max_dangle_atoms']))
    G_INV1_MSA = Input(name='G_INV1_MSA', shape=(params['FHO_attempts_per_batch'],params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1))
    G_INV1_MSB = Input(name='G_INV1_MSB', shape=(params['FHO_attempts_per_batch'],params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1))
    
    
    G_INV1_atoms_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV1_atoms_unstack'))(G_INV1_atoms)
    G_INV1_bonds_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV1_bonds_unstack'))(G_INV1_bonds)
    G_INV1_edges_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV1_edges_unstack'))(G_INV1_edges)
    G_INV1_DG_atoms_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV1_DG_atoms_unstack'))(G_INV1_DG_atoms) 
    G_INV1_MSA_unstack  = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV1_MSA_unstack'))(G_INV1_MSA)     
    G_INV1_MSB_unstack  = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV1_MSB_unstack'))(G_INV1_MSB) 

    
    
    G_INV2_atoms = Input(name='G_INV2_atoms', shape=(params['FHO_attempts_per_batch'], params['max_dangle_atoms'], params['num_atom_features'] ))
    G_INV2_bonds  = Input(name='G_INV2_bonds', shape=(params['FHO_attempts_per_batch'] ,params['max_dangle_atoms'], params['max_degree'], params['num_bond_features']+2))
    G_INV2_edges = Input(name='G_INV2_edges', shape=(params['FHO_attempts_per_batch'], params['max_dangle_atoms'], params['max_degree']))
    G_INV2_DG_atoms = Input(name='G_INV2_DG_atoms', shape=(params['FHO_attempts_per_batch'], params['max_dangle_atoms']))
    G_INV2_MSA = Input(name='G_INV2_MSA', shape=(params['FHO_attempts_per_batch'],params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1))
    G_INV2_MSB = Input(name='G_INV2_MSB', shape=(params['FHO_attempts_per_batch'],params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1))
 
    G_INV2_atoms_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV2_atoms_unstack'))(G_INV2_atoms)
    G_INV2_bonds_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV2_bonds_unstack'))(G_INV2_bonds)
    G_INV2_edges_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV2_edges_unstack'))(G_INV2_edges)
    G_INV2_DG_atoms_unstack = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV2_DG_atoms_unstack'))(G_INV2_DG_atoms)
    G_INV2_MSA_unstack  = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV2_MSA_unstack'))(G_INV2_MSA)     
    G_INV2_MSB_unstack  = tf.keras.layers.Lambda( lambda x:tf.unstack( x, axis=1,  name='G_INV2_MSB_unstack'))(G_INV2_MSB)
    
    subtract_layer = tf.keras.layers.subtract
    add_layer = Lambda(lambda x: tf.math.add_n(x),name='add_layer_train')
 
    inputs = [G_atoms,G_bonds,G_edges,G_DG_atoms,G_MSA,G_MSB,
              G_valid_atoms,G_valid_bonds,G_valid_edges,G_valid_DG_atoms,G_valid_MSA,G_valid_MSB,
              G_INV1_atoms,G_INV1_bonds,G_INV1_edges,G_INV1_DG_atoms,G_INV1_MSA,G_INV1_MSB,
              G_INV2_atoms,G_INV2_bonds,G_INV2_edges,G_INV2_DG_atoms,G_INV2_MSA,G_INV2_MSB]
    
    #inputs = [atoms,bonds,edges,edmol_DG_atoms]

    sys.stdout.write("\r" + 'Training Model Generation: Calling FHO_encoder G'+params['excess_space'])
    sys.stdout.flush()

    
    G_z,G_z_error,G_z1,G_z2,G_zR, G_zS = FHO_encoder([G_atoms,G_DG_atoms,G_bonds,G_edges,G_MSA,G_MSB])
    
    sys.stdout.write("\r" + 'Training Model Generation: Calling FHO_encoder G_0'+params['excess_space'])
    sys.stdout.flush()
    z_valid_start,z_error_valid_start,z1_valid_start,z2_valid_start,zR_valid_start,zS_valid_start =FHO_encoder([G_valid_atoms_unstack[0],G_valid_DG_atoms_unstack[0],G_valid_bonds_unstack[0],G_valid_edges_unstack[0],G_valid_MSA_unstack[0],G_valid_MSB_unstack[0]])
    
    
    total_error = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x[:,0:1,0]),name = 'start_error')(G_atoms)
    valid_Error_layer = FHO_Error(False,name='valid_error_layer')
    invalid_Error_layer = FHO_Error(True,name='invalid_error_layer')
    
    
    
    for index in range(0,params['FHO_attempts_per_batch']):
        sys.stdout.write("\r" + 'Training Model Generation: Calling FHO_encoder G_'+str(index)+'_valid'+params['excess_space'])
        sys.stdout.flush()
        z_valid_end,z_error_valid_end,z1_valid_end,z2_valid_end,zR_valid_end, zS_valid_end = FHO_encoder([G_valid_atoms_unstack[index+1],G_valid_DG_atoms_unstack[index+1],G_valid_bonds_unstack[index+1],G_valid_edges_unstack[index+1],G_valid_MSA_unstack[index+1],G_valid_MSB_unstack[index+1]])
        
        
        inputs_Decode_valid = [G_z,G_z1,z_valid_start,z1_valid_start,z2_valid_start,zR_valid_start,zS_valid_start,z_valid_end,z1_valid_end,z2_valid_end,zR_valid_end,zS_valid_end]
                
        perdict_mol = FHO_decoder(inputs_Decode_valid)
        total_error = valid_Error_layer([perdict_mol,total_error])
        
        sys.stdout.write("\r" + 'Training Model Generation: Calling FHO_encoder G_'+str(index)+'_in_valid1'+params['excess_space'])
        sys.stdout.flush()
        z_Invalid1,z_error_Invalid1,z1_Invalid1,z2_Invalid1,zR_Invalid1,zS_Invalid1 =FHO_encoder([G_INV1_atoms_unstack[index],G_INV1_DG_atoms_unstack[index],G_INV1_bonds_unstack[index],G_INV1_edges_unstack[index],G_INV1_MSA_unstack[index],G_INV1_MSB_unstack[index]])

        inputs_Decode_Invalid1 = [G_z,G_z1,z_valid_start,z1_valid_start,z2_valid_start,zR_valid_start,zS_valid_start,z_Invalid1,z1_Invalid1,z2_Invalid1,zR_Invalid1,zS_Invalid1]
        
        perdict_mol = FHO_decoder(inputs_Decode_Invalid1)
        total_error = invalid_Error_layer([perdict_mol,total_error])
        
        sys.stdout.write("\r" + 'Training Model Generation: Calling FHO_encoder G_'+str(index)+'_in_valid2'+params['excess_space'])
        sys.stdout.flush()
        
        z_Invalid2,z_error_Invalid2,z1_Invalid2,z2_Invalid2,zR_Invalid2,zS_Invalid2 =FHO_encoder([G_INV2_atoms_unstack[index],G_INV2_DG_atoms_unstack[index],G_INV2_bonds_unstack[index],G_INV2_edges_unstack[index],G_INV2_MSA_unstack[index],G_INV2_MSB_unstack[index]])
        
        inputs_Decode_Invalid2 = [G_z,G_z1,z_valid_start,z1_valid_start,z2_valid_start,zR_valid_start,zS_valid_start,z_Invalid2,z1_Invalid2,z2_Invalid2,zR_Invalid2,zS_Invalid2]

        
        perdict_mol = FHO_decoder(inputs_Decode_Invalid2)
        total_error = invalid_Error_layer([perdict_mol,total_error])
        
        
        
        z_valid_start =z_valid_end
        z_error_valid_start = z_error_valid_end
        z1_valid_start = z1_valid_end
        z2_valid_start = z2_valid_end
        zR_valid_start = zR_valid_end 
        zS_valid_start = zS_valid_end
    
    total_error_output =  tf.keras.layers.Lambda(lambda x: x,name = 'total_error_output')(total_error)
    sys.stdout.write("\r" + 'Training Model Generation Complete: '+name+params['excess_space'])
    sys.stdout.flush()
    
    #valid_mol=  tf.keras.layers.Lambda(lambda x: x,name = 'valid_mol')(z)
    return models.Model(inputs = inputs,outputs= [total_error_output],name =name)







