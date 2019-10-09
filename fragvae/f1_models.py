# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:26:50 2019

@author: ja550
"""

from .display_funs import*

from .layers import Variational, NeuralGraphHidden,NeuralGraphSparsify, next_node_atom_error,node_atom_error
from .convert_mol_smile_tensor import *

#from utils import display_F1_graphs
#from utils import *
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout,TimeDistributed,Lambda, GRU, Activation, RepeatVector
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import PIL
from io import BytesIO

#rdkit 
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem as Chem


# =============================
# Encoder functions
# =============================

def gen_F1_encoder(params,name):

    atoms = Input(name='atoms', shape=(params['max_atoms'], params['num_atom_features'] ))
    bonds  = Input(name='bonds', shape=(params['max_atoms'], params['max_degree'], params['num_bond_features']))
    edge = Input(name='edges', shape=(params['max_atoms'], params['max_degree']))

    '''
    #################################################################################
    MOL ENCODER
    
            self.conv_width ,self.msg_width 
    
    '''
    var_layer = Variational( name = 'Variational_layer', init_seed = params['kl_loss_weight'])
    F1_set = NeuralGraphHidden(params['F1_convo_width'] ,params['F1_convo_msg'],msg_hid_dim=params['F1_msg_hid_dim'],inner_hdd_dim=params['F1_inner_hdd_dim'], kernel_regularizer= regularizers.l2(params['NGF_reg']),activation = params['F1_encoder_activation'])([atoms, bonds, edge])
    for i in range(0, params['NGF_F1_Sparse_layers']):
        if(i == params['NGF_F1_Sparse_layers']-1):            
            F1_Z_log_var = NeuralGraphSparsify(params['finger_print'],activation = params['F1_encoder_activation'], use_bias=False , name = 'F1_error', kernel_regularizer= regularizers.l2(params['NGF_reg']))(F1_set)
            total_z_error = var_layer([F1_Z_log_var,F1_set])
        if(i>0): 
            F1_set = TimeDistributed(Dropout(params['F1_encode_dropout'],name = 'F1_encoder_dropout_'+str(i)),name = 'TD_drop_out_encoder_'+str(i))(F1_set)
        F1_set = NeuralGraphSparsify(params['finger_print'],activation = params['F1_encoder_activation'], use_bias=False , name = 'F1_sparse_'+str(i)+'_layer', kernel_regularizer= regularizers.l2(params['NGF_reg']))(F1_set)
        


       
    return models.Model(inputs =[atoms, bonds, edge],outputs= [F1_set,total_z_error],name = name)


##====================
## Middle part (var)
##====================




##====================
## Decoder functions
##====================

def gen_F1_N_decoder(params,name):
    """
    Caluclate probality  of Node being a sepcific node
    0th postion is a null node.
    """
    finger_print = Input(name='Finger_print', shape=(params['finger_print'], ))
    Selected_hist = Input(name='Selected_hist', shape=(params['finger_print'], ))
    inputs = [finger_print,Selected_hist]
    input_dense =  tf.keras.layers.concatenate(inputs, axis=-1)
    star_dim = finger_print.shape[1].value*2
    end_dim = 1+params['num_atom_features']
    increase_dim = int(( end_dim- star_dim)/params['NN_decoder_len'])

    
    n_decoder = Dense(star_dim, name = 'n_decoder_-1',activation = params['F1_decoder_activation'], kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(input_dense)
    n_decoder = Dropout(params['F1_decode_dropout'],name = 'F1_N_decoder_dropout_'+str(0))(n_decoder)
    n_decoder = Dense(star_dim+increase_dim, name = 'n_decoder_0',activation = params['F1_decoder_activation'], kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(n_decoder)
    for i in range(1,params['Node_decoder_len']-2):
        n_decoder = Dropout(params['F1_decode_dropout'],name = 'F1_N_decoder_dropout_'+str(i))(n_decoder)
        n_decoder = Dense(star_dim+increase_dim*i,name = 'n_decoder_'+str(i),activation = params['F1_decoder_activation'], kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(n_decoder)
    n_decoder = Dense(star_dim+increase_dim*(i+1),name = 'n_decoder_'+str(i+1),activation = 'tanh')(n_decoder)
    n_decoder = Dense(end_dim, name = 'n_decoder_output',activation = 'softmax', kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(n_decoder)
    
    return models.Model(inputs = inputs,outputs= [n_decoder],name =name)
    """
    Caluclate probality  of Node being a sepcific node
    0th postion is a null node.
    """
    

def gen_F1_NN_decoder(params,name):

    nn_decoder= []
    finger_print = Input(name='Finger_print', shape=(params['finger_print'], ))
    Selected_F1_hist = Input(name='Selected_hist', shape=(params['finger_print'], ))
    node = Input(name='Node', shape=(1+params['num_atom_features'], ))
    Selected_NN_hist = Input(name='NN_atom_bond_features', shape=(1+params['num_atom_features']*params['num_bond_features'], ))
    
    F1_selected_hist_cast = Lambda(lambda x: tf.cast(x,dtype='float32'),name = 'select_history')(Selected_NN_hist)

    
    list_of_inputs = [finger_print,node,Selected_F1_hist]
    for degree in range(0,params['max_degree'] ):
        star_dim_0 = finger_print.shape[1].value+1+(params['num_atom_features'])*params['num_bond_features']*params['decoder_0_expand']+(1+params['num_atom_features'])
        star_dim = finger_print.shape[1].value+1+(params['num_atom_features'])*params['num_bond_features']*params['decoder_1_expand']+(1+params['num_atom_features'])

        end_dim = 1+(params['num_atom_features'])*params['num_bond_features']
        increase_dim = int(( end_dim- star_dim)/params['NN_decoder_len'])

        
        inputs =  tf.keras.layers.concatenate(list_of_inputs+[F1_selected_hist_cast], axis=-1)
        
        nn_layer_decoder = Dense(star_dim_0+increase_dim, name = 'n_degree'+str(degree)+'_layer_-1',activation = params['F1_decoder_activation'], kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(inputs)
        nn_layer_decoder = Dropout(params['F1_decode_dropout'],name = 'F1_NN_decoder_dropout_'+str(0)+'_degree_'+str(degree))(nn_layer_decoder)
        nn_layer_decoder = Dense(star_dim+increase_dim, name = 'n_degree'+str(degree)+'_layer_0',activation = params['F1_decoder_activation'], kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(inputs)
        for layer_index in range(1,params['NN_decoder_len']-2):
            nn_layer_decoder = Dropout(params['F1_decode_dropout'],name = 'F1_NN_decoder_dropout_'+str(layer_index)+'_degree_'+str(degree))(nn_layer_decoder)
            nn_layer_decoder = Dense(star_dim+increase_dim*layer_index,name = 'n_degree'+str(degree)+'_layer_'+str(layer_index),activation =params['F1_decoder_activation'], kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(nn_layer_decoder)
        nn_layer_decoder = Dense(star_dim+increase_dim*(layer_index+1),name = 'n_degree'+str(degree)+'_layer_'+str(layer_index+1),activation = 'tanh')(nn_layer_decoder)

        nn_layer_decoder = Dense(end_dim, name = 'n_degree_'+str(degree)+'_output',activation = 'softmax', kernel_regularizer= regularizers.l2(params['F1_decode_reg']))(nn_layer_decoder)
            
        nn_decoder.append(models.Model(inputs = list_of_inputs+[Selected_NN_hist],outputs= [nn_layer_decoder],name = name+'_'+str(degree)))

    return nn_decoder


##====================
## Custon Training / Error propragation functions
##====================
def select_node_fun(params, inputs):
    AI_prob,True_prob = inputs
    
    '''  Max prob cal
    '''
    
    if(True):
        #automatically selects higher prob. Currently right now it is always set
        # to select the true prob. Note. True prob is not true prob in testing
        max_val = tf.math.multiply(True_prob,True_prob)
        arg_max = tf.dtypes.cast(tf.math.argmax(True_prob,axis=-1),dtype ='int32' )
    else:
        arg_max = tf.random.categorical(tf.log(True_prob), 1)[:,0]
    
    one_hot = tf.one_hot(arg_max,AI_prob.shape[1])
    
    return one_hot


def gen_node_error(params,name='Node_atom_error'):
    atoms = Input(name='atoms', shape=(params['max_atoms'], params['num_atom_features'] ),dtype ='int32')
    proposed_nodes  = Input(name='proposed_nodes_', shape=(1+params['num_atom_features'],))
    total_error = Input(name='total_error_input_', shape=(1, ))
    #remaining_nodes =  Input(name='remain_N', shape=(params['max_atoms'],))

    out_error,prob = node_atom_error(name =name+'_layer')([proposed_nodes,atoms, total_error])


    
    
    return models.Model(inputs =[proposed_nodes,atoms, total_error],outputs= [out_error,prob],name = name+'_model')

def gen_next_node_error(params,name='NN_error_'):
    total_error = Input(name='total_error_input', shape=(1, ))
    N_match= Input(name='N_match', shape=(params['max_atoms'],  ),dtype ='int32')
    Finger_print = Input(name='finger_print_', shape=(params['finger_print'],))
    NN_Tensor = Input(name='NN_Tensor', shape=(params['max_atoms'], 1+params['num_atom_features']*params['num_bond_features']),dtype ='int32')
    select_node =  Input(name='select_node', shape=(1+params['num_atom_features'],),dtype ='int32')
    selected_NN_hist =  Input(name='select_NN_hist', shape=(1+params['num_atom_features']*params['num_bond_features'],),dtype ='int32')
    NN_models = []



    


    for degree in range(0,params['max_degree'] ):
     
        proposed_NN = Input(name='proposed_NN'+'_degree_'+str(degree), shape=(1+params['num_atom_features']*params['num_bond_features'],))
        inputs = [total_error,proposed_NN,NN_Tensor,select_node, selected_NN_hist,N_match]
        total_error_out,prob,NN_match= next_node_atom_error(degree,name = 'NN_error_'+'_'+str(degree))(inputs)
        next_name =  name+'_degree_'+str(degree)
        NN_models.append(models.Model(inputs =inputs, outputs= [total_error_out,prob,NN_match], name =next_name))
    
    return NN_models

def gen_traning_node(atom_idx, atoms):
    non_zero_atoms = atoms[:,atom_idx]
    summed_atoms = K.sum(non_zero_atoms,axis = 1)
    node_zero = K.equal(summed_atoms,K.ones_like(summed_atoms)*0)
    node_zero = tf.dtypes.cast(node_zero,dtype = non_zero_atoms.dtype )
    return K.concatenate([K.reshape(node_zero,(K.shape(node_zero)[0],1)),non_zero_atoms],axis=-1)

def gen_select_node( params,name = 'select_node'):
    AI_prob = Input(name='AI_P_N_given_F', shape=(params['num_atom_features'] +1,))
    True_prob = Input(name='P_N_given_F', shape=( params['num_atom_features']+1, ))
    
    select_node = tf.keras.layers.Lambda(lambda x: select_node_fun(params,x))([AI_prob,True_prob])


    return models.Model(inputs =[AI_prob,True_prob],outputs= [select_node],name = name)

def gen_select_next_node(params,name = 'select_next_node'):
    AI_prob = Input(name='AI_P_N_given_F', shape=(params['num_atom_features']*params['num_bond_features'] +1,))
    True_prob = Input(name='P_N_given_F', shape=( params['num_atom_features']*params['num_bond_features']+1, ))
    select_node = tf.keras.layers.Lambda(lambda x: select_node_fun(params,x))([AI_prob,True_prob])


    return models.Model(inputs =[AI_prob,True_prob],outputs= [select_node],name = name)


def remain_fp(inputs):
    [F1_set,total_z_error]= inputs 
    

    remaining_fp = tf.math.add(K.sum(F1_set,axis = -2),total_z_error)
    
    return remaining_fp



def remove_atom_idx(inputs,atom_idx):
    selected_NN_hist,NN_Tensor,atoms,F1_set,N_match,F1_selected_hist = inputs
    
    
    Train_NN = tf.cast(selected_NN_hist,dtype ='int32')
    
    Train_NN = K.reshape(Train_NN,(K.shape(Train_NN)[0],1,Train_NN.shape[1]))

    #NN_Tensor = K.sum(NN_Tensor,axis =2)
    NN_Tensor_ref = tf.math.subtract(NN_Tensor,Train_NN)

    NN_match = K.equal(NN_Tensor_ref, tf.zeros_like(NN_Tensor_ref))
    NN_match = tf.cast(tf.math.reduce_all(NN_match,axis =2),dtype='int32')    
    
    
    match  = tf.argmax(tf.multiply(N_match,NN_match),axis = -1)
    remaining_nodes = tf.one_hot(match,atoms.shape[1],on_value =0,off_value =1 )
    remaining_nodes= K.reshape(remaining_nodes,(K.shape(remaining_nodes)[0],remaining_nodes.shape[1],1))
    select_nodes =   K.reshape(tf.one_hot(match,atoms.shape[1],on_value =1,off_value =0 ),(K.shape(remaining_nodes)[0],remaining_nodes.shape[1],1))

    NN_Tensor_out = tf.multiply(NN_Tensor,remaining_nodes,name = 'NN_T_'+str(atom_idx))
    atoms_out = tf.multiply(atoms,remaining_nodes,name = 'Atoms_'+str(atom_idx))
    F1_set_out = tf.multiply(F1_set,tf.cast(remaining_nodes,dtype='float32'),name = 'F1_'+str(atom_idx))
    F1_selected_out = K.sum(tf.multiply(F1_set,tf.cast(select_nodes,dtype='float32'),name = 'F1_'+str(atom_idx)),axis=1)
    
    F1_selected_out =tf.add(F1_selected_hist,F1_selected_out)
    return NN_Tensor_out,atoms_out,F1_set_out, match, F1_selected_out


def check_atoms_fun(inputs):
    atoms,select_node = inputs
    select_node = tf.cast(select_node,dtype ='int32')
    select_node = K.reshape(select_node,(K.shape(select_node)[0],1,select_node.shape[1]))
    check_atoms = tf.cast(tf.reduce_all(tf.equal(tf.subtract(atoms, select_node[:,:,1:select_node.shape[2]]), tf.zeros_like(atoms)),axis=2),dtype = 'int32')
    
    
    
    return check_atoms


def atoms_zero_for_pre_train(atoms):    
    
    zeros = tf.cast(tf.equal(K.sum(atoms[:,0,:],axis=-1),tf.zeros_like(K.sum(atoms[:,0,:],axis=-1))),dtype = 'float32')
    zeros = K.reshape(zeros,(K.shape(atoms)[0],1))
    
    select_node = tf.concat([zeros,tf.cast(atoms[:,0,:],dtype='float32')],axis=-1)
    return select_node


def Screen_NAN(error):
    AI_prob,True_prob = inputs
    
    '''  Max prob cal
    '''
    
    if(params['predict_model']):
        max_val = tf.math.multiply(True_prob,True_prob)
        arg_max = tf.dtypes.cast(tf.math.argmax(True_prob,axis=-1),dtype ='int32' )
    else:
        arg_max = tf.random.categorical(tf.log(True_prob), 1)[:,0]
    
    one_hot = tf.one_hot(arg_max,AI_prob.shape[1])
    
    return one_hot


##====================
## Build complete F1 predictive model
##====================

def gen_F1_training_model(params, F1_encoder = '', atom_decoder = '', NN_decoder =''):  
    
    
    NN_prob_list = []
    match_list  =[]
    N_prob_list =[]
    AI_N_list =[]
    AI_NN_list = []
    selected_NN_list = []
    selected_N_list = []
    AI_NN_selected  =[]
    AI_N_selected = []
    N_match_list = []
    
    atoms_in = Input(name='atoms', shape=(params['max_atoms'], params['num_atom_features'] ))
    edge = Input(name='edges', shape=(params['max_atoms'], params['max_degree']))
    bonds  = Input(name='bonds', shape=(params['max_atoms'], params['max_degree'], params['num_bond_features']))
    NN_Tensor_in = Input(name='NN_Tensor', shape=(params['max_atoms'], params['max_degree'], 1+params['num_atom_features']*params['num_bond_features']),dtype = 'int32')
    
    
    atoms= Lambda(lambda x:  tf.identity(x),name = 'atoms_out')(atoms_in)
    NN_Tensor_for_pre_train = Lambda(lambda x:  tf.identity(x),name = 'reduced_NN_Tensor_pre_train')(NN_Tensor_in) 
    NN_Tensor= Lambda(lambda x:   K.sum(x,axis =2),name = 'reduced_NN_Tensor_out')(NN_Tensor_in)
    total_error = Lambda(lambda x:  tf.zeros([K.shape(x)[0],1],dtype=x.dtype),name = 'total_error')(atoms)
    

    '''
    #################################################################################
    MOL ENCODER
    
                self.conv_width ,self.msg_width 
    
    '''
    
    if(F1_encoder==''):
        graph_encoder = gen_F1_encoder(params,name ='F1_Encoder')
    else:
        graph_encoder = F1_encoder
    
    if(atom_decoder==''):
        atom_decoder = gen_F1_N_decoder(params,name ='Atom_decoder')
    else:
        atom_decoder = atom_decoder
        
    if(NN_decoder==''):
        NN_decoder = gen_F1_NN_decoder(params,name ='Decoder_degree')
    else:
        NN_decoder = NN_decoder



    
    [F1_set,total_z_error] =graph_encoder([atoms, bonds, edge])
    
    # Cast to make comparable to NN_Tensor
    atoms = Lambda(lambda x: tf.cast(x, 'int32'),name='cast_atoms')(atoms)
    
    
    

    node_selector = gen_select_node( params,name = 'select_node')

    NN_selector = gen_select_next_node(params,name = 'select_NN')

    
    
    #Layer which remove a node from the remaining subset
    
    
    #Calculates remaining fingerprint based on remaining nodes
    calc_remain_fp = Lambda(lambda x:remain_fp(x),name='Cal_remain_fp')
    
    #Resets NN history to 0
    zero_selected_NN_hist =Lambda(lambda x:tf.zeros_like(x[:,0,:],dtype='int32'),name='zero_hist')
    
    # updates last NN choice
    add_selected_NN_hist = Lambda(lambda x:tf.add(x[0],tf.cast(x[1],dtype ='int32')),name='update_hist')
    
    
    next_node_error_list= gen_next_node_error(params)
    node_error = gen_node_error(params)
    
    STACK_1 = Lambda(lambda x: tf.cast(tf.stack(x,axis=1),dtype='float32'),name = 'STACK_Axis_1')
    
    F1_selected_hist = Lambda(lambda x: x[:,0,:]*0,name = 'select_history')(F1_set)
    
    for atom_idx in range(0,params['max_atoms']):
        
        sys.stdout.write("\r" + 'Generating atom index '+ str(atom_idx)+params['excess_space'])
        sys.stdout.flush()
    
        Finger_print = calc_remain_fp( [F1_set,total_z_error])
        AI_N = atom_decoder([Finger_print,F1_selected_hist])
        AI_N_list.append(AI_N)
        total_error,N_prob = node_error([AI_N,atoms, total_error])
        N_prob_list.append(N_prob)
        

        if(False):
            select_node = node_selector([AI_N,AI_N])
            AI_N_selected.append(node_selector([AI_N,AI_N]))
            if(not(params['True_Test'])):
                select_node = node_selector([AI_N,N_prob])
            
        else:
            select_node = node_selector([AI_N,N_prob])
        selected_N_list.append(select_node)
        training_node = select_node
    
        list_of_inputs = [Finger_print,training_node,F1_selected_hist]
    
        selected_NN_hist =  zero_selected_NN_hist(NN_Tensor)
        
        N_match = Lambda(lambda x:check_atoms_fun(x),name='Find_matching_nodes_'+str(atom_idx))([atoms,select_node])
    
        N_match_list.append(N_match)
        
        selected_NN_temp_list =[]
        NN_prob_temp_list = []
        AI_NN_temp_list = []
        AI_NN_temp_selected = []
        for degree in range(0,params['max_degree'] ):
            sys.stdout.write("\r" + 'Generating atom index '+ str(atom_idx)+ ' with degree ' + str(degree)+params['excess_space'])
            sys.stdout.flush()
            
            AI_NN = NN_decoder[degree](list_of_inputs+ [selected_NN_hist])
            
            total_error,NN_prob,NN_match= next_node_error_list[degree]([total_error,AI_NN,NN_Tensor,select_node, selected_NN_hist,N_match])
            
            if(False):
                selected_NN = NN_selector([AI_NN,AI_NN])

                if(not(params['True_Test'])):
                    selected_NN = NN_selector([AI_NN,NN_prob])

                
            else:
                selected_NN = NN_selector([AI_NN,NN_prob])
            
            selected_NN_hist =  add_selected_NN_hist([selected_NN_hist,selected_NN])

            
            
            selected_NN_temp_list.append(selected_NN)
            NN_prob_temp_list.append(NN_prob)
            AI_NN_temp_list.append(AI_NN)
            AI_NN_temp_selected.append(NN_selector([AI_NN,AI_NN]))
            
        selected_NN_list.append(STACK_1(selected_NN_temp_list))
        NN_prob_list.append(STACK_1(NN_prob_temp_list))
        AI_NN_list.append(STACK_1(AI_NN_temp_list))
        AI_NN_selected.append(STACK_1(AI_NN_temp_selected))
            

        NN_Tensor,atoms,F1_set,match,F1_selected_hist =tf.keras.layers.Lambda(lambda x:  remove_atom_idx(x,atom_idx),name = 'update_remaining_nodes_'+str(atom_idx))    ([ selected_NN_hist,NN_Tensor,atoms,F1_set,N_match,F1_selected_hist])
            
        match_list.append(match)

    total_error= Lambda(lambda x:  tf.identity(x),name = 'total_error_output')(total_error)
    
        
    
        
    '''
    #################################################################################
    Build Model
    '''
    

    


    
    if(False):
        NN_prob_output =STACK_1(NN_prob_list)
        
        match_output =STACK_1(match_list)

        N_prob_output= STACK_1(N_prob_list)

        AI_N_output = STACK_1(AI_N_list)
        
        AI_NN_output =STACK_1(AI_NN_list)

        
        selected_NN_output = STACK_1(selected_NN_list)
        
        
        selected_N_output =STACK_1(selected_N_list)
        
        sel_AI_NN_out = STACK_1(AI_NN_selected)
        
        sel_AI_N_out = STACK_1(AI_N_selected)

        
        N_match_out = STACK_1(N_match_list)
        

        
        model = models.Model(inputs=[atoms_in, edge, bonds,NN_Tensor_in], outputs=[NN_prob_output,match_output,N_prob_output,AI_N_output,AI_NN_output,selected_NN_output,selected_N_output,sel_AI_N_out,sel_AI_NN_out,N_match_out],name = params['model_name'])
    else:
        model = models.Model(inputs=[atoms_in, edge, bonds,NN_Tensor_in], outputs=[total_error],name = params['model_name'])
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,    beta1=0.9,    beta2=0.999,    epsilon=1e-08,    use_locking=False,    name='Adam')
    
    optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')


    return model



def gen_F1_testing_model(params_1,atom_decoder,NN_decoder):  
    
    params = copy.deepcopy(params_1)
    params['predict_model'] = not(params['rand_sample'])
    
   
    sys.stdout.write("\r" + 'Gen F1 decoder: Atom Decoder '+params['excess_space'])
    sys.stdout.flush()

    
    
    Finger_print = Input(name='Finger_print', shape=(params['finger_print'], ))
    F1_selected_hist = Input(name='Selected_hist', shape=(params['finger_print'], ))
    
    node_selector = gen_select_node( params,name = 'select_node')

    NN_selector = gen_select_next_node(params,name = 'select_NN')


    #Resets NN history to 0
    zero_selected_NN_hist =Lambda(lambda x:tf.zeros_like(tf.tile(x[:,0:1],[1,params['num_atom_features']*params['num_bond_features']+1]),dtype='int32'),name='zero_hist')
    
    # updates last NN choice
    add_selected_NN_hist = Lambda(lambda x:tf.add(x[0],tf.cast(x[1],dtype ='int32')),name='update_hist')

    STACK_1 = Lambda(lambda x: tf.cast(tf.stack(x,axis=1),dtype='float32'),name = 'STACK_Axis_1')
    AI_N = atom_decoder([Finger_print,F1_selected_hist])
    
    select_node = node_selector([AI_N,AI_N])
    list_of_inputs = [Finger_print,select_node,F1_selected_hist]

    selected_NN_hist =  zero_selected_NN_hist(Finger_print)
    

    
    NN_Tensor_slice_select =[]
    NN_Tensor_slice_prob = []

    for degree in range(0,params['max_degree'] ):
        sys.stdout.write("\r" + 'Gen F1 decoder: NN_decoder ' +str(degree)+params['excess_space'])
        sys.stdout.flush()
        
        AI_NN = NN_decoder[degree](list_of_inputs+ [selected_NN_hist])
        
        
        selected_NN = NN_selector([AI_NN,AI_NN])

        
        selected_NN_hist =  add_selected_NN_hist([selected_NN_hist,selected_NN])

        
        
        NN_Tensor_slice_select.append(selected_NN)
        NN_Tensor_slice_prob.append(AI_NN)
        
    NN_Tensor_slice_select = STACK_1(NN_Tensor_slice_select)
    NN_Tensor_slice_prob = STACK_1(NN_Tensor_slice_prob)


    F1_decoder = models.Model(inputs=[Finger_print, F1_selected_hist ], outputs=[select_node,NN_Tensor_slice_select,AI_N,NN_Tensor_slice_prob],name = 'F1_decoder')
    
    optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
    F1_decoder.compile(optimizer=optimizer, loss='mean_absolute_error')


    return F1_decoder
    



def vstack_img(img_top,img_bottom):
    
    imgs_comb = np.vstack( (np.asarray( i ) for i in [img_top,img_bottom] ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb 


    
