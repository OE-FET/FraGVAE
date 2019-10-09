''' Defines layers to build convolutional graph networks.
'''
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers,constraints,initializers,regularizers,activations
from tensorflow.keras.layers import Layer, Dense, InputSpec,Dropout
from tensorflow.python.framework import tensor_shape, common_shapes
from tensorflow.python.keras import backend as K
import sys
#from tensorflow.python.keras import constraints, initializers,regularizers,activations

import inspect
from itertools import cycle
# How to Modified to update to KERAS 2.0
#from tensorflow.layers.Layer import from_config as layer_from_config


class NeuralGraphHidden(layers.Layer):
    ''' Hidden Convolutional layer in a Neural Graph
    This layer takes a graph as an input. The graph is represented as by
    three tensors.

    - The atoms tensor represents the features of the nodes.NeuralGraphHidden(layers.Layer)
    - The bonds tensor represents the features of the edges.
    - The edges tensor represents the connectivity (which atoms are connected to
        which)

    It returns the convolved features tensor, which is very similar to the atoms
    tensor. Instead of each node being represented by a num_atom_features-sized
    vector, each node now is represented by a convolved feature vector of size
    conv_width.

    # Example
        Define the input:
        ```python
            atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features))
            bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features))
            edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
        ```
        The `NeuralGraphHidden` can be initialised in three ways:
        1. Using an integer `conv_width` and possible kwags (`Dense` layer is used)
            ```python
            atoms1 = NeuralGraphHidden(conv_width, activation='relu', bias=False)([atoms0, bonds, edges])
            ```
        2. Using an initialised `Dense` layer
            ```python
            atoms1 = NeuralGraphHidden(Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
            ```
        3. Using a function that returns an initialised `Dense` layer
            ```python
            atoms1 = NeuralGraphHidden(lambda: Dense(conv_width, activation='relu', bias=False))([atoms0, bonds, edges])
            ```

        Use `NeuralGraphOutput` to convert atom layer to fingerprint

    # Arguments
        inner_layer_arg: Either:
            1. an int defining the `conv_width`, with optional kwargs for the
                inner Dense layer
            2. An initialised but not build (`Dense`) keras layer (like a wrapper)
            3. A function that returns an initialised keras layer.
        kwargs: For initialisation 1. you can pass `Dense` layer kwargs

    # Input shape
        List of Atom and edge tensors of shape:
        `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
          bond_features), (samples, max_atoms, max_degrees)]`
        where degrees referes to number of neighbours

    # Output shape
        New atom featuers of shape
        `(samples, max_atoms, conv_width)`

    # References
        - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
        
        
            self.conv_width ,self.msg_width 
             inner_msg_arg

    '''

    def __init__(self, conv_width,msg_width,
                 update_bonds = False,
                 bond_hid_dim = 1,
                 z_bond_out_width = 10,
                 activation='elu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 alpha = 0,
                 inner_hdd_dim = 2, 
                 msg_hid_dim = 2,
                 bias_constraint=None, **kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.update_bonds = update_bonds
        self.z_bond_out_width = int(z_bond_out_width)
        self.bond_hid_dim = int(bond_hid_dim)
        self.msg_hid_dim = msg_hid_dim
        self.inner_hdd_dim = inner_hdd_dim
        self.alpha = alpha
        self.conv_width = int(conv_width)
        self.msg_width = int(msg_width)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        
        super(NeuralGraphHidden, self).__init__(          **kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
        num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        self.max_degree = max_degree

        input_dim = num_atom_features+ num_bond_features
        # Add the dense layers (that contain trainable params)
        self.inner_kernel_weights = []

        if(self.msg_hid_dim==0):
            self.msg_width = num_atom_features+num_bond_features
        
        
        for degree in range(0,max_degree):
            inner_degree = []
            inner_degree.append(self.add_weight( self.name + '_inner_kernel_degree_' + str(degree)+'_hid_'+str(0),  shape=[ self.msg_width+num_atom_features,self.conv_width],  initializer=self.kernel_initializer,    regularizer=self.kernel_regularizer,constraint=self.kernel_constraint, dtype=self.dtype,       trainable=True))
            for hid_index in range(1,self.inner_hdd_dim):
                inner_degree.append(self.add_weight( self.name + '_inner_kernel_degree_' + str(degree)+'_hid_'+str(hid_index),  shape=[self.conv_width, self.conv_width],  initializer=self.kernel_initializer,    regularizer=self.kernel_regularizer,constraint=self.kernel_constraint, dtype=self.dtype,       trainable=True))
            self.inner_kernel_weights.append(inner_degree)

        
        
        self.msg_layer = []
        self.msg_layer.append(self.add_weight(  self.name + '_msg_hid_'+str(0),     
              shape=[input_dim, self.msg_width],
              initializer=self.kernel_initializer,
              regularizer=self.kernel_regularizer,
              constraint=self.kernel_constraint,
              dtype=self.dtype,
              trainable=True))
        for hid_index in range(1,self.msg_hid_dim):
            self.msg_layer.append(self.add_weight( self.name + '_msg_hid_'+str(hid_index),     
              shape=[self.msg_width, self.msg_width],
              initializer=self.kernel_initializer,
              regularizer=self.kernel_regularizer,
              constraint=self.kernel_constraint,
              dtype=self.dtype,
              trainable=True))
            
            
        
        if(self.update_bonds):
            input_dim = self.conv_width+ num_bond_features
            self.bond_hid = []
            self.bond_hid.append(self.add_weight(  self.name + '_bond_hid_'+str(0),     
                  shape=[input_dim, self.z_bond_out_width],
                  initializer=self.kernel_initializer,
                  regularizer=self.kernel_regularizer,
                  constraint=self.kernel_constraint,
                  dtype=self.dtype,
                  trainable=True))
            for hid_index in range(1,self.bond_hid_dim):
                self.bond_hid.append(self.add_weight( self.name + '_bond_hid_'+str(hid_index),     
                  shape=[self.z_bond_out_width, self.z_bond_out_width],
                  initializer=self.kernel_initializer,
                  regularizer=self.kernel_regularizer,
                  constraint=self.kernel_constraint,
                  dtype=self.dtype,
                  trainable=True))
            

        super(NeuralGraphHidden, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        
        batch_n = K.shape(atoms)[0]
        # Import dimensions
        
        max_atoms = atoms.shape.as_list()[1]
        num_atom_features = atoms.shape.as_list()[-1]
        num_bond_features = bonds.shape.as_list()[-1]
        
        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = K.sum(K.cast(K.not_equal(edges, -1),'float32'), axis=-1, keepdims=True)
        #print('atom_degrees.shape'  + str(atom_degrees.shape))
        # For each atom, look up the features of it's neighbour
        neighbour_atom_features = neighbour_lookup(atoms, edges, include_self=False)
        #print('here')
        # Sum along degree axis to get summed neighbour features
        
        pre_msg_features = K.concatenate([neighbour_atom_features, bonds], axis=-1)
        
        flattened_pre_msg = K.reshape(pre_msg_features, (batch_n, max_atoms*self.max_degree, num_atom_features+num_bond_features))
        
        flattened_mask = K.reshape(K.not_equal(K.sum(neighbour_atom_features,axis =-1),0),[batch_n, max_atoms*self.max_degree, 1])
        if(self.msg_hid_dim>=1):
            flattened_mask_zeros = tf.broadcast_to(flattened_mask, [batch_n, max_atoms*self.max_degree,self.msg_width])
        else:
            flattened_mask_zeros = tf.broadcast_to(flattened_mask, [batch_n, max_atoms*self.max_degree,num_atom_features+num_bond_features])
        
        flattened_mask_zeros = tf.dtypes.cast(tf.dtypes.cast( flattened_mask_zeros, dtype = 'int32'), dtype = flattened_pre_msg.dtype)
        msg_kernel = []
        for i in range(0, self.msg_hid_dim):
            #print(self.msg_layer[i])
            msg_kernel.append(K.reshape(self.msg_layer[i],(1,self.msg_layer[i].shape[-2],self.msg_layer[i].shape[-1])))
            msg_kernel[i] = tf.tile(msg_kernel[i],(batch_n,1,1))
            #print(self.msg_layer[i] )
        

        msg_features = flattened_pre_msg
        for i in range(0,        self.msg_hid_dim):
            product = tf.linalg.matmul(msg_features,msg_kernel[i])
            msg_features  = self.activation(product)*flattened_mask_zeros

        if(self.msg_hid_dim>=1):
            msg_features = K.reshape(msg_features, (batch_n, max_atoms, self.max_degree, self.msg_width))
        else:
            msg_features = flattened_pre_msg*flattened_mask_zeros
            msg_features = K.reshape(msg_features, (batch_n, max_atoms, self.max_degree, num_atom_features+num_bond_features))
        
       
        
        summed_msg_features = K.sum(msg_features, axis=-2)
        #print('here')
        # Sum the edge features for each atom
        #print('here')
        # Concatenate the summed atom and bond features

        
        summed_features = K.concatenate([atoms, summed_msg_features], axis=-1)
        
        
        
        
        # For each degree we convolve with a different weight matrix
        new_features_by_degree = []
        
        
        
        

        for degree in range(0,self.max_degree):

            # Create mask for this degree

            atom_masks_this_degree = K.cast(K.equal(atom_degrees, degree), K.floatx())


            # Multiply with hidden merge layer
            #   (use time Distributed because we are dealing with 2D input/3D for batches)
            # Add keras shape to let keras now the dimensions
            
            inner_kernel = []
            for i in range(0, self.inner_hdd_dim):
                #print(self.msg_layer[i])
                inner_kernel.append(K.reshape(self.inner_kernel_weights[degree][i],(1,self.inner_kernel_weights[degree][i].shape[-2],self.inner_kernel_weights[degree][i].shape[-1])))
                inner_kernel[i] = tf.tile(inner_kernel[i],(batch_n,1,1))
            
            if(self.msg_hid_dim>=1):
                summed_features._keras_shape = (None, max_atoms, num_atom_features+self.msg_width)
            else:
                summed_features._keras_shape = (None, max_atoms, 2*num_atom_features+num_bond_features)
            new_unmasked_features = summed_features

            for i in range(0,   self.inner_hdd_dim):
                product = tf.linalg.matmul(new_unmasked_features,inner_kernel[i])
                new_unmasked_features  = self.activation(product)
            
            new_masked_features = tf.math.multiply(new_unmasked_features , atom_masks_this_degree)

            new_features_by_degree.append(new_masked_features)

        # Finally sum the features of all atoms
        new_atom_features = tf.math.add_n(new_features_by_degree)
        
        
        
        if(self.update_bonds):


            expanded_atoms = K.reshape(tf.tile(tf.expand_dims(new_atom_features,2),[1,1,self.max_degree,1]), (batch_n, max_atoms*self.max_degree, new_atom_features.shape[-1]))
            neighbour_atom_features = neighbour_lookup(new_atom_features, edges, include_self=False)
            neighbour_atom_features = K.reshape(neighbour_atom_features, (batch_n, max_atoms*self.max_degree, new_atom_features.shape[-1]))

            
            bond_atom_neighbours = tf.math.add(neighbour_atom_features,expanded_atoms)
                    
            update_bond_features = K.concatenate([bond_atom_neighbours, K.reshape(bonds, (batch_n, max_atoms*self.max_degree, num_bond_features))], axis=-1)
            
                            
            bond_mask = tf.broadcast_to(flattened_mask, [batch_n, max_atoms*self.max_degree,self.z_bond_out_width])
            
            bond_mask = tf.dtypes.cast(tf.dtypes.cast( bond_mask, dtype = 'int32'), dtype = update_bond_features.dtype)
            
            bond_kernel = []
            for i in range(0, self.bond_hid_dim):
                #print(self.msg_layer[i])
                bond_kernel.append(K.reshape(self.bond_hid[i],(1,self.bond_hid[i].shape[-2],self.bond_hid[i].shape[-1])))
                bond_kernel[i] = tf.tile(bond_kernel[i],(batch_n,1,1))
                #print(self.msg_layer[i] )
    
            product_unmasked = tf.linalg.matmul(update_bond_features,bond_kernel[0])
            bond_z  = tf.math.multiply(self.activation(product_unmasked),bond_mask)
            

            new_bonds_features = K.reshape(bond_z, (batch_n, max_atoms,self.max_degree, self.z_bond_out_width))
            
            
            return [new_atom_features,new_bonds_features,bond_z]
        
        else:
            bond_z = K.reshape(bonds, (batch_n, max_atoms*self.max_degree, self.z_bond_out_width))
            #return [new_atom_features,bonds,bond_z]
            return new_atom_features

    def get_output_shape_for(self, inputs_shape):

        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        return (num_samples, max_atoms, self.conv_width)
    def compute_output_shape(self, inputs_shape):
        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        return (num_samples, max_atoms, self.conv_width)
    
    
    def get_config(self):
                # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'z_bond_out_width':self.z_bond_out_width,
            'bond_hid_dim':self.bond_hid_dim,
            'update_bonds':self.update_bonds,
            'msg_hid_dim':self.msg_hid_dim,
            'inner_hdd_dim':self.inner_hdd_dim,
            'alpha':self.alpha,
            'conv_width': self.conv_width,
            'msg_width':self.msg_width,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NeuralGraphHidden, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






class NeuralGraphSparsify(layers.Layer):


    def __init__(self, fingerprint,activation = 'elu',kernel_regularizer= regularizers.l2(0.1),use_bias = False,**kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width

        self.fingerprint = fingerprint
        self.activation = activation
        self.use_bias=use_bias
        self.kernel_regularizer =  kernel_regularizer

        
        super(NeuralGraphSparsify, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        
        
        inner_layer = Dense(self.fingerprint ,name = self.name+'_GraphSparsify_dense',activation = self.activation, use_bias=self.use_bias,kernel_regularizer= self.kernel_regularizer)

        
        inner_3D_layer_name = self.name + '_GraphSparsify_inner_TD'
        self.sparsifyTD = layers.TimeDistributed(inner_layer, name=inner_3D_layer_name)

        super(NeuralGraphSparsify, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        Fp_set = inputs
        
        sparsify_Fp = self.sparsifyTD(Fp_set)


        
        mask = tf.math.reduce_sum(tf.math.abs(Fp_set),axis=-1)
        mask = tf.cast( tf.math.not_equal(mask, tf.zeros_like(mask)),dtype = Fp_set.dtype)
        mask = K.reshape(mask,[K.shape(Fp_set)[0],Fp_set.shape[1],1])
        
        sparsify_Fp = tf.math.multiply(sparsify_Fp, mask)
        # Sum outputs to obtain fingerprint
        
        
        return sparsify_Fp
    
    def get_config(self):
                # Case 1: Check if inner_layer_arg is conv_width


        
        config = {          
            'fingerprint':self.fingerprint,
            'activation':self.activation,
            'use_bias':self.use_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)}
        base_config = super(NeuralGraphSparsify, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class node_atom_error(layers.Layer):


    
    def __init__(self,   **kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width 
        
        super(node_atom_error, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
                
        super(node_atom_error, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        
        proposed_nodes,atoms, total_error = inputs 
        
        atoms = tf.cast(atoms,dtype = 'float32')
        #tiled_ra = K.reshape(remaining_nodes,(K.shape(remaining_nodes)[0],remaining_nodes.shape[1],1)) 
        #tiled_ra = tf.tile(tiled_ra,[1,1,atoms.shape[2]])
        remaining_atoms =  K.sum(atoms,axis =1)
        total_atoms = K.sum(remaining_atoms,axis =1) 
        
        no_atoms = K.equal(total_atoms, tf.math.multiply(tf.ones_like(total_atoms,dtype='float32'),tf.constant([0],dtype='float32')))
        no_atoms = tf.dtypes.cast(no_atoms,dtype = total_error.dtype )
        
        
        total_atoms = tf.math.add(  total_atoms,tf.ones_like(total_atoms)/10E4)
        
        total_atoms = K.reshape( total_atoms,(K.shape(total_atoms)[0],1))
        total_atoms = tf.tile(total_atoms,[1,remaining_atoms.shape[1].value])
        prob = tf.math.divide(remaining_atoms,total_atoms)
        
        true_prob = K.concatenate([K.reshape( no_atoms,[K.shape(proposed_nodes)[0],1]),prob],axis=1)
        
        AI_error = tf.math.multiply(tf.math.log(proposed_nodes),true_prob)
        true_error = tf.math.multiply(tf.math.log(tf.math.add(  true_prob,tf.ones_like(true_prob)/10E4)),true_prob)
        true_error = tf.stop_gradient(true_error)
        
        AI_error= K.sum(AI_error,axis=1)
        true_error= K.sum(true_error,axis=1)
        
        total_AI_error = tf.math.add(total_error, K.reshape(tf.math.subtract(AI_error,true_error),(K.shape(proposed_nodes)[0],1)))
        

        return [total_AI_error,true_prob]
    
    


class next_node_atom_error(layers.Layer):


    
    def __init__(self, degree, **kwargs):
        
        self.degree = degree
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width        
        super(next_node_atom_error, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        

    

        super(next_node_atom_error, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        
        total_error,AI_NN,  NN_Tensor, select_node, selected_NN_hist,N_match = inputs 
        '''
        atom_idx = 1
        degree = 0
        AI_NN = np.array([AI_NN[atom_idx][degree]])
        NN_Tensor = K.sum(np.array([mol_NN_Tensor[0]]),axis=2)
        selected_NN_hist = np.array([selected_NN[atom_idx][degree]])*0
        N_match  = np.array([N_match_out[atom_idx]])
        sess = tf.Session()
        NN_Tensor = K.sum(np.array([mol_NN_Tensor[0]]),axis=2)
        '''
        #atoms = tf.math.multiply(atoms,K.reshape(remaining_nodes,(K.shape(atoms)[0],atoms.shape[1],1)))
        
            
        Train_NN = tf.cast(selected_NN_hist,dtype ='int32')
        
        no_zeros = Train_NN[:,0]
        no_zeros = tf.cast(tf.math.equal(no_zeros,tf.zeros_like(no_zeros)),dtype = 'int32')
        no_zeros = K.reshape(no_zeros,(K.shape(no_zeros)[0],1))
        
        Train_NN = K.reshape(Train_NN,(K.shape(Train_NN)[0],1,Train_NN.shape[1]))

        #NN_Tensor = K.sum(NN_Tensor,axis =2)
        NN_Tensor = tf.math.subtract(NN_Tensor,Train_NN)

        NN_match = K.less(NN_Tensor, tf.zeros_like(NN_Tensor))
        NN_match = tf.cast(tf.math.logical_not(tf.math.reduce_any(NN_match,axis =2)),dtype='int32')
        F1_match = tf.math.multiply(  tf.math.multiply(  NN_match,N_match),no_zeros)

        
        NN_Tensor0, NN_TensorR = tf.split(NN_Tensor, [1,NN_Tensor.shape[2].value-1], 2)
        
        zero_candiates = tf.cast(tf.cast(NN_Tensor0[:,:,0],dtype='bool'),dtype='int32')
        zero_candiates = tf.math.multiply(  F1_match,zero_candiates)
        only_zero = tf.cast(tf.reduce_all(tf.equal(NN_TensorR, tf.zeros_like(NN_TensorR)),axis=2),dtype='int32')
        zero_candiates = tf.math.multiply(  only_zero,zero_candiates)
        zero_candiates = K.sum(zero_candiates,axis = 1)
        #find total number of canidates 
        zero_candiates = tf.cast(zero_candiates,dtype = 'float32')

        F1_match_reshaped = K.reshape(F1_match,(K.shape(F1_match)[0],F1_match.shape[1],1))
        NN_TensorR = tf.math.multiply(  NN_TensorR,F1_match_reshaped)
        
        
        NN_TensorR = tf.cast(NN_TensorR,dtype = 'float32')

        normal_Tensor =K.sum(NN_TensorR,axis = 2)
        normal_Tensor = tf.add(normal_Tensor,tf.ones_like(normal_Tensor)/10E4)
        normal_Tensor = K.reshape(normal_Tensor,(K.shape(normal_Tensor)[0],normal_Tensor.shape[1],1))
        NN_TensorR = tf.divide(NN_TensorR,normal_Tensor)
        
        
        remaining_NN = K.sum(NN_TensorR,axis = 1)
        remaining_NN = tf.concat([K.reshape(zero_candiates,(K.shape(zero_candiates)[0],1)),remaining_NN],axis =-1)
        total_remaining_NN  =  K.sum(remaining_NN ,axis =1)
        
        no_nodes = tf.cast(K.equal(total_remaining_NN, tf.zeros_like(total_remaining_NN,dtype=NN_TensorR.dtype)),dtype='float32')
        
        yes_nodes = K.not_equal(total_remaining_NN, tf.zeros_like(total_remaining_NN))
        yes_nodes = tf.dtypes.cast(yes_nodes,dtype = 'float32')
        
        total_remaining_NN = tf.math.add(  total_remaining_NN,tf.ones_like(total_remaining_NN,dtype='float32')/10E4)
        
        
        total_remaining_NN = K.reshape( total_remaining_NN,(K.shape(total_remaining_NN)[0],1))
        
        prob = tf.math.divide(remaining_NN,total_remaining_NN)

        
        

        No_valid_nodes = tf.concat([K.reshape(no_nodes,(K.shape(no_nodes)[0],1)),tf.tile(K.reshape(tf.zeros_like(no_nodes,dtype='float32'),
                                    (K.shape(no_nodes)[0],1)),[1,prob.shape[1]-1])],axis=1)
        
        

        true_prob = tf.math.multiply(prob,K.reshape(yes_nodes,(K.shape(yes_nodes)[0],1)))        
        true_prob = tf.math.add(true_prob,No_valid_nodes)

        
        

        
        AI_error =tf.math.multiply(tf.math.log(AI_NN),true_prob)
        true_error = tf.math.multiply(tf.math.log(tf.math.add(  true_prob,tf.ones_like(true_prob)/10E4)),true_prob)
        true_error = tf.stop_gradient(true_error)

        
        AI_error= K.sum(AI_error,axis=1)
        true_error= K.sum(true_error,axis=1)
        total_AI_error = tf.math.add(total_error, K.reshape(tf.math.subtract(AI_error,true_error),(K.shape(AI_error)[0],1)))
        
        
        
        return [total_AI_error,true_prob,NN_match]
        #return [total_error,NN_Tensor[:,0,0,:]]

    def get_config(self):
                # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'degree':self.degree,

        }
        base_config = super(next_node_atom_error, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






class Find_Ring_Atoms(layers.Layer):


    
    def __init__(self,max_atoms,max_degree,max_rings,  **kwargs):
        
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.max_atoms =max_atoms      
        self.max_degree =max_degree
        self.max_rings = max_rings
        super(Find_Ring_Atoms, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        

    

        super(Find_Ring_Atoms, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        
        
        edge =  inputs
        
        masked_edges = tf.cast(edge,dtype = 'int32')+1
        masked_edges = tf.pad(masked_edges,[[0,0],[1,0],[0,0]])
        
        leaf_not_found = tf.ones_like(masked_edges[:,:,0],dtype='int32')
        node_not_visted = tf.ones_like(leaf_not_found)
        Tree = tf.zeros_like(leaf_not_found)
        zero_batch_size = tf.reshape(tf.zeros_like(leaf_not_found[:,0]),shape = (tf.shape(edge)[0],1))
        ones_batch_size = tf.ones_like(zero_batch_size)
        num_rings = tf.zeros_like(zero_batch_size)
        ring_idx = tf.zeros_like(masked_edges[:,:,0:2],dtype='int32')
        
        index_batch = tf.reshape(tf.range(0, tf.shape(masked_edges)[0],dtype=tf.dtypes.int32),shape = (tf.shape(masked_edges)[0],1))
        for atom_idx in range(1, self.max_atoms+1):
            if atom_idx == 1:
                # select initial node to check from                
                node_not_visted = tf.tensor_scatter_update(node_not_visted, tf.concat([index_batch,zero_batch_size],axis = -1),zero_batch_size[:,0])
                leaf_not_found = tf.tensor_scatter_update(leaf_not_found, tf.concat([index_batch,zero_batch_size],axis = -1),zero_batch_size[:,0])
                leaf_not_found = tf.tensor_scatter_update(leaf_not_found, tf.concat([index_batch,ones_batch_size],axis = -1),zero_batch_size[:,0])
                # might have to change this value
                visiting_node = ones_batch_size
            else:
                # next node to visit must not have been visited previous
                # next node to visit must be a found leaf 
                leaf_found = tf.subtract(ones_batch_size, leaf_not_found)
                visiting_node = tf.reshape(tf.argmax(tf.multiply(node_not_visted,leaf_found),axis = -1,output_type=tf.dtypes.int32),shape = (tf.shape(visiting_node)[0],1))
                 
            
            
            for degree_idx in range(0, self.max_degree):
                # what does this node want to attach too
                next_node = tf.reshape(tf.gather_nd(masked_edges,tf.concat([index_batch,visiting_node],axis = -1))[:,degree_idx],shape = (tf.shape(visiting_node)[0],1))
                
                # determine if leaf has not been visited
                specific_leaf_not_found = tf.reshape(tf.gather_nd(leaf_not_found, tf.concat([index_batch,next_node],axis = -1)),shape = (tf.shape(visiting_node)[0],1))
                
                # determine if leaf was visited
                specific_leaf_found = tf.subtract(ones_batch_size, specific_leaf_not_found)
                
                specific_leaf_not_visited = tf.reshape(tf.gather_nd(node_not_visted, tf.concat([index_batch,next_node],axis = -1)),shape = (tf.shape(visiting_node)[0],1))
                
                # cycle found will find 2 cyles. 
                cycle_found = tf.multiply( tf.multiply(specific_leaf_found,specific_leaf_not_visited), tf.cast(tf.math.not_equal(next_node,tf.zeros_like(next_node)),dtype='int32'))
                
                add_node = tf.multiply(visiting_node,cycle_found)[:,0]
                ring_idx = tf.tensor_scatter_add(ring_idx, tf.concat([index_batch,num_rings,zero_batch_size],axis = -1), add_node)
                
                add_leaf = tf.multiply(next_node,cycle_found)[:,0]
                ring_idx = tf.tensor_scatter_add(ring_idx, tf.concat([index_batch,num_rings,ones_batch_size],axis = -1),add_leaf)
                
                num_rings = tf.add(num_rings, cycle_found)
                
                
                update_leaf = tf.multiply(specific_leaf_not_found,visiting_node)[:,0]
                Tree = tf.tensor_scatter_add(Tree, tf.concat([index_batch,next_node],axis = -1),update_leaf)
                
                leaf_not_found = tf.tensor_scatter_update(leaf_not_found, tf.concat([index_batch,next_node],axis = -1),zero_batch_size[:,0])
            node_not_visted = tf.tensor_scatter_update(node_not_visted, tf.concat([index_batch,visiting_node],axis = -1),zero_batch_size[:,0])
    
        max_num_rings = tf.reduce_max(num_rings)
        tf.print("max_num_rings:", max_num_rings,  output_stream=sys.stdout)
        
        rings = tf.tile(tf.zeros_like(masked_edges[:,0:1,0:1]),[1,self.max_rings,self.max_atoms],name = 'output_rings')
        while_inputs = [max_num_rings,rings,index_batch,ones_batch_size,Tree,ring_idx] 
            
        body = self.find_rings
        cond = self.more_rings
        outputs = tf.while_loop(cond, body, (0, while_inputs),return_same_structure= True)
        rings_output = outputs[1][1]
        rings_output = rings_output -1
        
        return [Tree,ring_idx,rings_output]
    
        #return [total_error,NN_Tensor[:,0,0,:]]
    def more_rings(self,i, while_inputs):
        max_num_rings = while_inputs[0]
        return tf.less(tf.ones_like(max_num_rings)*i,max_num_rings)
    def find_rings(self,i,while_inputs):
        [max_num_rings,rings,index_batch,ones_batch_size,Tree,ring_idx] = while_inputs
        
        atom_in_ring_idx= tf.zeros_like(ones_batch_size)
        
        
        
        # go up first branch of tree and label all elememnts which get visited
        starting_node = ring_idx[:,i][:,0:1]
        visted_branch_0 = tf.zeros_like(Tree)
        visted_branch_0 = tf.tensor_scatter_update(visted_branch_0,tf.concat([index_batch,starting_node],axis = -1),ones_batch_size[:,0])
        for j in range(0,self.max_atoms):
            
            next_node = tf.reshape(tf.gather_nd(Tree, tf.concat([index_batch,starting_node],axis = -1)),shape = (tf.shape(starting_node)[0],1))
            visted_branch_0 = tf.tensor_scatter_update(visted_branch_0,tf.concat([index_batch,next_node],axis = -1),ones_batch_size[:,0])
            starting_node = next_node
        # go up secound branch of tree and label all elememnts which get visited
        # add all elments to ring list which were not in bracnh 0 (except the first one that is in both)
        
        starting_node = ring_idx[:,i][:,1:2]
        visted_branch_1 = tf.zeros_like(Tree)
        visted_branch_1 = tf.tensor_scatter_update(visted_branch_1,tf.concat([index_batch,starting_node],axis = -1),ones_batch_size[:,0])
        rings = tf.tensor_scatter_update(rings, tf.concat([index_batch,tf.ones_like(atom_in_ring_idx)*i,atom_in_ring_idx],axis = -1),starting_node[:,0])
        
        

        for j in range(0,self.max_atoms):
            
            next_node =  tf.reshape(tf.gather_nd(Tree, tf.concat([index_batch,starting_node],axis = -1)),shape = (tf.shape(starting_node)[0],1))
            
            same_node_visted = tf.math.reduce_any(tf.equal(tf.multiply(visted_branch_0 , visted_branch_1),ones_batch_size),axis=-1)
            not_same_node_visted = tf.reshape(tf.cast(tf.logical_not(same_node_visted),dtype = 'int32'),shape = (tf.shape(starting_node)[0],1))
            
            atom_in_ring_idx = tf.math.add(atom_in_ring_idx, not_same_node_visted)
            
            rings = tf.tensor_scatter_add(rings,tf.concat([index_batch,tf.ones_like(atom_in_ring_idx)*i,atom_in_ring_idx],axis = -1),tf.multiply(next_node,not_same_node_visted)[:,0] ,name='jaun')
            visted_branch_1 = tf.tensor_scatter_update(visted_branch_1,tf.concat([index_batch,next_node],axis = -1),ones_batch_size[:,0])
            starting_node = next_node
        
        starting_node = ring_idx[:,i][:,0:1]
        visted_branch_0 = tf.zeros_like(Tree)
        visted_branch_0 = tf.tensor_scatter_update(visted_branch_0,tf.concat([index_batch,starting_node],axis = -1),ones_batch_size[:,0])
        atom_in_ring_idx= tf.math.add(atom_in_ring_idx, ones_batch_size)
        rings = tf.tensor_scatter_update(rings, tf.concat([index_batch,tf.ones_like(atom_in_ring_idx)*i,atom_in_ring_idx],axis = -1),starting_node[:,0])
        
        for j in range(0,self.max_atoms):
            
            next_node = tf.reshape(tf.gather_nd(Tree, tf.concat([index_batch,starting_node],axis = -1)),shape = (tf.shape(starting_node)[0],1))
            visted_branch_0 = tf.tensor_scatter_update(visted_branch_0,tf.concat([index_batch,next_node],axis = -1),ones_batch_size[:,0])

            same_node_visted = tf.math.reduce_any(tf.equal(tf.multiply(visted_branch_0 , visted_branch_1),ones_batch_size),axis=-1)
            
            not_same_node_visted =  tf.reshape(tf.cast(tf.logical_not(same_node_visted),dtype = 'int32'),shape = (tf.shape(starting_node)[0],1))
            
            atom_in_ring_idx = tf.math.add(atom_in_ring_idx, not_same_node_visted)
            
            rings = tf.tensor_scatter_add(rings,tf.concat([index_batch,tf.ones_like(atom_in_ring_idx)*i,atom_in_ring_idx],axis = -1),tf.multiply(next_node,not_same_node_visted)[:,0],name='jaunetta')
            starting_node = next_node
            
        i = i + 1 
        return (i,[max_num_rings,rings,index_batch,ones_batch_size,Tree,ring_idx])
    
    def get_config(self):
                # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'degree':self.max_degree,
            'max_atoms':self.max_atoms,
            'max_rings':self.max_rings

        }
        base_config = super(Find_Ring_Atoms, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))










class Find_Ring_Bonds(layers.Layer):


    
    def __init__(self,max_atoms,max_degree,max_rings,  **kwargs):
        
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.max_atoms =max_atoms      
        self.max_degree =max_degree
        self.max_rings = max_rings
        super(Find_Ring_Bonds, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        

    

        super(Find_Ring_Bonds, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        
        
        edges,rings =  inputs
        edges = tf.cast(edges,dtype='int32')
        bonds_in_ring = tf.zeros_like(edges)

        index_batch = tf.reshape(tf.range(0, tf.shape(edges)[0],dtype=tf.dtypes.int32),shape = (tf.shape(edges)[0],1))
        ones_like_index_batch = tf.ones_like(index_batch)
        for atom_idx in range(0, self.max_atoms):
            atom_idx_in_ring = tf.cast(tf.math.reduce_any(tf.equal(rings,tf.ones_like(rings)*atom_idx),axis =-1),dtype = 'int32')
            
            
            for degree_idx in range(0, self.max_degree):
                # what does this node want to attach too
                NN_atom  =  edges[:,atom_idx,degree_idx]
                
                NN_atom_in_ring = tf.cast(tf.math.reduce_any(tf.equal(rings,edges[:,atom_idx:atom_idx+1,degree_idx:degree_idx+1]),axis=-1),dtype = 'int32')
                                
                NN_atom_not_null =  tf.cast(tf.math.not_equal(NN_atom,tf.ones_like(NN_atom,dtype=NN_atom.dtype)*-1),dtype = 'int32')
                
                valid_ring_bond = tf.multiply(tf.math.reduce_max(tf.multiply(NN_atom_in_ring,atom_idx_in_ring),axis=-1),NN_atom_not_null)
                
                bonds_in_ring = tf.tensor_scatter_add(bonds_in_ring, tf.concat([index_batch,ones_like_index_batch*atom_idx,ones_like_index_batch*degree_idx],axis = -1), valid_ring_bond)
                    
                   

        return [tf.cast(tf.reshape(bonds_in_ring,shape = (tf.shape(edges)[0],self.max_atoms,self.max_degree,1)),dtype='float32')]
 
    
    def get_config(self):
                # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'max_degree':self.max_degree,
            'max_degree':      self.max_degree,
            'max_rings':self.max_rings

        }
        base_config = super(Find_Ring_Bonds, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Mask_DB_atoms(layers.Layer):

    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.


    def __init__(self, **kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width

        

        
        super(Mask_DB_atoms, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions

        super(Mask_DB_atoms, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        atoms,DB_atoms = inputs
        DB_atoms_mask = 1-K.reshape(DB_atoms,[K.shape(DB_atoms)[0],DB_atoms.shape[1],1])
        atoms = tf.math.multiply(atoms,DB_atoms_mask)

        
        return atoms
    

class Ring_Edge_Mask(layers.Layer):

    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.


    def __init__(self,with_aromatic , **kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.with_aromatic  = with_aromatic
        

        
        super(Ring_Edge_Mask, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions

        super(Ring_Edge_Mask, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        
        if(self.with_aromatic):
            edge,bonds_in_rings,bonds_types,R_Dangling_atoms = inputs
            
            bonds_in_rings_or_aromatic = tf.cast(tf.math.not_equal( tf.math.add(bonds_in_rings[:,:,:,0],bonds_types[:,:,:,-1]),tf.zeros_like(bonds_in_rings[:,:,:,0])),dtype = bonds_types.dtype)
            bonds_not_in_rings = 1-bonds_in_rings_or_aromatic
            edge_in_rings = tf.math.multiply(edge,bonds_in_rings_or_aromatic)
            edge_in_rings = tf.math.subtract(edge_in_rings,bonds_not_in_rings)
            edge_in_rings = tf.cast(edge_in_rings,dtype = edge.dtype)\
            

            
            
        else:
            edge,bonds_in_rings,R_Dangling_atoms = inputs
        
            bonds_in_rings = bonds_in_rings[:,:,:,0]
            bonds_not_in_rings = 1-bonds_in_rings
            edge_in_rings = tf.math.multiply(edge,bonds_in_rings)
            edge_in_rings = tf.math.subtract(edge_in_rings,bonds_not_in_rings)
            edge_in_rings = tf.cast(edge_in_rings,dtype = edge.dtype)

        atoms_with_zero_edges = tf.cast(tf.math.reduce_all(tf.math.equal(edge_in_rings, -1),axis=-1),dtype = bonds_in_rings.dtype)
        atoms_with_zero_edges = K.reshape(atoms_with_zero_edges,[K.shape(atoms_with_zero_edges)[0],atoms_with_zero_edges.shape[1],1])
        
        atoms_not_in_rings = tf.cast(tf.math.not_equal(tf.math.add(atoms_with_zero_edges,R_Dangling_atoms),tf.zeros_like(R_Dangling_atoms)),dtype = atoms_with_zero_edges.dtype)
        return [edge_in_rings,atoms_not_in_rings]
    
    def get_config(self):
        # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'with_aromatic': self.with_aromatic}
        base_config = super(Ring_Edge_Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Variational(layers.Layer):

    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.


    def __init__(self,  init_seed = 1,**kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.init_seed = init_seed
        self.kernel_initializer = tf.initializers.constant(value = self.init_seed)
        

        
        super(Variational, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        self.kl_loss_var = self.add_weight( 'kl_loss_var',  shape=(),  initializer=self.kernel_initializer,   dtype=self.dtype,       trainable=False)

        super(Variational, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        z_log_var,z = inputs
        z_log_var = tf.math.tanh(z_log_var)
        epsilon = tf.random.normal(shape=K.shape(z_log_var), mean=0.0, stddev=1.0)
        # insert kl loss here
        
        
        z_error =  tf.math.multiply(tf.math.exp(z_log_var / 2) , self.kl_loss_var * epsilon)
        
        mask = tf.math.reduce_sum(tf.math.abs(z),axis=-1)
        mask = tf.cast( tf.math.not_equal(mask, tf.zeros_like(mask)),dtype = z_log_var.dtype)
        mask = K.reshape(mask,[K.shape(z)[0],z.shape[1],1])
        z_error = tf.math.multiply(z_error,mask)
        z_error = tf.math.reduce_sum(z_error,axis = -2)
        
        
        
        
        
        
        return K.in_train_phase(z_error, z_error*0)
    
    def get_config(self):
        # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'init_seed': self.init_seed}
        base_config = super(Variational, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class FHO_Error(layers.Layer):

    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.


    def __init__(self,invalid, **kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.invalid = invalid
        if(invalid):
            self.class_idx = 1
        else:
            self.class_idx = 0

        
        super(FHO_Error, self).__init__(**kwargs)

    def build(self, inputs_shape):


        super(FHO_Error, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        
        perdict,  error = inputs 

        AI_error = tf.math.log(tf.math.add( perdict[:,self.class_idx],tf.ones_like(perdict[:,self.class_idx])/10E12))
        total_AI_error = tf.math.add(error,AI_error )
        return total_AI_error
    
    def get_config(self):
        # Case 1: Check if inner_layer_arg is conv_width


        
        config = {                  'invalid': self.invalid}
        base_config = super(FHO_Error, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Hide_N_Drop(layers.Layer):

    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.


    def __init__(self,  hide_num_bonds = 0,hide_num_atoms = 0, dropout_rate = 0,**kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        self.hide_num_bonds = hide_num_bonds
        self.hide_num_atoms = hide_num_atoms
        self.dropout_rate = dropout_rate

        
        super(Hide_N_Drop, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        self.drop_layer = Dropout(self.dropout_rate,name = self.name+'_dropout')
        super(Hide_N_Drop, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        bonds_sparse,atoms_sparse = inputs
        slice_bonds = bonds_sparse
        slice_atoms = atoms_sparse
        if(self.hide_num_bonds!=0):
            slice_bonds = bonds_sparse[:,:,self.hide_num_bonds:int(bonds_sparse.shape[-1])]

            
        if(self.hide_num_atoms!=0):
            slice_atoms = atoms_sparse[:,:,self.hide_num_atoms:int(atoms_sparse.shape[-1])]

        
        if(self.dropout_rate!=0):
            slice_atoms = self.drop_layer(slice_atoms)
            slice_bonds = self.drop_layer(slice_bonds)
        
        
        
        
        return [slice_bonds,slice_atoms]
    
    def get_config(self):
        # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'hide_num_bonds': self.hide_num_bonds,
            'hide_num_atoms': self.hide_num_atoms,
            'dropout_rate': self.dropout_rate
            }
        base_config = super(Hide_N_Drop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





def filter_func_args(fn, args, invalid_args=[], overrule_args=[]):
    '''
    Separate a dict of arguments into one that a function takes, and the rest

    # Arguments:
        fn: arbitrary function
        args: dict of arguments to separate
        invalid_args: list of arguments that will be removed from args
        overrule_args: list of arguments that will be returned in other_args,
            even if they are arguments that `fn` takes

    # Returns:
        fn_args, other_args: tuple of separated arguments, ones that the function
            takes, and the others (minus `invalid_args`)
    
    '''
    fn_valid_args = inspect.getargspec(fn)[0]
    fn_args = {}
    other_args = {}
    for arg, val in args.items():
        if not arg in invalid_args:
            if (arg in fn_valid_args) and (arg not in overrule_args):
                fn_args[arg] = val
            else:
                other_args[arg] = val
    return fn_args, other_args

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False



def mol_dims_to_shapes(max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules=None):
    '''
     Helper function, returns shape for molecule tensors given dim sizes
    '''
    
    atoms_shape = (num_molecules, max_atoms, num_atom_features)
    bonds_shape = (num_molecules, max_atoms, max_degree, num_bond_features)
    edges_shape = (num_molecules, max_atoms, max_degree)
    return [atoms_shape, bonds_shape, edges_shape]

def mol_shapes_to_dims(mol_tensors=None, mol_shapes=None):
    '''
     Helper function, returns dim sizes for molecule tensors given tensors or
    tensor shapes
    '''
    
    if not mol_shapes:
        mol_shapes = [t.shape for t in mol_tensors]

    num_molecules0, max_atoms0, num_atom_features = mol_shapes[0].as_list()
    num_molecules1, max_atoms1, max_degree1, num_bond_features = mol_shapes[1].as_list()
    num_molecules2, max_atoms2, max_degree2 = mol_shapes[2].as_list()

    num_molecules_vals = [num_molecules0, num_molecules1, num_molecules2]
    max_atoms_vals = [max_atoms0, max_atoms1, max_atoms2]
    max_degree_vals = [max_degree1, max_degree2]

    assert len(set(num_molecules_vals))==1, 'num_molecules does not match within tensors (found: {})'.format(num_molecules_vals)
    assert len(set(max_atoms_vals))==1, 'max_atoms does not match within tensors (found: {})'.format(max_atoms_vals)
    assert len(set(max_degree_vals))==1, 'max_degree does not match within tensors (found: {})'.format(max_degree_vals)

    return max_atoms1, max_degree1, num_atom_features, num_bond_features, num_molecules1




def neighbour_lookup(atoms, edges, maskvalue=0, include_self=False):
    ''' 
    Looks up the features of an all atoms neighbours, for a batch of molecules.

    # Arguments:
        atoms (K.tensor): of shape (batch_n, max_atoms, num_atom_features)
        edges (K.tensor): of shape (batch_n, max_atoms, max_degree) with neighbour
            indices and -1 as padding value
        maskvalue (numerical): the maskingvalue that should be used for empty atoms
            or atoms that have no neighbours (does not affect the input maskvalue
            which should always be -1!)
        include_self (bool): if True, the featurevector of each atom will be added
            to the list feature vectors of its neighbours

    # Returns:
        neigbour_features (K.tensor): of shape (batch_n, max_atoms(+1), max_degree,
            num_atom_features) depending on the value of include_self

    # Todo:
        - make this function compatible with Tensorflow, it should be quite trivial
            because there is an equivalent of `T.arange` in tensorflow.
    
    '''
    # The lookup masking trick: We add 1 to all indices, converting the
    #   masking value of -1 to a valid 0 index.
    masked_edges = edges + 1
    # We then add a padding vector at index 0 by padding to the left of the
    #   lookup matrix with the value that the new mask should get
    
    #Original
    #masked_atoms = temporal_padding(atoms, (1,0), padvalue=maskvalue)
    #Replaced
    masked_atoms = tf.pad(atoms,[[0,0],[1,0],[0,0]])

    # Import dimensions
    atoms_shape = K.shape(masked_atoms)
    batch_n = atoms_shape[0]
    lookup_size = atoms_shape[1]
    num_atom_features = atoms_shape[2]

    edges_shape = K.shape(masked_edges)
    max_atoms = edges_shape[1]
    max_degree = edges_shape[2]

    # create broadcastable offset
    offset_shape = (batch_n, 1, 1)
    offset = K.reshape(tf.range(batch_n, dtype='int32'), offset_shape)
    offset *= lookup_size
    
    masked_edges = tf.dtypes.cast( masked_edges, dtype = 'int32')
    
    # apply offset to account for the fact that after reshape, all individual
    #   batch_n indices will be combined into a single big index
    flattened_atoms = K.reshape(masked_atoms, (-1, num_atom_features))
    flattened_edges = K.reshape(masked_edges + offset, (batch_n, -1))

    # Gather flattened
    flattened_result = K.gather(flattened_atoms, flattened_edges)

    # Unflatten result
    output_shape = (batch_n, max_atoms, max_degree, num_atom_features)
    output = tf.reshape(flattened_result, output_shape)

    if include_self:
        return K.concatenate([K.expand_dims(atoms, axis=2), output], axis=2)
    return output







class Tanimoto(layers.Layer):

    # @inp mean : mean generated from encoder
    # @inp enc : output generated by encoding
    # @inp params : parameter dictionary passed throughout entire model.


    def __init__(self,FHO_activation, **kwargs):
        # Initialise based on one of the three initialisation methods
        # Case 1: Check if inner_layer_arg is conv_width
        
        if(FHO_activation!='elu' and FHO_activation!='relu'  and FHO_activation!='tanh' ):
            print('Tanimoto test is currently NOT VALID!')
            
            
        self.FHO_activation = FHO_activation


        
        super(Tanimoto, self).__init__(**kwargs)

    def build(self, inputs_shape):

        # Import dimensions
        super(Tanimoto, self).build(inputs_shape)
        
    def call(self, inputs, mask=None):
        z,z_edmol_next,z_edmol = inputs
        if(self.FHO_activation=='elu' or self.FHO_activation=='tanh'):
            z = tf.math.maximum(tf.math.add(z , tf.ones_like(z)),  tf.zeros_like(z))
            
            z_edmol_next = tf.math.add(z_edmol_next , tf.ones_like(z_edmol_next))
            
            z_edmol = tf.math.add(z_edmol, tf.ones_like(z_edmol))
        elif(self.FHO_activation=='relu'):
            z = tf.math.maximum(z,  tf.zeros_like(z))
            z_edmol_next = z_edmol_next
            
            z_edmol = z_edmol

        
        def Tanimoto_cal(z1,z2):
            tanimoto_max = tf.math.maximum(z1,z2)
            tanimoto_min = tf.math.minimum(z1,z2)
            
            tanimoto_max = tf.reduce_sum(tanimoto_max,-1)
            tanimoto_max = tf.add(tanimoto_max,tf.ones_like(tanimoto_max)*1E-15)
            tanimoto_min = tf.reduce_sum(tanimoto_min,-1)
            
            tanimoto_val = tf.math.divide(tanimoto_min,tanimoto_max)
            
            return tanimoto_val
        
        edmol_next_Tanimoto =  K.reshape(Tanimoto_cal(z_edmol_next,z),[K.shape(z)[0],1])
        
        edmol_Tanimoto =  K.reshape(Tanimoto_cal(z_edmol,z),[K.shape(z)[0],1])
        
        diff_Tanimoto = tf.math.subtract(edmol_next_Tanimoto,edmol_Tanimoto)
        
        
        
        tanimoto_values = tf.keras.layers.concatenate([edmol_next_Tanimoto,edmol_Tanimoto,diff_Tanimoto],-1)
        
        return tanimoto_values
    
    def get_config(self):
        # Case 1: Check if inner_layer_arg is conv_width


        
        config = {
            'FHO_activation': self.FHO_activation
            }
        base_config = super(Tanimoto, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))