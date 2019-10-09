# -*- coding: utf-8 -*-
"""
@author: John Armitage
Email: jwarmitage@gmail.com

"""

from .hyperparameters import load_params,save_params, calc_num_atom_features
from .generators import f1_data_generator,fho_data_generator,gen_train_mol_tensor
from .f1_models import gen_F1_encoder, gen_F1_N_decoder,gen_F1_NN_decoder,gen_F1_training_model,gen_F1_testing_model
from .fho_models import gen_FHO_encoder,gen_FHO_decoder, gen_train_FHO_VAE
from .vae_callbacks import WeightAnnealer_epoch,sigmoid_schedule
from .convert_mol_smile_tensor import smile_to_tensor,smile_to_mol,mol_to_tensor,gen_atom_features,gen_bond_features,atom_from_atom_feature,bond_from_bond_feature,gen_sparse_NN_Tensor,atom_bond_from_sparse_NN_feature,gen_sparse_NN_feature
from .display_funs import display_F1_graphs
from io import BytesIO
from rdkit.Chem import AllChem as Chem
import tensorflow as tf
import numpy as np
import sys
import copy 
import matplotlib.pyplot as plt
import PIL
import os
import pandas as pd



class FraGVAE:
    '''
    FraGVAE object
    
    '''
    
    def __init__(self, experiment_number,verbose = False,train_dataset='',CV_dataset=''): 
        self.params = load_params( 'models/experiment'+str(experiment_number).zfill(6)+'/exp.json',verbose=verbose)
        self.params['exp'] =   experiment_number
        self.params['model_dir'] =   'models/experiment'+str(self.params['exp']).zfill(6)+'/'
        if not os.path.exists(self.params['model_dir'] ):
            os.makedirs(self.params['model_dir'] )
        if(train_dataset!=''):
            self.params['train_dataset'] = train_dataset
        if(CV_dataset!=''):
            self.params['CV_dataset'] = CV_dataset

        self.save_params()
        
    def get_params(self):
        return self.params
    
    def set_params(self,params):
        self.params = params
    
    def save_params(self):
        save_params(self.params, self.params['model_dir']+'exp.json')    
    
    
    def autoencode_mol(self,mol, display_process = False, trial_walkthrough = False ):
        '''
        Complete autoencoding of a molecular graph. 
        - mol: rdkit mol for autoencoding
        - display_process: param for creating images of full process
        - trial_walkthrough: param for determining troubleshooting walkthrough
        '''
        if(trial_walkthrough):
            mol_recon =  mol
        else:
            mol_recon = -1
            
                    
        atoms, edges,bonds = mol_to_tensor(mol, self.params,FHO_Ring_feature=self.params['FHO_Ring_feature']) 
        
        atoms = np.array([atoms])
        edges = np.array([edges])
        bonds = np.array([bonds])
        Z_1 , Z_HO,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS = self.Z_encoder(atoms, bonds ,edges)

        mol_output = self.Z_decoder(Z_1,Z_HO,display_process = display_process, mol_recon=mol_recon, trial_walkthrough= trial_walkthrough, max_P = self.params['FHO_max_P'])

    

        return mol_output
    
    
    



    def load_models(self,rnd=False,testing = True):
        
        self.load_encoders(rnd=rnd)
        self.load_decoders(rnd=rnd, testing = testing)

        
    def load_decoders(self, rnd = False, testing = True):
        
        self.load_F1_decoders(rnd = rnd)
        
        if(testing):
            sys.stdout.write("\r" + 'Generating F1 testing model'+self.params['excess_space'])
            sys.stdout.flush()
            self.F1_decoder = gen_F1_testing_model(self.params,self.atom_decoder,self.NN_decoder)
        
        self.load_FHO_decoders(rnd = rnd)


    def load_F1_decoders(self,rnd = False):
        '''
        load F1 FraGVAE decoder models 
        '''
        if(rnd==False):
            from os import path
            if(self.params['Best_F1_epoch_num']!=-1):
                file_name = self.params['model_dir']+'F1_Atom_Decoder_epoch_'+str(self.params['Best_F1_epoch_num'])+'.h5'
            else:
                file_name = self.params['model_dir']+'F1_Atom_Decoder.h5'
                
            if(not(path.isfile(file_name))):
                print('F1 decoder model is not saved, generating random model')
                rnd=True
        
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        
        sys.stdout.write("\r" + 'Generating F1 atom model'+self.params['excess_space'])
        sys.stdout.flush()
        self.atom_decoder = gen_F1_N_decoder(self.params,name ='Atom_decoder')
        self.atom_decoder.compile(optimizer=optimizer, loss='mean_absolute_error')
        if(not(rnd)):
            if(self.params['Best_F1_epoch_num']!=-1):
                self.atom_decoder.load_weights(self.params['model_dir']+'F1_Atom_Decoder_epoch_'+str(self.params['Best_F1_epoch_num'])+'.h5')
            else:
                self.atom_decoder.load_weights(self.params['model_dir']+'F1_Atom_Decoder.h5')
        
        self.NN_decoder = gen_F1_NN_decoder(self.params,name ='Decoder_degree')
        for degree_idx in range(0,self.params['max_degree'] ):
            sys.stdout.write("\r" + 'Generating F1_NN model degree '+str(degree_idx)+self.params['excess_space'])
            sys.stdout.flush()
            self.NN_decoder[degree_idx].compile(optimizer=optimizer, loss='mean_absolute_error')
            if(not(rnd)):
                if(self.params['Best_F1_epoch_num']!=-1):
                    self.NN_decoder[degree_idx].load_weights(self.params['model_dir']+'F1_NN_decoder_degree_'+str(degree_idx)+'_epoch_'+str(self.params['Best_F1_epoch_num'])+'.h5')  
                else:
                    self.NN_decoder[degree_idx].load_weights(self.params['model_dir']+'F1_NN_decoder_degree_'+str(degree_idx)+'.h5')  

    def load_FHO_decoders(self, rnd = False):
        
        
        if(rnd==False):
            from os import path
            if(self.params['Best_FHO_epoch_num']!=-1):
                file_name = self.params['model_dir']+'FHO_decoder_epoch_'+str(self.params['Best_FHO_epoch_num'])+'.h5'
            else:
                file_name = self.params['model_dir']+'FHO_decoder.h5'
                
            if(not(path.isfile(file_name))):
                print('FHO decoder model is not saved, generating random model')
                rnd=True
                
                
        sys.stdout.write("\r" + 'Generating FHO model'+self.params['excess_space'])
        sys.stdout.flush()
        self.FHO_decoder = gen_FHO_decoder(self.params,'FHO_decoder')
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        self.FHO_decoder.compile(optimizer=optimizer, loss='mean_absolute_error')
        if(not(rnd)):
            if(self.params['Best_FHO_epoch_num']!=-1):
                self.FHO_decoder.load_weights(self.params['model_dir']+'FHO_decoder_epoch_'+str(self.params['Best_FHO_epoch_num'])+'.h5')
            else:
                self.FHO_decoder.load_weights(self.params['model_dir']+'FHO_decoder.h5')
        self.FHO_decoder_fun = self.gen_dropout_fun(self.FHO_decoder)
        
    def load_encoders(self, rnd = False):
        '''
        load FraGVAE encoder models from save file
        '''        
        self.load_FHO_encoders(rnd = rnd)

        self.load_F1_encoders(rnd = rnd)
        
    def load_F1_encoders(self, rnd = False):
       
        
        if(rnd==False):
            from os import path
            if(self.params['Best_F1_epoch_num']!=-1):
                file_name = self.params['model_dir']+'F1_Encoder_epoch_'+str(self.params['Best_F1_epoch_num'])+'.h5'
            else:
                file_name = self.params['model_dir']+'F1_Encoder.h5'
                
            if(not(path.isfile(file_name))):
                print('F1 encoder model is not saved, generating random model')
                rnd=True

        
        self.F1_encoder = gen_F1_encoder(self.params,'F1_Encoder')
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        self.F1_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')
        if(not(rnd)):
            if(self.params['Best_F1_epoch_num']!=-1):
                self.F1_encoder.load_weights(self.params['model_dir']+'F1_Encoder_epoch_'+str(self.params['Best_F1_epoch_num'])+'.h5')
            else:
                self.F1_encoder.load_weights(self.params['model_dir']+'F1_Encoder.h5')

    def load_FHO_encoders(self, rnd = False):

        
        if(rnd==False):
            from os import path
            if(self.params['Best_FHO_epoch_num']!=-1):
                file_name = self.params['model_dir']+'FHO_encoder_epoch_'+str(self.params['Best_FHO_epoch_num'])+'.h5'
            else:
                file_name = self.params['model_dir']+'FHO_encoder.h5'
                
            if(not(path.isfile(file_name))):
                print('FHO encoder model is not saved, generating random model')
                rnd=True
                
        self.FHO_encoder = gen_FHO_encoder(self.params,'FHO_encoder')
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        self.FHO_encoder.compile(optimizer=optimizer, loss='mean_absolute_error')
        if(not(rnd)):
            if(self.params['Best_FHO_epoch_num']!=-1):
                self.FHO_encoder.load_weights(self.params['model_dir']+'FHO_encoder_epoch_'+str(self.params['Best_FHO_epoch_num'])+'.h5')
            else:
                self.FHO_encoder.load_weights(self.params['model_dir']+'FHO_encoder.h5')
                
    
    def filter_csv(self,filename):
        '''
        Filters csv file of smiles to remove molecules which can not be handled by FraGVAE based on preset params
        '''
        libExamples = pd.read_csv('data_lib/Unfiltered_data/'+filename+'.csv')
        remove_IDX = []
        max_num_bonds = 0
        remove_count = 0
        for i in range(0, len(libExamples)):
    
            remove = False
            smile = libExamples['smiles'][i]
            mol = Chem.MolFromSmiles(smile)
            if(mol.GetNumBonds()>max_num_bonds):
                max_num_bonds = mol.GetNumBonds()
                print(max_num_bonds)
            if(mol.GetNumBonds()<=2):
                remove = True
                reason = 'not enough bonds'
            elif(mol.GetNumAtoms()>self.params['max_atoms'] ):
                remove = True
                reason = 'too many atoms'
            elif(mol.GetNumBonds()>self.params['max_bonds']):
                remove = True
                reason = 'too many bonds'
            
            for j in range(0,mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(j)
                if(atom.GetDegree()>self.params['max_degree']):
                    remove = True
                    reason = 'too high valence'
                elif(not(any([ ATOM == atom.GetSymbol() for ATOM in self.params['atoms']]))):
                    remove = True
                    reason = atom.GetSymbol() + ' is not recognized'
            if(remove):
                remove_count = remove_count+1
                remove_IDX.append(i)
                print('Remove mol @ index '+ str(i)+' due to '+reason+'. Removed a total of '+str(remove_count))
            
                
        filter_libExamples = libExamples.drop(remove_IDX)
        filter_libExamples = filter_libExamples.reset_index(drop=True)
        
        filter_libExamples.to_csv('data_lib/'+filename+'filtered.csv',index=False)

    def train_f1_models(self):
        
        self.f1_train_model = gen_F1_training_model(self.params, F1_encoder = self.F1_encoder, atom_decoder = self.atom_decoder, NN_decoder =self.NN_decoder)
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        self.f1_train_model.compile(optimizer=optimizer, loss='mean_absolute_error')
        
        self.train_model(self.f1_train_model, training_fho = False)
        
    def train_fho_models(self):
        
        self.fho_train_model = gen_train_FHO_VAE(self.params,self.params['model_name'],self.FHO_encoder,self.FHO_decoder)
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        self.fho_train_model.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.train_model(self.fho_train_model, training_fho = True)

    
    def train_model(self,model, training_fho = True):
        from functools import partial

        '''
        Trains either F1 or FHO model
        '''
        
        libExamples_train = pd.read_csv('data_lib/'+self.params['train_dataset']+'.csv')
        libExamples_train = libExamples_train.sample(frac=1).reset_index(drop=True) 
    
        libExamples_CV = pd.read_csv('data_lib/'+self.params['CV_dataset']+'.csv')
        libExamples_CV = libExamples_CV.sample(frac=1).reset_index(drop=True) 

        if(training_fho):
            anneal_sigmod_slope = self.params['FHO_anneal_sigmod_slope']
            kl_loss_weight =  self.params['FHO_kl_loss_weight']
            anneal_epoch_start = self.params['FHO_anneal_epoch_start']
            epoch_start = self.params['FHO_epoch_start']
            steps_per_epoch = self.params['FHO_steps_per_epoch']
        else:
            anneal_sigmod_slope = self.params['F1_anneal_sigmod_slope']
            kl_loss_weight =  self.params['F1_kl_loss_weight']
            anneal_epoch_start = self.params['F1_anneal_epoch_start']
            epoch_start = self.params['F1_epoch_start']
            steps_per_epoch = self.params['F1_steps_per_epoch']

                      
     
            
        # Add variational loss parameter    
        num_kl_loss_var = 0
        for i in range(0,len(model.non_trainable_variables)):
            if('Variational_layer' in model.non_trainable_variables[i].name):
                kl_loss_var = model.non_trainable_variables[i]
                num_kl_loss_var =num_kl_loss_var+1
        if(num_kl_loss_var>1):
            print('')
            print('UNCERTAIN ABOUT WHAT VARIATIONAL LOSS PARAMTER TO USE IN TRAINING!!!!!! STOP!!!!')
            print('')
            
        # Create all of the Callbacks
        vae_sig_schedule = partial(sigmoid_schedule, slope=anneal_sigmod_slope,   start=anneal_epoch_start)
        vae_anneal_callback = WeightAnnealer_epoch(vae_sig_schedule, kl_loss_var, kl_loss_weight, 'kl_loss_var' )
        csv_clb = tf.keras.callbacks.CSVLogger(self.params['model_dir']+self.params['model_name']+'_hist.csv', append=True)
        nan_train = tf.keras.callbacks.TerminateOnNaN()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.params['model_dir']+"Training_model_best_CV.h5",     verbose=1,save_best_only=True)
        
        print('')
        if(training_fho):
            print('Training higher order fragments connecitivty VAE (FHO,Zc)')
        else:
            print('Training first order fragment VAE (F1, Zf)')
        print('Model name : '+self.params['model_name'])
        print('Experiment number : '+str(self.params['exp'] ) )
        
        for epoch in range(epoch_start,self.params['max_epoch']):
            
            '''
            Generator currently can sometimes not be thread safe. It is currently
            unclear why this is the case. A safe working alternative was found 
            to reload/reset generators.
            '''
                    
                    
            if(training_fho):
                g = fho_data_generator(self.params,libExamples_train)
                CV_g = fho_data_generator(self.params, libExamples_CV)
            else:
                g = f1_data_generator(self.params,libExamples_train)
                CV_g = f1_data_generator(self.params,libExamples_CV) 
                
            hist = model.fit_generator(g, steps_per_epoch=steps_per_epoch, epochs=epoch+1, verbose=int(1),   initial_epoch=epoch,
                                       callbacks = [cp_callback,vae_anneal_callback,csv_clb,nan_train],
                                       validation_data = CV_g, validation_steps = 10,
                                       use_multiprocessing=False)
            

            if(training_fho):
                model.get_layer('FHO_encoder').save(self.params['model_dir']+'FHO_encoder.h5')
                model.get_layer('FHO_decoder').save(self.params['model_dir']+'FHO_decoder.h5') 
                
                model.get_layer('FHO_encoder').save(self.params['model_dir']+'FHO_encoder_epoch_'+str(epoch)+'.h5')
                model.get_layer('FHO_decoder').save(self.params['model_dir']+'FHO_decoder_epoch_'+str(epoch)+'.h5') 
                self.params['FHO_epoch_start'] = epoch


            else:
                model.get_layer('F1_Encoder').save(self.params['model_dir']+'F1_Encoder.h5')
                model.get_layer('F1_Encoder').save(self.params['model_dir']+'F1_Encoder_epoch_'+str(epoch)+'.h5')
                
                model.get_layer('Atom_decoder').save(self.params['model_dir']+'F1_Atom_Decoder.h5')
                model.get_layer('Atom_decoder').save(self.params['model_dir']+'F1_Atom_Decoder_epoch_'+str(epoch)+'.h5')

                for degree_idx in range(0,self.params['max_degree'] ):
                    model.get_layer('Decoder_degree_'+str(degree_idx)).save(self.params['model_dir']+'F1_NN_decoder_degree_'+str(degree_idx)+'.h5')
                    model.get_layer('Decoder_degree_'+str(degree_idx)).save(self.params['model_dir']+'F1_NN_decoder_degree_'+str(degree_idx)+'_epoch_'+str(epoch)+'.h5') 
                self.params['F1_epoch_start'] = epoch
                
            save_params(self.params, self.params['model_dir']+'exp.json')
        
    
    


            
    def autoencode_smile(self,smile, display_process = True, trial_walkthrough = False ):
        '''
        Example function how to decode and encode molecular graph
        Decoding is problasitic and dregree of 
        '''
        
        #createing a reference molecular for 
        if(trial_walkthrough):
            mol_recon =  Chem.MolFromSmiles(smiles)
        else:
            mol_recon = -1
           
        atoms, edges, bonds =  smile_to_tensor(smile, self.params,self.params['include_rings'])
      
        atoms = np.array([atoms])
        edges = np.array([edges])
        bonds = np.array([bonds])
        Z_1 , Z_HO,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS = self.Z_encoder(atoms, bonds ,edges)
        mol = self.Z_decoder(Z_1,Z_HO,models, params,display_process = display_process, mol_recon=mol_recon, trial_walkthrough= trial_walkthrough, max_P = self.params['FHO_max_P'])
    
        
        return mol
    
    
    def Z_encoder(self,atoms, bonds ,edges):
        '''
        Encodes complete moleclar graph using FraGVAE
        
        '''
        
        
        # defines the maximum radius of ciruclar message symmetry
        MSA = np.ones((len(atoms),self.params['Num_Graph_Convolutions'],self.params['max_dangle_atoms'],1))
        MSB = np.ones((len(atoms),self.params['Num_Graph_Convolutions'],self.params['max_dangle_atoms']*self.params['max_degree'],1))
        Dangling_atoms = copy.deepcopy(atoms[:,:,0])*0
        
        
        
        bonds_with_DB = np.concatenate((bonds[:,:,:,0:params['num_bond_features']], np.zeros_like(bonds[:,:,:,0:1]),bonds[:,:,:,len(bonds[0,0,0])-1:len(bonds[0,0,0])]), axis=-1)
        ZHO,ZHO_error,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS = self.FHO_encoder.predict([atoms,Dangling_atoms,bonds_with_DB,edges,MSA,MSB])
        
        atoms = atoms[:,0:self.params['max_atoms'],:]
        edges = edges[:,0:self.params['max_atoms'],:]
        bonds = bonds[:,0:self.params['max_atoms'],:,:]
        bonds = bonds[:,:,:,0:len(self.params['bonds'])]
        Z1_set, Z1_error = self.F1_encoder.predict([atoms,bonds,edges])
        
        Z1 = np.sum(Z1_set,axis=-2)
        return Z1 , ZHO,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS

  
    
    def sample_FHO_decoder(self,inputs_decoder,samples = 1):
        '''
         Samples FHO decoder with dropout
         
        '''
        
        result = np.zeros((samples,) + (inputs_decoder[0].shape[0], 2) )
        
        X = tuple(inputs_decoder+[1])

        for i in range(samples):
            result[i,:, :] = self.FHO_decoder_fun(X)[0]
        
        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty
                
        
        
    
    def Z_decoder(self,Z_1,Z_HO, display_process = True, mol_recon=-1, trial_walkthrough= False, max_P = True):
        
        
        NN_Tensor_sampled, atoms_sampled, BagOfF1 = self.decode_F1(Z_1,display_construct = display_process)
        if(display_process):
            print('')
        
        # generate FHO encoding of complete bag of fragments
        max_batch = int(self.params['max_atoms']/(self.params['max_degree']+1)-1)
        if(True):
            edmol = Chem.EditableMol(Chem.MolFromSmiles(''))
            map_edmol_Idx_to_mol = np.ones(self.params['max_dangle_atoms'],dtype = int)*-1
            edmol_Idx_to_mol = 0
            Z1_frags = np.zeros((1,self.params['FHO_finger_print']))
            
            list_BagOfF1 = np.zeros(len(BagOfF1))
            for i in range(0,len(BagOfF1)):
                list_BagOfF1[i]=int(BagOfF1[i])

        for i in range(0,self.params['max_atoms']):
            current_f1 = np.argmax(list_BagOfF1)
            if(list_BagOfF1[current_f1]>0):
                atom = atom_from_atom_feature(int(np.argmax(atoms_sampled[current_f1])),self.params)
                edmol.AddAtom(atom)
                map_edmol_Idx_to_mol[edmol_Idx_to_mol] = current_f1
                edmol_Idx_to_mol = edmol_Idx_to_mol + 1
                list_BagOfF1[current_f1]=list_BagOfF1[current_f1]-1
                #If crashes too many times
                if(edmol_Idx_to_mol>=max_batch or np.max(list_BagOfF1)==0):
                    atoms, edges,bonds,dangling_atoms,MSA,MSB = gen_train_mol_tensor(edmol,NN_Tensor_sampled,map_edmol_Idx_to_mol,self.params,display_dangle=False)
                    '''
                    for trouble shooting return fp from gen_train_mol_tensor to see all of the F1 fragments
                    display(PIL.Image.open(fp))
                    '''
                    temp1,z1_frag_bag,temp2,temp3,temp4 = self.sample_FHO_encoder(np.array([atoms]),np.array([dangling_atoms]),np.array([bonds]),np.array([edges]),np.array([MSA]),np.array([MSB]))
                    Z1_frags = z1_frag_bag + Z1_frags
                    edmol = Chem.EditableMol(Chem.MolFromSmiles(''))
                    map_edmol_Idx_to_mol = np.ones(self.params['max_dangle_atoms'],dtype = int)*-1
                    edmol_Idx_to_mol = 0
    
    
        mol = self.decode_FHO(atoms_sampled,NN_Tensor_sampled,Z_HO,Z1_frags, BagOfF1, display_construct = display_process, mol_recon =mol_recon, trial_walkthrough = trial_walkthrough, max_P = max_P)
        
        return mol
    
    
    def decode_F1(self,Z1,display_construct = True):
        
        
        NN_Tensor = np.array([[[0 for k in range(self.params['num_atom_features']*self.params['num_bond_features']+1)] for i in range(self.params['max_degree'])] for j in range(self.params['max_atoms'])])
        Atoms = np.array([[0 for i in range(self.params['num_atom_features'])] for j in range(self.params['max_atoms'])])
        
        if(display_construct):
            NN_Tensor_with_duplicates  = np.array([[[0 for k in range(self.params['num_atom_features']*self.params['num_bond_features']+1)] for i in range(self.params['max_degree'])] for j in range(self.params['max_atoms'])])
            
            Atoms_with_duplicates = np.array([[0 for i in range(self.params['num_atom_features'])] for j in range(self.params['max_atoms'])])
            single_space = 25
            half_pic_size = (800, 400)
            font_size = 22
            gif_num = 0 
        
        Z1_hist = copy.deepcopy(Z1)*0
        new_F1_idx = 0
        
        BagOfF1 = {}
        for sample_idx in range(0,self.params['max_atoms']):
            sys.stdout.write("\r" + 'F1 Decode: Sampling Fragment '+ str(sample_idx)+self.params['excess_space'])
            sys.stdout.flush()
            select_node,NN_Tensor_slice_select,AI_N,NN_Tensor_slice_prob = self.F1_decoder.predict([Z1-Z1_hist, Z1_hist ])
            node_feature = int(np.argmax(select_node[0]))
            
            if(node_feature!=0):
                node_feature =node_feature-1
                
                add_structure = True
                for F1_check in range(0,sample_idx):
                    if(node_feature == int( np.argmax(Atoms[F1_check])) and all(np.sum( NN_Tensor[F1_check],axis=0)==np.sum( NN_Tensor_slice_select[0],axis=0))):
                        add_structure = False
                        BagOfF1[F1_check] = BagOfF1[F1_check] + 1
                        break
                if(add_structure):
                    Atoms[new_F1_idx,node_feature] =1
                    NN_Tensor[new_F1_idx,:,:] = NN_Tensor_slice_select[0]
                    BagOfF1[new_F1_idx] = 1
                    new_F1_idx= new_F1_idx+1
                    
                
                if(display_construct):
                    Atoms_with_duplicates[sample_idx,node_feature] =1
                    gif_num,given = plot_N_selection(select_node,AI_N,Atoms_with_duplicates,NN_Tensor_with_duplicates,half_pic_size,single_space,gif_num,font_size,self.params)
                    for degree_idx in range(0,self.params['max_degree']):
                        if(degree_idx == 0 or np.argmax(NN_Tensor_slice_select[0][degree_idx-1])!=0 ):
                            NN_Tensor_with_duplicates_next = copy.deepcopy(NN_Tensor_with_duplicates)
                            NN_Tensor_with_duplicates_next[sample_idx,degree_idx,0] = 0
                            NN_Tensor_with_duplicates_next[sample_idx,degree_idx,int(np.argmax(NN_Tensor_slice_select[0][degree_idx]))] =1
                            
                            gif_num,given = plot_NN_selection(NN_Tensor_slice_select[0:1,degree_idx], NN_Tensor_slice_prob[0:1,degree_idx], Atoms_with_duplicates, NN_Tensor_with_duplicates,NN_Tensor_with_duplicates_next, half_pic_size, single_space, gif_num, font_size,given, degree_idx,self.params)
                            NN_Tensor_with_duplicates = NN_Tensor_with_duplicates_next                
                    
                
                    
                
                if(not(self.params['sample_with_replacement'])):
                    edmol = Chem.EditableMol(Chem.MolFromSmiles(''))
                    edmol.AddAtom(atom_from_atom_feature(node_feature,self.params))
                    
                    for degree_idx in range(0,self.params['max_degree']):
                        
                        if(not(self.params['sample_with_replacement'])):
                            if(np.argmax(NN_Tensor_slice_select[0][degree_idx])!=0):
                                NN_sparse_feature = np.argmax(NN_Tensor_slice_select[0][degree_idx])
                                atom_add, bond_add = atom_bond_from_sparse_NN_feature(NN_sparse_feature-1,self.params)
                                edmol.AddAtom(atom_add)
                                edmol.AddBond(0,degree_idx+1,bond_add)
                                
                    if(not(self.params['sample_with_replacement'])):
                        F1_ex_atoms, F1_ex_edge, F1_ex_bonds = mol_to_tensor(edmol.GetMol(),self.params)
                        
                        F1_ex_atoms =np.array([F1_ex_atoms])
                        F1_ex_edge =np.array([F1_ex_edge])
                        F1_ex_bonds =np.array([F1_ex_bonds])
                        
                        F1_ex_atoms = F1_ex_atoms[:,0:self.params['max_atoms'],:]
                        F1_ex_edge = F1_ex_edge[:,0:self.params['max_atoms'],:]
                        F1_ex_bonds = F1_ex_bonds[:,0:self.params['max_atoms'],:,:]
                        F1_ex_bonds = F1_ex_bonds[:,:,:,0:len(self.params['bonds'])]
                        F1_set,total_z_error = self.F1_encoder.predict([F1_ex_atoms, F1_ex_bonds, F1_ex_edge])
                        Z1_hist = Z1_hist + F1_set[0][0]
                
                    
            
        return NN_Tensor, Atoms, BagOfF1
    
    def Z_encoder(self,atoms, bonds ,edges):
        '''
        Encodes fragnments bag and fragment connectivity
        
        '''
        
    
        MSA = np.ones((len(atoms),self.params['Num_Graph_Convolutions'],self.params['max_dangle_atoms'],1))
        MSB = np.ones((len(atoms),self.params['Num_Graph_Convolutions'],self.params['max_dangle_atoms']*self.params['max_degree'],1))
        Dangling_atoms = copy.deepcopy(atoms[:,:,0])*0
        
        
        
        bonds_with_DB = np.concatenate((bonds[:,:,:,0:self.params['num_bond_features']], np.zeros_like(bonds[:,:,:,0:1]),bonds[:,:,:,len(bonds[0,0,0])-1:len(bonds[0,0,0])]), axis=-1)
        ZHO,ZHO_error,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS = self.FHO_encoder.predict([atoms,Dangling_atoms,bonds_with_DB,edges,MSA,MSB])
        
        atoms = atoms[:,0:self.params['max_atoms'],:]
        edges = edges[:,0:self.params['max_atoms'],:]
        bonds = bonds[:,0:self.params['max_atoms'],:,:]
        bonds = bonds[:,:,:,0:len(self.params['bonds'])]
        Z1_set, Z1_error = self.F1_encoder.predict([atoms,bonds,edges])
        
        Z1 = np.sum(Z1_set,axis=-2)
        return Z1 , ZHO,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS
    
    def sample_FHO_encoder(self,atoms,Dangling_atoms,bonds,edge,MSA,MSB):
       
        z,z_error,z1,z2,zR,zS = self.FHO_encoder.predict([atoms,Dangling_atoms, bonds, edge,MSA,MSB])
        return z,z1,z2,zR,zS
    
    def reset_weights(self):
        '''
        Reset all models weights
        '''
        def reset_model(model):
            weights =model.get_weights()
            new_weights = []
            for weight in weights: 
                if(len(weight.shape)>1):
                    limit= np.sqrt(6/(weight.shape[1]+weight.shape[0]))
                    new_weights.append(np.random.uniform(-limit,limit,(weight.shape[0],weight.shape[1])))
                elif(len(weight.shape)==1):
                    limit= np.sqrt(6/(weight.shape[0]))
                    new_weights.append(np.random.uniform(-limit,limit,weight.shape[0]))
                else:
                    limit= np.sqrt(6/(weight))
                    new_weights.append(np.random.uniform(-limit,limit,(1,))[0])
            model.set_weights(new_weights)
            return model   
        self.F1_encoder = reset_model(self.F1_encoder)
        
        
        self.atom_decoder = reset_model(self.atom_decoder )
        
        for degree_idx in range(0,params['max_degree'] ):
            self.NN_decoder[degree_idx] = reset_model(self.NN_decoder[degree_idx])
        
        self.F1_decoder = reset_model(self.F1_decoder)
        self.FHO_decoder = reset_model(self.FHO_decoder)
        self.FHO_encoder = reset_model(self.FHO_encoder)



    def decode_FHO(self, Atoms,NN_Tensor,Z_HO,Frag_z1,BagOfF1, display_construct = True,  mol_recon =-1, trial_walkthrough = False, max_P = True,save_bond_hist=False):
        '''
        Prodeccure for reconstructing the complete graph using tesnsorflow models and bag of fragments.
        Process can generate GIFs to determine point of failure.
        '''
        
        #randomizing the nucleating fragment 
        nucleation_F1 = int(np.random.choice(list(BagOfF1)))
        
        if(display_construct):
            font_size = 13
            single_space = 15
            full_sixth_pic_size = (320, 270)
            mol_pic_size = (full_sixth_pic_size[0], full_sixth_pic_size[1]-single_space*2)
            num_F1  = np.sum((np.sum(Atoms,axis=-1)>0)*1) 
            
            
            
        if(trial_walkthrough):
            recon_atoms, recon_edges,recon_bonds =mol_to_tensor(mol_recon,self.params)
            NN_Tensor_summed_recon =gen_sparse_NN_Tensor(recon_atoms,recon_edges,recon_bonds)
            NN_Tensor_summed_recon = np.sum(NN_Tensor_summed_recon,axis=-2)
            NN_Tensor_summed_recon[:,0] = 0
            
            if(save_bond_hist):
                max_non_valid_Z_Score =np.array(mol_recon.GetNumBonds())
                max_valid_Z_Score =np.array(mol_recon.GetNumBonds())
        NN_Tensor_summed = np.sum(NN_Tensor,axis=-2)
        NN_Tensor_summed[:,0] = 0
        decoding_mol = True
        
        
        edmol = Chem.EditableMol(Chem.MolFromSmiles(''))
        
        map_edmol_Idx_to_mol = np.ones(self.params['max_dangle_atoms'],dtype = int)*-1
        map_edmol_Idx_to_mol[0] = nucleation_F1
        cur_edmol_Idx = 0
        
        edmol.AddAtom(atom_from_atom_feature(int(np.argmax(Atoms[nucleation_F1])),self.params))
        
    
        test_Tanimoto_bag = []
        test_Tanimoto_idx = []
        
        best_Tanimoto_bag = []
        best_Tanimoto_idx = []
        
        iteration_idx = -1
        best_Tanimoto =0
        gif_num = 0
        while_build_safety =0
        reduce_F1_bag = -1
        BagOfF1[nucleation_F1] =BagOfF1[nucleation_F1] -1
        bond_idx = 0
        while decoding_mol:
    
            best_z_test = - np.inf
            test_hist = []
            best_prediction = np.array([0.5,0.5])
            best_uncertainty= np.array([100.0,100.0])
            iteration_idx = iteration_idx + 1
            best_Tanimoto_idx.append(iteration_idx)
            best_Tanimoto_bag.append(best_Tanimoto)
            if(display_construct):
                img = PIL.Image.new('RGB', mol_pic_size, color = 'white')
                best_Frag_img_fp = BytesIO()
                img.save( best_Frag_img_fp,'PNG' )
            
            cur_atoms, cur_edges,cur_bonds,cur_dangling_atoms,cur_MSA,cur_MSB =gen_train_mol_tensor(edmol,NN_Tensor,map_edmol_Idx_to_mol,self.params)
            cur_atoms = np.array([cur_atoms])
            cur_edges = np.array([cur_edges])
            cur_bonds = np.array([cur_bonds])
            cur_MSA = np.array([cur_MSA])
            cur_MSB = np.array([cur_MSB])
            cur_dangling_atoms = np.array([cur_dangling_atoms])
            z_edmol_previous,z1_edmol_previous,z2_edmol_previous,zR_edmol_previous,zS_edmol_previous = self.sample_FHO_encoder(cur_atoms,cur_dangling_atoms,cur_bonds,cur_edges,cur_MSA,cur_MSB)
            
    
            atoms_edmol,edges_edmol,bonds_edmol = mol_to_tensor(edmol.GetMol(),self.params,FHO_Ring_feature=True)
            NN_Tensor_edmol = gen_sparse_NN_Tensor(atoms_edmol,edges_edmol,bonds_edmol[:,:,0:len(self.params['bonds'])])
            
            NN_Tensor_edmol = np.sum(NN_Tensor_edmol,axis=-2)
            NN_Tensor_edmol[:,0]=0
            
            NN_Tensor_edmol_DB =  copy.deepcopy(NN_Tensor_edmol)*0
            
            
            # update NN_Tensor_edmol_DB
            
            for  atom_idx in range(0,len(map_edmol_Idx_to_mol)):
                if(map_edmol_Idx_to_mol[atom_idx]!=-1):
                    NN_Tensor_edmol_DB[atom_idx] = copy.deepcopy(NN_Tensor_summed[map_edmol_Idx_to_mol[atom_idx]])
            NN_Tensor_edmol_DB[:,0]=0
            
            NN_Tensor_edmol_DB = NN_Tensor_edmol_DB-NN_Tensor_edmol
            solve_DB = np.sum(np.sum(NN_Tensor_edmol_DB)) ==0
            for  atom_idx in range(0,len(map_edmol_Idx_to_mol)):
                
                if(map_edmol_Idx_to_mol[atom_idx]!=-1 and not(solve_DB)):
                    
                    
                    # check if has dangling bonds
                    if(np.argmax(NN_Tensor_edmol_DB[atom_idx])!=0):
                        
                        atom_root = Atoms[map_edmol_Idx_to_mol[atom_idx]] 
                        
                        
                        #go through danling bonds 
                        DB_feature_idx = np.argmax(NN_Tensor_edmol_DB[atom_idx])
                        while_safety = 0
                        while(DB_feature_idx !=0 and while_safety <= self.params['max_degree']):
                            if(display_construct):
                                cur_atoms, cur_edges,cur_bonds,cur_dangling_atoms,cur_img_fp,cur_MSA,cur_MSB =gen_train_mol_tensor(edmol,NN_Tensor,map_edmol_Idx_to_mol,self.params,cur_edmol_Idx =cur_edmol_Idx,edmol_root_idx = atom_idx,display_dangle=display_construct,updated_atom=False,leaf_sparse_NN_feature=DB_feature_idx-1,molSize=mol_pic_size)
    
                            while_safety = while_safety + 1
                            #print('marco')
                            leaf_atom, cur_bond = atom_bond_from_sparse_NN_feature(int(DB_feature_idx-1),self.params)                                            
                            requred_frag_connection = int(np.argmax(gen_sparse_NN_feature(atom_root,np.array([cur_bond ==temp_bond for temp_bond in self.params['bonds']])*1) ))                       
                            
                            # check through NN_Tensor to find fragments with viable connceitons
                            for atom_idx2 in range(0,len(NN_Tensor_summed)):
                                if(all(Atoms[atom_idx2] == gen_atom_features(leaf_atom,self.params)) and NN_Tensor_summed[atom_idx2,requred_frag_connection+1]>0 and not(self.params['no_replacement'] and BagOfF1[int(atom_idx2)] <=0 and best_Tanimoto<=0.9999)):
                                    sys.stdout.write("\r" + 'FHO Decode: Adding Frag, Total Atoms: ' +str(cur_edmol_Idx)+' Total Bonds: '+str(edmol.GetMol().GetNumBonds())+self.params['excess_space'])
                                    sys.stdout.flush()
                                    if(display_construct):
                                        display_F1= np.ones(len(NN_Tensor))
                                        flash_F1 = np.zeros(len(NN_Tensor))
                                        flash_F1[atom_idx2] =1
                                        Frag_collect = display_F1_graphs(self.params,num_F1,display_F1,flash_F1,mol_pic_size,Atoms, NN_Tensor)
                                        Frag_img_fp = BytesIO()
                                        Frag_collect.save( Frag_img_fp,'PNG' )
                                    
                                    test_edmol = copy.deepcopy(edmol.GetMol())
                                    test_edmol = Chem.EditableMol(test_edmol)
                                    test_map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol)
                                    test_map_edmol_Idx_to_mol[cur_edmol_Idx+1] = atom_idx2
                                    
                                    not_print = test_edmol.AddAtom(leaf_atom)
                                    not_print = test_edmol.AddBond(atom_idx,cur_edmol_Idx+1,cur_bond)
                                    if(display_construct):
                                        test_atoms, test_edges,test_bonds,test_dangling_atoms,test_img_fp,test_MSA,test_MSB =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,self.params,cur_edmol_Idx =cur_edmol_Idx+1,edmol_root_idx =atom_idx,display_dangle=display_construct,updated_atom=True,leaf_sparse_NN_feature=DB_feature_idx-1,molSize=mol_pic_size)
                                    else:
                                        test_atoms, test_edges,test_bonds,test_dangling_atoms,test_MSA,test_MSB =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,self.params,cur_edmol_Idx =cur_edmol_Idx+1,edmol_root_idx =atom_idx,display_dangle=display_construct,updated_atom=True,leaf_sparse_NN_feature=DB_feature_idx-1)
                                    test_atoms = np.array([test_atoms])
                                    test_edges = np.array([test_edges])
                                    test_bonds = np.array([test_bonds])
                                    test_MSA = np.array([test_MSA])
                                    test_MSB = np.array([test_MSB])
                                    test_dangling_atoms = np.array([test_dangling_atoms])
                                    
                                    z_edmol_update,z1_edmol_update,z2_edmol_update,zR_edmol_update,zS_edmol_update = self.sample_FHO_encoder(test_atoms,test_dangling_atoms,test_bonds,test_edges,test_MSA,test_MSB)
                                    
                                    inputs_decoder = [Z_HO,Frag_z1,z_edmol_previous,z1_edmol_previous,z2_edmol_previous,zR_edmol_previous,zS_edmol_previous,z_edmol_update,z1_edmol_update,z2_edmol_update,zR_edmol_update,zS_edmol_update]
    
                                    test_prediction, test_uncertainty =  self.sample_FHO_decoder(inputs_decoder,samples = 5)
                                    test_prediction = test_prediction[0]
                                    test_uncertainty= test_uncertainty[0]
                                    
                                    
                                    z_test = (test_prediction[0]-test_prediction[1])/np.sqrt(np.abs(test_uncertainty[0]**2+ test_uncertainty[1]**2))
    
                                    if(self.params['no_replacement'] and BagOfF1[int(atom_idx2)] <=0):
                                        z_test = z_test 
                                    # P_z_test = (np.tanh(z_test)+1)/2
                                    
                                    #expect_cat_1 = test_prediction[0][0]*P_z_test
                                    
                                    test_Tanimoto = self.calc_Tanimoto(Z_HO,z_edmol_update)
                                    test_Tanimoto_bag.append(test_Tanimoto)
                                    test_Tanimoto_idx.append(iteration_idx)
                                    
                                    
                                    if(display_construct):
                                        plot_Normal_fp = plot_Normal(test_prediction, test_uncertainty, best_prediction, best_uncertainty, z_test,best_z_test,full_sixth_pic_size)
                                        plot_Tanimoto_fp = plot_Tanimoto(test_Tanimoto_bag,test_Tanimoto_idx,best_Tanimoto_bag,best_Tanimoto_idx,full_sixth_pic_size)
                                        display_img_fp = [cur_img_fp,best_Frag_img_fp,plot_Normal_fp,Frag_img_fp,test_img_fp,plot_Tanimoto_fp]
                                        gen_FHO_construct_img(display_img_fp,gif_num,best_z_test,z_test,best_Tanimoto,test_Tanimoto,trial_walkthrough,0,full_sixth_pic_size,font_size,single_space,mol_pic_size)
                                        gif_num = gif_num+1
                                    
                                    if(not(trial_walkthrough) or ( trial_walkthrough and is_valid_substructure(test_edmol.GetMol(),mol_recon,NN_Tensor_summed,NN_Tensor_summed_recon,test_map_edmol_Idx_to_mol))):
    
                                        
                                        if(not(max_P)):
                                            test_hist.append([0,z_test,atom_idx,atom_idx2,int(np.argmax(np.array([cur_bond ==temp_bond for temp_bond in self.params['bonds']])*1))])
                            
                                        if( z_test > best_z_test ):
                                            best_Tanimoto_bag[iteration_idx] = test_Tanimoto
                                            best_z_test = z_test
                                            best_edmol = Chem.EditableMol(copy.deepcopy(test_edmol.GetMol()))
                                            best_map_edmol_Idx_to_mol = copy.deepcopy(test_map_edmol_Idx_to_mol)
                                            best_cur_edmol_Idx = cur_edmol_Idx+1
                                            best_Tanimoto = test_Tanimoto
                                            best_prediction = test_prediction
                                            best_uncertainty= test_uncertainty
                                            reduce_F1_bag = atom_idx2                                           
                                            if(display_construct):                                            
                                                best_Frag_img_fp = test_img_fp
                                            if(save_bond_hist):
                                                max_valid_Z_Score[bond_idx] = z_test                                        
                                    elif(save_bond_hist):
                                        max_non_valid_Z_Score[bond_idx] = z_test
                                        
                                        
                                    
                            if(display_construct):
                                display_F1= np.ones(len(NN_Tensor))
                                flash_F1 = np.zeros(len(NN_Tensor))
                                
                                Frag_collect = display_F1_graphs(self.params,num_F1,display_F1,flash_F1,mol_pic_size,Atoms, NN_Tensor)
                                Frag_img_fp = BytesIO()
                                Frag_collect.save( Frag_img_fp,'PNG' )
                            
                            
                            for atom_idx2 in range(0,len(map_edmol_Idx_to_mol)):
                                #print('Is map_edmol_Idx_to_mol[atom_idx2]!=-1 and atom_idx !=atom_idx2 ' + str(map_edmol_Idx_to_mol[atom_idx2]!=-1 and atom_idx !=atom_idx2))
                                if(map_edmol_Idx_to_mol[atom_idx2]!=-1 and atom_idx !=atom_idx2):
                                    if(all(Atoms[map_edmol_Idx_to_mol[atom_idx2]] == gen_atom_features(leaf_atom,self.params)) and NN_Tensor_edmol_DB[atom_idx2,requred_frag_connection+1]>0):                                    
                                        if(edmol.GetMol().GetBondBetweenAtoms(atom_idx2, atom_idx) is None):
                                            sys.stdout.write("\r" + 'FHO Decode: Forming Ring, Total Atoms: ' +str(cur_edmol_Idx)+' Total Bonds: '+str(edmol.GetMol().GetNumBonds())+self.params['excess_space'])
                                            sys.stdout.flush()
                                            test_edmol = copy.deepcopy(edmol.GetMol())
                                            test_edmol =Chem.EditableMol(test_edmol)
                                            test_map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol)
                                            test_edmol.AddBond(atom_idx,atom_idx2,cur_bond)
                                            
                                            if(display_construct):
                                                test_atoms, test_edges,test_bonds,test_dangling_atoms,test_img_fp,test_MSA,test_MSB =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,self.params,cur_edmol_Idx=  atom_idx2,edmol_root_idx =atom_idx,display_dangle=display_construct,updated_atom=False,leaf_sparse_NN_feature=DB_feature_idx-1,molSize=mol_pic_size)
                                            else:
                                                test_atoms, test_edges,test_bonds,test_dangling_atoms,test_MSA,test_MSB =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,self.params,cur_edmol_Idx =atom_idx2,edmol_root_idx =atom_idx,display_dangle=display_construct,updated_atom=False,leaf_sparse_NN_feature=DB_feature_idx-1)
                                                                            
                                            test_atoms = np.array([test_atoms])
                                            test_edges = np.array([test_edges])
                                            test_bonds = np.array([test_bonds])
                                            test_MSA = np.array([test_MSA])
                                            test_MSB = np.array([test_MSB])
                                            test_dangling_atoms = np.array([test_dangling_atoms])
    
                                            
                                            z_edmol_update,z1_edmol_update,z2_edmol_update,zR_edmol_update,zS_edmol_update = self.sample_FHO_encoder(test_atoms,test_dangling_atoms,test_bonds,test_edges,test_MSA,test_MSB)
    
                                            inputs_decoder = [Z_HO,Frag_z1,z_edmol_previous,z1_edmol_previous,z2_edmol_previous,zR_edmol_previous,zS_edmol_previous,z_edmol_update,z1_edmol_update,z2_edmol_update,zR_edmol_update,zS_edmol_update]
                                            
                                            test_prediction, test_uncertainty =  self.sample_FHO_decoder(inputs_decoder,samples = 12)
                                            test_prediction = test_prediction[0]
                                            test_uncertainty = test_uncertainty[0]
                                            
                                                                                    
                                            z_test = (test_prediction[0]-test_prediction[1])/np.sqrt(np.abs(test_uncertainty[0]**2+ test_uncertainty[1]**2))
                                            test_Tanimoto = self.calc_Tanimoto(Z_HO,z_edmol_update)
                                            test_Tanimoto_bag.append(test_Tanimoto)
                                            test_Tanimoto_idx.append(iteration_idx)                                           
            
                                            
                                            if(display_construct):
                                                plot_Normal_fp = plot_Normal(test_prediction, test_uncertainty, best_prediction, best_uncertainty, z_test,best_z_test,full_sixth_pic_size)
                                                plot_Tanimoto_fp = plot_Tanimoto(test_Tanimoto_bag,test_Tanimoto_idx,best_Tanimoto_bag,best_Tanimoto_idx,full_sixth_pic_size)
                                                display_img_fp = [cur_img_fp,best_Frag_img_fp,plot_Normal_fp,Frag_img_fp,test_img_fp,plot_Tanimoto_fp]
                                                gen_FHO_construct_img(display_img_fp,gif_num,best_z_test,z_test,best_Tanimoto,test_Tanimoto,trial_walkthrough,1,full_sixth_pic_size,font_size,single_space,mol_pic_size)
                                                gif_num = gif_num+1
                                            
                                            if(not(trial_walkthrough) or ( trial_walkthrough and is_valid_substructure(test_edmol.GetMol(),mol_recon,NN_Tensor_summed,NN_Tensor_summed_recon,test_map_edmol_Idx_to_mol))):
                                                
                                                if(not(max_P)):
                                                    test_hist.append([1,z_test,atom_idx,atom_idx2,int(np.argmax(np.array([cur_bond ==temp_bond for temp_bond in self.params['bonds']])*1))])
                                                    
                                                if( z_test > best_z_test ):
                                                    best_z_test = z_test
                                                    best_Tanimoto_bag[iteration_idx] = test_Tanimoto
                                                    best_edmol = Chem.EditableMol(copy.deepcopy(test_edmol.GetMol()))
                                                    best_map_edmol_Idx_to_mol = copy.deepcopy(test_map_edmol_Idx_to_mol)
                                                    best_cur_edmol_Idx = cur_edmol_Idx
                                                    best_Tanimoto = test_Tanimoto
                                                    best_prediction =test_prediction
                                                    best_uncertainty= test_uncertainty
                                                    reduce_F1_bag = -1
                                                    if(display_construct):
                                                        
                                                        best_Frag_img_fp = test_img_fp
                                                    if(save_bond_hist):
                                                        max_valid_Z_Score[bond_idx] = z_test                                        
                                            elif(save_bond_hist):
                                                max_non_valid_Z_Score[bond_idx] = z_test
                            
                            
                            NN_Tensor_edmol_DB[atom_idx,DB_feature_idx] =0
                            DB_feature_idx = int(np.argmax(NN_Tensor_edmol_DB[atom_idx]))
            if(not(max_P)):
                test_hist.append([1,z_test,atom_idx,atom_idx2,int(np.argmax(np.array([cur_bond ==temp_bond for temp_bond in self.params['bonds']])*1))])
                test_hist = np.array(test_hist)
                approx_P = (np.tanh(test_hist[:,1] - np.max(test_hist[:,1]))+1)/2
                rnd_connection = np.random.choice(len(approx_P), 1, p=approx_P)[0]
                if(test_hist[rnd_connection,0]==1):
                    edmol.AddBond(test_hist[rnd_connection,2],test_hist[rnd_connection,3],bond_from_bond_feature(int(test_hist[rnd_connection,4]),self.params))
                else:              
                    edmol.AddAtom(atom_from_atom_feature(int(np.argmax(Atoms[test_hist[rnd_connection,3]]))),self.params)
                    edmol.AddBond(test_hist[rnd_connection,2],cur_edmol_Idx+1,bond_from_bond_feature(int(test_hist[rnd_connection,4]),self.params))
                    map_edmol_Idx_to_mol[cur_edmol_Idx+1] = test_hist[rnd_connection,3]
                    cur_edmol_Idx=cur_edmol_Idx+1
                    BagOfF1[rnd_connection] = BagOfF1[rnd_connection] -1
            else:
                try:
                    edmol = best_edmol
                    map_edmol_Idx_to_mol = best_map_edmol_Idx_to_mol
                    cur_edmol_Idx = best_cur_edmol_Idx 
                    if(reduce_F1_bag!=-1):
                        BagOfF1[reduce_F1_bag] = BagOfF1[reduce_F1_bag] -1
                except:
                    mol = edmol.GetMol()
                    decoding_mol = False
                        
            bond_idx = bond_idx + 1
            while_build_safety =while_build_safety+1
            if(self.params['max_atoms']*0.7<=edmol.GetMol().GetNumAtoms() or solve_DB or while_build_safety>=self.params['max_bonds']):
                decoding_mol = False
                mol = best_edmol.GetMol()
                
            
        if(save_bond_hist):
            diff_Z_score = max_valid_Z_Score-max_non_valid_Z_Score
            
            matches = mol.GetSubstructMatches(mol_recon,uniquify=False)
            
            diff_Z_score_recon = np.ones(diff_Z_score)*np.inf
            for match in matches:
                for bond_idx in range(0,mol_recon.GetNumBonds()):
                    bond = mol_recon.GetBondWithIdx(bond_idx)
                    
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    mol_begin_idx = match[begin_idx]
                    mol_end_idx = match[end_idx]
                    mol_bond_idx = mol.GetBondBetweenAtoms(mol_begin_idx,mol_end_idx)
                    
                    diff_Z_score_recon[bond_idx] = min(diff_Z_score_recon[bond_idx],diff_Z_score[mol_bond_idx])
            
            return mol,diff_Z_score_recon
        else:   
            return mol 


    def gen_dropout_fun(self,model):
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
    
    def calc_Tanimoto(self,z1,z2):
    
        z1 = z1[0]
        z2 = z2[0]
        
        if(self.params['FHO_decoder_activation']=='elu'or self.params['FHO_decoder_activation']=='tanh'):
            z1 = np.maximum(z1 + np.ones_like(z1),np.zeros_like(z1))
            z2 = np.maximum((z2 + np.ones_like(z2)),np.zeros_like(z2))
        
        
    
        tanimoto_max = np.maximum(z1,z2)
        tanimoto_min = np.minimum(z1,z2)
        
        tanimoto_max = np.sum(tanimoto_max,axis=-1)
        tanimoto_min = np.sum(tanimoto_min,axis=-1)
        
        tanimoto_val = tanimoto_min/(tanimoto_max+1E-20)
        
        return tanimoto_val


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"
Addtional hidden funcutions to support FraGVAE

"'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def plot_Normal(test_prediction, test_uncertainty, best_prediction, best_uncertainty, test_z_score,best_z_score,pic_size):
    
    x = (np.array(range(0,220))-10)/200
    
    def norm_fun(std, mean, x):
        norm = 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mean)**2/2/std**2)
        return norm
    
    plt.figure(figsize=(pic_size[0]/100,pic_size[1]/100), dpi=100)
    plt.rcParams.update({'font.size': 10})
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    
    root_atom_color = (139/250,69/250,19/250)
    #if(valid_connection):
    root_bond_color = (205/250,133/250,63/250)
    
    lighter_red = (255/255,51/255,51/255)
    dark_green = (0,100/255,0)
    
    
    labels = ['','Current best      ', '     New proposed','']
    
    if(best_z_score!=-np.inf):
        y = np.array([best_prediction[0]-best_prediction[1],test_prediction[0]-test_prediction[1]])
        yerr = [np.sqrt(best_uncertainty[0]**2+best_uncertainty[1]**2),np.sqrt(test_uncertainty[0]**2+test_uncertainty[1]**2)]
    else:
        y = np.array([-1,test_prediction[0]-test_prediction[1]])
        yerr = [0,np.sqrt(test_uncertainty[0]**2+test_uncertainty[1]**2)]
        
    x = [0,1] 
    plt.bar(x, y*((y>=0)*1.0),yerr=yerr*((y>=0)*1.0), label='+ score',color ='c',align='center',alpha = 0.25,ecolor ='c',capsize =23)
    plt.bar(x, y*((y<0)*1.0),yerr=yerr*((y<0)*1.0),  label='- score',color ='r',align='center',alpha = 0.25,ecolor ='r',capsize =23)
    plt.bar(x, [0,0],yerr=[0,0],  color ='k',align='center',alpha = 0.5,ecolor ='k',capsize =100)

    #plt.bar(x, y*((y>=0)*1.0),yerr*((y>=0)*1.0), label='+ score',color ='c',align='center',alpha = 0.5)
    #plt.bar(x, y*((y<0)*1.0),yerr*((y<0)*1.0),  label='- score',color ='r',align='center',alpha = 0.5)
    axes.set_xticks([-1]+x+[2])
    axes.set_xticklabels(labels)

    
    #plt.plot(x, norm_fun(best_uncertainty[0], best_prediction[0], x),'--',alpha = 0.5, color = dark_green,label = 'High Prob. Valid')
    #plt.plot(x, norm_fun(best_uncertainty[1], best_prediction[1], x),'--',alpha = 0.5, color = lighter_red,label = 'High Prob. Not Valid')
    

    #plt.plot(x, norm_fun(test_uncertainty[0], test_prediction[0], x),alpha = 0.5, color = dark_green,label= 'Proposed Valid')
    #plt.plot(x, norm_fun(test_uncertainty[1], test_prediction[1], x),alpha = 0.5, color = lighter_red,label= 'Proposed Not Valid')
    
    #plt.ylabel('Probability of categorical regression',wrap=True)
    #plt.xlabel('Categorical regression values',wrap=True)

    plt.title('Proposed fragment rankings',wrap=True)

    axes.set_ylim([-1,1])
    
    #axes.set_xlim([0,1])
    #plt.legend()
    plt.tight_layout()

    P_F1_N = BytesIO()
    plt.savefig(P_F1_N, dpi=100)
    plt.close()
    
    return P_F1_N





def plot_Tanimoto(test_Tanimoto_bag,test_Tanimoto_idx,best_Tanimoto_bag,best_Tanimoto_idx,pic_size):
    

    
    plt.figure(figsize=(pic_size[0]/100,pic_size[1]/100), dpi=100)
    plt.rcParams.update({'font.size': 10})
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    
    root_atom_color = (139/250,69/250,19/250)
    #if(valid_connection):
    root_bond_color = (205/250,133/250,63/250)

                
    plt.plot(test_Tanimoto_idx, test_Tanimoto_bag, 'o',alpha = 1,color = (0,1,1),label ='All')
    
    plt.plot(best_Tanimoto_idx, best_Tanimoto_bag,'o',alpha = 1,color = root_bond_color)
    plt.plot(best_Tanimoto_idx, best_Tanimoto_bag,alpha = 1,color = root_bond_color)
    plt.plot(best_Tanimoto_idx[len(best_Tanimoto_idx)-1], best_Tanimoto_bag[len(best_Tanimoto_idx)-1],'o',alpha = 1,color = root_atom_color,label='Best')

    plt.plot(test_Tanimoto_idx[len(test_Tanimoto_idx)-1], test_Tanimoto_bag[len(test_Tanimoto_bag)-1],'o',alpha = 1,color =(0,0,0),label='New')

    #plt.ylabel('Tanimoto Index',wrap=True)
    plt.xlabel('Iterations',wrap=True)
    axes.set_ylim([-0.05,1.05])
    	
    
    plt.title('History of Tanimoto index',wrap=True)

    #axes.set_ylim([0,1])
    plt.legend()
    
    P_F1_N = BytesIO()
    plt.savefig(P_F1_N, dpi=100)
    plt.close()
    
    return P_F1_N

def vstack_img(img_top,img_bottom):
    
    imgs_comb = np.vstack( (np.asarray( i ) for i in [img_top,img_bottom] ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb 
 
def gen_FHO_construct_img(display_img_fp,gif_num,best_z_test,z_test,best_Tanimoto,test_Tanimoto,trial_walkthrough,check_ring,full_sixth_pic_size,font_size,single_space,mol_pic_size):
    [cur_img_fp,best_Frag_img_fp,plot_Normal_fp,Frag_img_fp,test_img_fp,plot_Tanimoto_fp] =display_img_fp
    if(best_z_test<-10E10):
        img = PIL.Image.new('RGB', mol_pic_size, color = 'white')
        best_Frag_img_fp = BytesIO()
        img.save( best_Frag_img_fp,'PNG' )
        best_z_test=-np.inf
        best_Tanimoto=0
        
     
    
    display_img   = [ PIL.Image.open(i) for i in display_img_fp ]

    
    text_size = (full_sixth_pic_size[0], single_space*2)
    
    
    
    text_current_structure = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_current_structure)
    if(not(trial_walkthrough)):
        d.text((0,0), "FraGVAE: Molecular reconstruction from fragments", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    else:
        d.text((0,0), "FraGVAE: Trial walkthrough", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
        
    if(check_ring):
        d.text((0,single_space), "Previous best proposed fragment", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    else:
        d.text((0,single_space), "Previous best proposed fragment", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
        
    display_img[0] = vstack_img(text_current_structure,display_img[0])     
    
    
    text_best_of_structure = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_best_of_structure)
    d.text((0,0), "Current best proposed fragment", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), 'Z-score: {:+.2f}, Tanimoto index: {:.3f}'.format( best_z_test, best_Tanimoto ), font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
        
    display_img[1] = vstack_img(text_best_of_structure,display_img[1])  
    
    
    text_best_of_structure = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_best_of_structure)
    d.text((0,0), "Bag of fragments", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    if(check_ring):
        d.text((0,single_space), 'Status: Formation of ring proposed', font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    else:
        d.text((0,single_space), 'Status: Addition of fragment proposed', font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    display_img[3] = vstack_img(text_best_of_structure,display_img[3])  
    
    
    
    
    
    text_best_of_structure = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_best_of_structure)
    d.text((0,0), "New proposed fragment", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))

    d.text((0,single_space), 'Z-score: {:+.2f}, Tanimoto index: {:.3f}'.format( z_test, test_Tanimoto ), font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
        
    display_img[4] = vstack_img(text_best_of_structure,display_img[4])  
    
    
    imgs_comb = [np.asarray( i ) for i in display_img]

    imgs_comb = [np.hstack([imgs_comb[0],imgs_comb[1],imgs_comb[2][:,:,0:3]]),np.hstack([imgs_comb[3],imgs_comb[4],imgs_comb[5][:,:,0:3]])]
    imgs_comb = np.vstack([imgs_comb[0],imgs_comb[1]])

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    
    #if(z_test>-10E10 and (np.random.choice([True, False, False]) or best_z_test<z_test)):
    imgs_comb.save( 'imgs_gifs/imgs_to_gif/'+'FHO_construct_'+str(gif_num).zfill(6)+'.png' )
    
    
    return
    


def plot_N_selection(select_node,AI_N,atoms,NN_Tensor,half_pic_size,single_space,gif_num,font_size,params):
    AI_N = AI_N[0]
    select_node = int(np.argmax(select_node[0]))
    num_F1 = np.sum( np.sum(atoms))

    display_F1 = np.ones(params['max_atoms'])
    display_F1[num_F1-1] = 0
    flash_F1 = display_F1 * 0
    
    bag_frags = display_F1_graphs(params,18,display_F1,flash_F1, (half_pic_size[0],(int(half_pic_size[1]/2)-single_space*2)),atoms,NN_Tensor)
    
    display_F1 = np.ones(params['max_atoms'])
    flash_F1[num_F1-1] = 1
    bag_frags_flash = display_F1_graphs(params,18,display_F1,flash_F1, (half_pic_size[0],(int(half_pic_size[1]/2)-single_space*2)),atoms,NN_Tensor)

    
    text_selected_frags = PIL.Image.new('RGB', (half_pic_size[0],single_space), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_selected_frags)
    d.text((0,0), "Decoded bag of fragments centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    text_selected_frags_flash = PIL.Image.new('RGB', (half_pic_size[0],single_space), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_selected_frags_flash)
    d.text((0,0), "Decoded bag of fragments centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    img_bag_frags = vstack_img(text_selected_frags_flash,bag_frags)    
    img_bag_frags_flash = vstack_img(text_selected_frags_flash,bag_frags_flash)  
    
    
    plt.figure(figsize=(half_pic_size[0]/100,(half_pic_size[1]-single_space*2)/100), dpi=100)
    plt.rcParams.update({'font.size': 13})
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    axes.set_ylim([0,1.2])

                
    
    select_atoms = ['0']+params["atoms"]
    plt.bar(select_atoms, AI_N, width=1, bottom=None,color ='c',label= 'P(Node|Zf , Frag Bag)')
    plt.ylabel('Probability',wrap=True)
    plt.xlabel('Possible center atoms',wrap=True)

    plt.title('Probability of sampling center atom of fragment',wrap=True)


    plt.legend()
    
    P_F1_N = BytesIO()
    plt.savefig(P_F1_N, dpi=100)
    P_F1_N = PIL.Image.open(P_F1_N)
    plt.close()
    
    
    text_plots = PIL.Image.new('RGB', (half_pic_size[0],single_space*2), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_plots)
    d.text((0,0), "FraGVAE: Bag of fragments reconstruction centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), "Status: Probability of sampling a center atom", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    imgs_comb = [np.asarray( i ) for i in [text_plots,P_F1_N,img_bag_frags]]
    imgs_comb = np.vstack([imgs_comb[0],imgs_comb[1][:,:,0:3],imgs_comb[2]])

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    
    
    imgs_comb.save( 'imgs_gifs/imgs_to_gif/'+'F1_frag_'+str(gif_num).zfill(6)+'.png' )
    gif_num = gif_num +1
    
    
    
    plt.figure(figsize=(half_pic_size[0]/100,(half_pic_size[1]-single_space*2)/100), dpi=100)
    plt.rcParams.update({'font.size': 13})
    plt.gcf().subplots_adjust(bottom=0.15)
    axes = plt.gca()
    axes.set_ylim([0,1.2])

                
    AI_N = copy.deepcopy(AI_N)
    P_val = AI_N[select_node]
    flah_AI_N = AI_N*0
    flah_AI_N[select_node] = P_val
    AI_N[select_node] = 0
    select_atoms = ['0']+params["atoms"]
    
    
    plt.bar(select_atoms, AI_N, width=1, bottom=None,color ='c',label= 'P(Node|Zf , Frag Bag)')
    plt.bar(select_atoms, flah_AI_N, width=1, bottom=None,color ='b',label= 'Sampled '+select_atoms[select_node])
    plt.ylabel('Probability',wrap=True)
    plt.xlabel('Possible center atoms',wrap=True)

    plt.title('Probability of sampling center atom of fragment',wrap=True)


    plt.legend()
    
    P_F1_N_flash = BytesIO()
    plt.savefig(P_F1_N_flash, dpi=100)
    P_F1_N_flash = PIL.Image.open(P_F1_N_flash)
    plt.close()




    
    text_plots_flash = PIL.Image.new('RGB', (half_pic_size[0],single_space*2), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_plots_flash)
    d.text((0,0), "FraGVAE: Bag of fragments reconstruction centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), "Status: Probability of sampling a center atom", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))


    imgs_comb = [np.asarray( i ) for i in [text_plots_flash,P_F1_N_flash,img_bag_frags_flash]]
    imgs_comb = np.vstack([imgs_comb[0],imgs_comb[1][:,:,0:3],imgs_comb[2]])

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    
    
    imgs_comb.save( 'imgs_gifs/imgs_to_gif/'+'F1_frag_'+str(gif_num).zfill(6)+'.png' )
    gif_num = gif_num +1
    
    given = select_atoms[select_node]
    
    return gif_num, given




def plot_NN_selection(select_NN, AI_NN, atoms, NN_Tensor,NN_Tensor_next, half_pic_size, single_space, gif_num, font_size,given, degree_idx,params):
    select_NN =int( np.argmax(select_NN[0]))
    AI_NN= AI_NN[0]
    NN_categories = ['0']+['-'+i for i in params["atoms"]]+['='+i for i in params["atoms"]]+['#'+i for i in params["atoms"]]+['≈'+i for i in params["atoms"]]

    num_F1 = np.sum( np.sum(atoms))

    display_F1 = np.ones(params['max_atoms'])
    flash_F1 = np.zeros(params['max_atoms'])
    
    bag_frags = display_F1_graphs(params,18,display_F1,flash_F1, (half_pic_size[0],(int(half_pic_size[1]/2)-single_space*2)),atoms,NN_Tensor)
    
    flash_F1[num_F1-1] = 1
    bag_frags_flash = display_F1_graphs(params,18,display_F1,flash_F1, (half_pic_size[0],(int(half_pic_size[1]/2)-single_space*2)),atoms,NN_Tensor_next)

    
    text_selected_frags = PIL.Image.new('RGB', (half_pic_size[0],single_space), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_selected_frags)
    d.text((0,0), "Decoded bag of fragments centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    text_selected_frags = PIL.Image.new('RGB', (half_pic_size[0],single_space), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_selected_frags)
    d.text((0,0), "Decoded bag of fragments centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
           
    img_bag_frags = vstack_img(text_selected_frags,bag_frags)    
    img_bag_frags_flash = vstack_img(text_selected_frags,bag_frags_flash)  
    
    
       
                    
    plt.figure(figsize=(half_pic_size[0]/100,(half_pic_size[1]-single_space*2)/100), dpi=100)

    plt.xticks(rotation='vertical')
    plt.bar(NN_categories, AI_NN, width=1, bottom=None,color ='c',label= 'P(N.N.|Zf, Frag Bag, [' + given+'])')
    plt.xticks(rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.2,top=0.9)
    
    axes = plt.gca()
    axes.set_ylim([0,1.2])

    plt.ylabel('Probability',wrap=True)
    plt.xlabel('Possible N.N. atoms',wrap=True)
    
    plt.title('Probability of #'+str(degree_idx)+' N.N. in fragment',wrap=True)
    
    
    plt.legend()
    P_F1_NN = BytesIO()
    plt.savefig(P_F1_NN, dpi=100)
    plt.close()
    P_F1_NN = PIL.Image.open(P_F1_NN)
    
    
    
    text_plots = PIL.Image.new('RGB', (half_pic_size[0],single_space*2), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_plots)
    d.text((0,0), "FraGVAE: Bag of fragments reconstruction centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), "Status: Probability of sampling nearest neighbour (N.N.) atoms", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    
    imgs_comb = [np.asarray( i ) for i in [text_plots,P_F1_NN,img_bag_frags]]
    imgs_comb = np.vstack([imgs_comb[0],imgs_comb[1][:,:,0:3],imgs_comb[2]])

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    
    
    imgs_comb.save( 'imgs_gifs/imgs_to_gif/'+'F1_frag_'+str(gif_num).zfill(6)+'.png' )
    gif_num = gif_num +1
    
    
    plt.figure(figsize=(half_pic_size[0]/100,(half_pic_size[1]-single_space*2)/100), dpi=100)
    plt.xticks(rotation='vertical')
    AI_NN_temp = AI_NN*0
    AI_NN_temp[select_NN] = AI_NN[select_NN]
    plt.bar(NN_categories, AI_NN, width=1, bottom=None,color ='c',label= 'P(N.N.|Zf, Frag Bag, [' + given+'])')
    plt.bar(NN_categories, AI_NN_temp, width=1, bottom=None,color ='b',label= 'Sampled '+NN_categories[select_NN])
    plt.xticks(rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.2,top=0.9)
    
    axes = plt.gca()
    axes.set_ylim([0,1.2])

    plt.ylabel('Probability',wrap=True)
    plt.xlabel('Possible N.N. atoms',wrap=True)
    
    plt.title('Probability of #'+str(degree_idx)+' N.N. in fragment',wrap=True)
    
    
    plt.legend()
    P_F1_NN = BytesIO()
    plt.savefig(P_F1_NN, dpi=100)
    plt.close()
    P_F1_NN = PIL.Image.open(P_F1_NN)


    
    text_plots_flash = PIL.Image.new('RGB', (half_pic_size[0],single_space*2), color = (255, 255, 255))
    d = PIL.ImageDraw.Draw(text_plots_flash)
    d.text((0,0), "FraGVAE: Bag of fragments reconstruction centered on atoms with radius 1", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), "Status: Probability of sampling nearest neighbour (N.N.) atoms", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))


    imgs_comb = [np.asarray( i ) for i in [text_plots_flash,P_F1_NN,img_bag_frags_flash]]
    imgs_comb = np.vstack([imgs_comb[0],imgs_comb[1][:,:,0:3],imgs_comb[2]])

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    
    
    imgs_comb.save( 'imgs_gifs/imgs_to_gif/'+'F1_frag_'+str(gif_num).zfill(6)+'.png' )
    gif_num = gif_num +1

    given = given +', '+ NN_categories[select_NN]
    
    return gif_num,given

        
                

    
        
        
def is_valid_substructure(mol_subs,mol,NN_Tensor_summed,NN_Tensor_summed_recon,map_edmol_Idx_to_mol):
    '''
    Determines if molecular subfragment is a subgraph of mol
    
    '''
    match_found = False
    if(mol!=-1):
        matches = mol.GetSubstructMatches(mol_subs,uniquify=False)
       
        for match in matches:
            match_found = True
            for sub_Idx in range(0,len(match)):
                if(not(all(NN_Tensor_summed_recon[match[sub_Idx]] - NN_Tensor_summed[map_edmol_Idx_to_mol[sub_Idx]]==0))):
                    match_found = False
                    break
            
            if(match_found):
                break        
    return match_found       
        