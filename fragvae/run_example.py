# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:28:50 2019

@author: ja550
libExamples = pd.read_csv('data_lib/250k_rndm_zinc_drugs.csv')"""


from load_model_parameters import load_params, calc_num_atom_features,save_model_params
from layers import Variational, NeuralGraphHidden,NeuralGraphSparsify, next_node_atom_error,node_atom_error,Hide_N_Drop,FHO_Error,Tanimoto,Ring_Edge_Mask,Mask_DB_atoms
from generators import data_generator,FHO_data_generator
from characterization import *
from utils import *
from F1_models import gen_F1_model
from FHO_models import gen_FHO_decoder_model,gen_FHO_encoder,gen_train_FHO_VAE
from VAE_callbacks import sigmoid_schedule,WeightAnnealer_epoch
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from functools import partial
import tensorflow as tf
from generators import *
from characterization import *
from PIL import Image, ImageDraw,ImageFont
import PIL
import copy
from sklearn.neighbors.kde import KernelDensity
import pickle
import keyboard


#tensorflow_shutup()
experiment = 27
params = load_params( )

params = load_params( 'models/experiment'+str(experiment).zfill(6)+'/FHO_training_params.json')
"""
Experiment Parameters

"""
params['exp'] =   experiment
params['model_dir'] =   'models/experiment'+str(params['exp']).zfill(6)+'/'
if not os.path.exists(params['model_dir'] ):
    os.makedirs(params['model_dir'] )

#save_model_params(params, 'Global_params.json')

#Load Old hyperparameters
#params = load_params('models/experiment'+str(params['exp']).zfill(6)+'/Global_params.json')


# Training models
Train_model = False

# Decode a random structure
Decode_rnd_Z = False

#Testing the molecular space by linerly steping between two molecular structures
Step_between_structures = False

# GIF gerneration of autoencoding molecular structures 
Autoencode_mol = False

# Test how well autoencoder can autoencode structures
Autoencode_mol_recontrust_Test = False

#Example of how to decode the latent space Z
Decode_around_Z = False

# Example of where the AI fails to autoencode the structure
AI_Interpret_VAE_fail = False

# Example of what part of the structure is necessary for predicting performance
AI_Interpret_predict = False

# Example of AI dreaming of a new fragments on molecular structure (concept not tested)
AI_Interpret_DeepDream = False

# Compare the predictive capabilities of FragVAE space vs ECFP space
Compare_ECFP_FragVAE_space = False

# Run experimental data (Screening solid state additives)

Compare_expt_methods = False






""""
AI_Interpret_VAE_fail, display bonds of VAE decoding where AI fails to reconstructure structure

"""
if(Compare_expt_methods):
    '''
    params['Pred_NN_layers']= [0,1]
    params['Pred_NN_Dropout'] =[0.2,0.35,0.5]
    params['Pred_NN_regl2'] =[0.01,0.05,0.1]
    params['Pred_NN_Dense_layer_size'] =[-1]
    params['num_features_expt']= [20,30,40]
    params['Pred_NN_growth']= [1,2]
    '''
    params['Pred_NN_layers'] = [0]
    params['Pred_NN_Dropout'] = [0]
    params['Pred_NN_regl2'] = [1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9]
    params['Pred_NN_regl2'] = [1E-9]

    params['Pred_NN_Dense_layer_size'] = [-1]
    params['num_features_expt'] = [15,17,19,21,23,25,26,28,30]
    params['num_features_expt'] = [30]
    params['Pred_NN_growth'] = [0]
    #x = compared_fingerprints_Rnd_forrest_expt_data(params)
    #x =compared_fingerprints_Rnd_forrest_expt_data_test(params)

    #..?:?x = compared_fingerprints_NN_expt_data(params)

    print('')

if(Compare_ECFP_FragVAE_space):
    y_predict_name = 'SAS'
    y_predict_name = 'qed'
    #y_predict_name = 'logP'
    Expt_results = compared_fingerprints(params,y_predict_name)

if(AI_Interpret_VAE_fail):
    AI_Interpret_VAE_num_tests = 5

    
    libExamples = pd.read_csv('data_lib/250k_rndm_zinc_drugs.csv')
    libExamples = libExamples.sample(frac=1).reset_index(drop=True)
    F1_encoder, FHO_encoder = load_encoders(params)
    F1_decoder, FHO_decoder = load_decoders(params)
    
    display_img_y =[]
    for y in range(0,9):
        display_img_x =[]
        for x in range(0,6):
            smile = libExamples["smiles"][y*x+x]
            fp = interpret_VAE_fails(smile,F1_encoder,FHO_encoder,FHO_decoder,params, max_change = 20,molSize=(150,150),max_decode = 10)
            display_img_x.append(fp)
            
        display_img_x   = [ PIL.Image.open(i) for i in display_img_x ]
        display_img_y.append(np.hstack( [np.asarray( i ) for i in display_img_x ]))    
    
    display_img = np.vstack(display_img_y)
    imgs_comb = PIL.Image.fromarray( display_img)
    
    imgs_comb.save( params['model_dir']+'AI_Interpret_VAE_fail.png' )
    display(imgs_comb)
        
if(AI_Interpret_predict):
    num_training_samples = 220000
    y_predict_name = 'logP'
    
    F1_encoder, FHO_encoder = load_encoders(params)
    
    libExamples = pd.read_csv('data_lib/250k_rndm_zinc_drugs.csv')     
    data_train_subset = libExamples[0:num_training_samples]

    rel_features = np.concatenate((np.zeros(params['finger_print']),np.ones(params['FHO_finger_print'])),axis = -1)
    

    FragVAE_features_train = gen_features_from_Frag_VAE(data_train_subset['smiles'], F1_encoder,FHO_encoder, rel_features,params)
    
    
    RND_Forreset_model = RandomForestRegressor(  n_estimators=params['n_estimators'],min_samples_leaf =params['min_samples_leaf'],max_features =params['max_features'])
    RND_Forreset_model.fit(FragVAE_features_train,data_train_subset[y_predict_name])
    display_img_y =[]
    for y in range(0,9):
        display_img_x =[]
        for x in range(0,6):
            smile = data_train_subset['smiles'][y*x+x]
            mol = smile_to_mol(smile, params)   
            fp  = interpret_predict(mol,RND_Forreset_model,F1_encoder,FHO_encoder,rel_features,params, remove_atom  = True, RFR = True,max_change = 3,molSize=(150,150))
            display_img_x.append(fp)
        display_img_x   = [ PIL.Image.open(i) for i in display_img_x ]
        
        display_img_y.append(np.hstack( [np.asarray( i ) for i in display_img_x ]))
    display_img = np.vstack(display_img_y)
    imgs_comb = PIL.Image.fromarray( display_img)
    imgs_comb.save( params['model_dir']+'AI_Interpret_predict_'+str(y_predict_name)+'.png' )
    
if(Decode_around_Z):
    structure_name = 'Ibruprofen'
    std_around_smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
    molecular_error = 1

    
    # Decodes a series of randomly generated molecules to test: novelity, uniquness, 

    F1_encoder, FHO_encoder = load_encoders(params)
    F1_decoder, FHO_decoder = load_decoders(params)
    
    atoms_1, edges_1, bonds_1 =  smile_to_tensor(std_around_smiles, params)
    Z1 , ZHO = Z_encoder(atoms_1, bonds_1 ,edges_1,F1_encoder, FHO_encoder,params)
    
    gen_set = set([])
    
    mol_set = []
    
    
    display_img_y =[]
    for y in range(0,9):
        display_img_x =[]
        for x in range(0,6):
            
            rand_Z1 = Z1+np.random.normal(0,(molecular_error+1)/10, len(Z1))
            rand_Z_HO = ZHO+np.random.normal(0,(molecular_error+1)/10, len(ZHO))

    
            NN_Tensor_sampled, atoms_sampled = decode_F1(rand_Z1,F1_decoder,F1_encoder, params)
            
            FHO_decoder_fun = gen_dropout_fun(Z_HO)
            
            mol = decode_FHO(atoms_sampled,NN_Tensor_sampled,params,rand_Z_HO, FHO_decoder_fun, FHO_encoder)
            
            smile = Chem.MolToSmiles(mol)
            if(smile is None):
                not_valid =not_valid+1
                print('Failed to generate smile')

                
            else:
                not_novel = not_novel + int(set([smile]).issubset(training_examples))
                not_unqiue = not_unqiue + int(set([smile]).issubset(gen_set))
                mol_set.append([mol])
                
                
            img, svg = gen_img_svg_mol(mol,molSize=(150,150))
            fp = BytesIO()
            svg2png(bytestring=svg,write_to=fp,scale=1)
            
            gen_set = gen_set.union(set([smile]))
            
            display_img_x.append(fp)
            
        display_img_x   = [ PIL.Image.open(i) for i in display_img_x ]
        
        display_img_y.append(np.hstack( [np.asarray( i ) for i in display_img_x ]))
        
    display_img = np.vstack(display_img_y)
    imgs_comb = PIL.Image.fromarray( display_img)
    imgs_comb.save( params['model_dir']+'Decode_around_Z_'+str(structure_name)+'_error_val_'+str(molecular_error)+'.png' )
    display(imgs_comb)
        

    print('Out of ' +str(num_decode)+' not_novel: ' +str(not_novel)+' not_unqiue:'+str(not_unqiue)+' not_valid:'+str(not_valid))     
     
if(Step_between_structures):
    smile_1 = 'c1cc(=C(C#N)C#N)ccc1=C(C#N)C#N'
    smile_2 = 'C(#N)C(=c1c(c(c(=C(C#N)C#N)c(c1F)F)F)F)C#N' 
    intervals = 3
    
    print('smile_1')
    img, svg = gen_img_svg_mol(MolFromSmiles(smile_1),molSize=(150,150))
    display(img)
    
    print('smile_2')
    img, svg = gen_img_svg_mol(MolFromSmiles(smile_2),molSize=(150,150))
    display(img)
    
    
    F1_encoder, FHO_encoder = load_encoders(params)
    F1_decoder, FHO_decoder = load_decoders(params)
    
    atoms_1, edges_1, bonds_1 =  smile_to_tensor(smile_1, params)
    Z_1_1 , Z_HO_1 = Z_encoder(atoms_1, bonds_1 ,edges_1,F1_encoder, FHO_encoder,params)
    
    atoms_2, edges_2, bonds_2 =  smile_to_tensor(smile_2, params)
    Z_1_2 , Z_HO_2 = Z_encoder(atoms_2, bonds_2 ,edges_2,F1_encoder, FHO_encoder,params)
    
    print('Auotencoding smile 1')
    mol = example_encode_decode(models,params,smiles = Auto_smiles, display_process = True, trial_walkthrough = True )
    img, svg = gen_img_svg_mol(mol)
    display(img)
    
    d_Z1 = (Z_1_2-Z_1_1)/(intervals+1)
    d_Z_HO = (Z_HO_2-Z_HO_1)/(intervals+1)
    
    Z_1 = Z_1_1
    Z_HO = Z_HO_1
    for i in range(0,intervals):
        Z_1 = Z_1 + d_Z1
        Z_HO = Z_HO + d_Z_HO 
        NN_Tensor_sampled, atoms_sampled = decode_F1(Z_1,F1_decoder,F1_encoder, params)
        
        FHO_decoder_fun = gen_dropout_fun(Z_HO)
        
        mol = decode_FHO(atoms_sampled,NN_Tensor_sampled,params,Z_HO, FHO_decoder_fun, FHO_encoder)
        print('Interval ' + str(i))
        img, svg = gen_img_svg_mol(mol,molSize=(150,150))
        display(img)

        
    print('Auotencoding smile 2')
    mol = example_encode_decode(models,params, smiles = smile_2,display_process = True, trial_walkthrough = True )
    img, svg = gen_img_svg_mol(mol,molSize=(150,150))
    display(img)

if(Autoencode_mol):

    F1_encoder, FHO_encoder = load_encoders(params)
    F1_decoder, FHO_decoder = load_decoders(params)
    smiles ='c1ccccc1'
    
    smiles = 'CCCN(CC)c1cc[nH+]c(C(=O)[O-])c1'
    smiles ='c1cc(=C(C#N)C#N)ccc1=C(C#N)C#N'
    #smiles = 'O=c1n(CCO)c2ccccc2n1CCO'
    smiles ='CCOc1ncnc(S(=O)(=O)CC)c1N'
    #smiles ='CC[C@@H](C)CNc1nc2ccc(Cl)cc2s1'
    smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
    smiles='CCOc1ccccc1N1C(=O)N=C(O)/C(=C/c2cc(OC)c(OC)cc2OC)C1=O'
    mol =  Chem.MolFromSmiles(smiles)
    imgs,svg = gen_img_svg_mol(mol)
    display(imgs)
    models =[F1_encoder,FHO_encoder,F1_decoder,FHO_decoder] 
    mol = example_encode_decode(models,params,smiles = smiles, display_process = True, trial_walkthrough = False )
    img,svg = gen_img_svg_mol(Chem.MolFromSmiles(smiles))
    display(img)
    img,svg = gen_img_svg_mol(mol)
    display(img)
    
if(Autoencode_mol_recontrust_Test):
    def calc_Tanimoto(z1_F1,z1_FHO,z2_F1,z2_FHO,params):
        z1_F1 = z1_F1[0]
        z1_FHO = z1_FHO[0]
        z2_F1 =z2_F1[0]
        z2_FHO=z2_FHO[0]
        if(params['F1_encoder_activation']=='elu'or params['F1_encoder_activation']=='tanh'):
            z1_F1 = np.maximum(z1_F1 + np.ones_like(z1_F1),np.zeros_like(z1_F1))
            z2_F1 = np.maximum((z2_F1 + np.ones_like(z2_F1)),np.zeros_like(z2_F1))
            
        if(params['FHO_decoder_activation']=='elu'or params['FHO_decoder_activation']=='tanh'):
            z1_FHO = np.maximum(z1_FHO + np.ones_like(z1_FHO),np.zeros_like(z1_FHO))
            z2_FHO = np.maximum((z2_FHO + np.ones_like(z2_FHO)),np.zeros_like(z2_FHO))
        
        z1_F1=z1_F1/(np.average(z1_F1)+1E-20)
        z2_F1=z2_F1/(np.average(z2_F1)+1E-20)
        z1_FHO=z1_FHO/(np.average(z1_FHO)+1E-20)
        z2_FHO=z2_FHO/(np.average(z2_FHO)+1E-20)
        
        z1 = np.array(list(z1_F1)+list(z1_FHO))
        z2 = np.array(list(z2_F1)+list(z2_FHO))
    
        tanimoto_max = np.maximum(z1,z2)
        tanimoto_min = np.minimum(z1,z2)
        
        tanimoto_max = np.sum(tanimoto_max,axis=-1)
        tanimoto_min = np.sum(tanimoto_min,axis=-1)
        
        tanimoto_val = tanimoto_min/(tanimoto_max+1E-20)
        
        return tanimoto_val
    
    from rdkit import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    mol_recontrust_num_tests =  100
    params['no_replacement'] = True

    
    libExamples = pd.read_csv('data_lib/250k_rndm_zinc_drugs.csv')
    libExamples = libExamples.sample(frac=1).reset_index(drop=True)
    '''
    num_atoms = []
    for i in range(0,len(libExamples)):
        num_atoms.append(  Chem.MolFromSmiles(libExamples["smiles"][i]).GetNumAtoms() )
    
    sorted_atoms = np.argsort(num_atoms)
    '''
    F1_encoder, FHO_encoder = load_encoders(params)
    F1_decoder, FHO_decoder = load_decoders(params)
    models =[F1_encoder,FHO_encoder,F1_decoder,FHO_decoder] 
    count = 0 
    non_valid = 0
    similarity = []
    similar_mols = []
    non_valid_mols = []
    recon_or_not =[]
    
    atom_size_test ={i:0 for i in list(range(0,80))}
    working_count =  {i:0 for i in list(range(0,80))}
    
    best_tanimoto=-1
    tanimoto_values = []
    encode_decode_mols = []
    for i in range(0,800+mol_recontrust_num_tests):
        found_structure = False
        smiles = libExamples["smiles"][i]
        print(smiles)
        mol_og = smile_to_mol(smiles, params)
        smiles = Chem.MolToSmiles(mol_og)
        atoms, edges, bonds =  smile_to_tensor(smiles, params,FHO_Ring_feature=True)
        Z_1_og , Z_HO_og,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS= Z_encoder(np.array([atoms]), np.array([bonds]) ,np.array([edges]),F1_encoder, FHO_encoder,params)
        for j in range(0,10):
            print('Attempt '+str(j))
            #smiles = libExamples["smiles"][sorted_atoms[i]]
            smiles = libExamples["smiles"][i]
            print(smiles)
            mol_og = smile_to_mol(smiles, params)
            smiles = Chem.MolToSmiles(mol_og)
            
            smiles = Chem.MolToSmiles(mol_og)
            mol = example_encode_decode(models,params,smiles = smiles, display_process = False, trial_walkthrough = False )
            if(not(params['include_charges'])):
                mol_og = remove_charge(mol_og)
                smiles = Chem.MolToSmiles(mol_og)
            
            img,svg = gen_img_svg_mol(mol_og)
            display(img)
            img,svg = gen_img_svg_mol(mol)
            display(img)
            
            
            atoms, edges, bonds =  mol_to_tensor(mol, params,FHO_Ring_feature=True)
            Z_1_new , Z_HO_new ,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS= Z_encoder(np.array([atoms]), np.array([bonds]) ,np.array([edges]),F1_encoder, FHO_encoder,params)
            Z_new = np.array(list(Z_1_new) +list(Z_HO_new))
            tanimoto_val = calc_Tanimoto(Z_1_og,Z_HO_og,Z_1_new,Z_HO_new,params)
            print('tanimoto_val ' +str(tanimoto_val) )
            try:
                if(tanimoto_val>best_tanimoto):
                    print('best_tanimoto')
                    best_decode_mol = copy.deepcopy(mol)
                    best_tanimoto=tanimoto_val
                auto_smiles = Chem.MolToSmiles(mol)
                if( mol.HasSubstructMatch(mol_og) and mol_og.HasSubstructMatch(mol)):
                    count = count+1
                    found_structure = True
                    working_count[mol_og.GetNumAtoms()]=working_count[mol_og.GetNumAtoms()]+1
                    break


            except:
                non_valid = non_valid + 1
                #non_valid_mols.append(smiles)
        atom_size_test[mol_og.GetNumAtoms()]=atom_size_test[mol_og.GetNumAtoms()]+1       
        if(found_structure):
            
            recon_or_not.append(1)
        else:
            recon_or_not.append(0)
            #similarity.append( DataStructs.FingerprintSimilarity(FingerprintMols.FingerprintMol(mol),FingerprintMols.FingerprintMol(mol_og)))
            #similar_mols =[[copy.deepcopy(mol_og),copy.deepcopy(mol)]]  
        print('Reconstructions: '+str(count)+ ' Tests:'+str(i)+' press space to continue...')
        
        
        tanimoto_values.append(copy.deepcopy(best_tanimoto))
        encode_decode_mols.append([copy.deepcopy(mol_og),copy.deepcopy(best_decode_mol)])
        best_tanimoto=-1
        #keyboard.wait(' ') 
    similarity = np.array(similarity)
    print('Number of correct reconstructions: '+str(count) +' out of '+str(mol_recontrust_num_tests))
    print('Mean similarity of non-correct reconstructions: '+str(np.mean(similarity)) )
    print('Number of non valid reconstructions: '+str(non_valid) +' out of '+str(mol_recontrust_num_tests))
    
#if(compare_methods):
    #3.2.4.3.2. sklearn.ensemble.RandomForestRegressor
    

if(Train_model):
    
    params['FHO'] = True
    reload = False
    
    params['predict_model'] = False
    
    if(not(params['FHO'])):
        params['model_name'] = 'F1_training'
    else:
        params['model_name'] = 'FHO_training'
    
    save_model_params(params, params['model_dir']+params['model_name']+'_params.json')
    
    if(not(params['FHO'])):
        if(reload):
            model = load_model(params['model_dir']+params['model_name']+'.h5',{'NeuralGraphHidden':NeuralGraphHidden,'node_atom_error':node_atom_error,'next_node_atom_error':next_node_atom_error,'NeuralGraphSparsify':NeuralGraphSparsify,'Variational':Variational})
            optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
            model.compile(optimizer=optimizer, loss='mean_absolute_error')
        else:
            model = gen_F1_model(params)
            g = data_generator(params)
            
        
        


        anneal_sigmod_slope = params['F1_anneal_sigmod_slope']
        kl_loss_weight =  params['F1_kl_loss_weight']
        anneal_epoch_start = params['F1_anneal_epoch_start']
    else:
        if(reload):
            '''
            #either works
            FHO_encoder = load_model(params['model_dir']+'FHO_encoder.h5',{'NeuralGraphHidden':NeuralGraphHidden,'NeuralGraphSparsify':NeuralGraphSparsify,'Variational':Variational,'Hide_N_Drop':Hide_N_Drop,'FHO_Error':FHO_Error})
            FHO_decoder = load_model(params['model_dir']+'FHO_decoder.h5',{'NeuralGraphHidden':NeuralGraphHidden,'NeuralGraphSparsify':NeuralGraphSparsify,'Variational':Variational,'Hide_N_Drop':Hide_N_Drop,'FHO_Error':FHO_Error})
            model = gen_train_FHO_VAE(params,params['model_name'],FHO_encoder,FHO_decoder)
            '''
            FHO_encoder = gen_FHO_encoder(params,'FHO_encoder')
            FHO_decoder = gen_FHO_decoder_model(params,'FHO_decoder')
            model = gen_train_FHO_VAE(params,params['model_name'],FHO_encoder,FHO_decoder)
            model.load_weights(params['model_dir']+params['model_name']+'_FHO_train_epoch_'+str(params['epoch_start'])+'.h5',{'NeuralGraphHidden':NeuralGraphHidden,'NeuralGraphSparsify':NeuralGraphSparsify,'Variational':Variational,'Hide_N_Drop':Hide_N_Drop,'FHO_Error':FHO_Error})  
        else:
            FHO_encoder = gen_FHO_encoder(params,'FHO_encoder')
            FHO_decoder = gen_FHO_decoder_model(params,'FHO_decoder')
            model = gen_train_FHO_VAE(params,params['model_name'],FHO_encoder,FHO_decoder)
            
            
        optimizer  = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='mean_absolute_error')

         

        
        
        anneal_sigmod_slope = params['FHO_anneal_sigmod_slope']
        kl_loss_weight =  params['FHO_kl_loss_weight']
        anneal_epoch_start = params['FHO_anneal_epoch_start']
        
    if(params['train_dataset'] == 'ESOL'):
        steps_per_epoch = 400
    elif(params['train_dataset']=='Additives'):
        steps_per_epoch = 400
    elif(params['train_dataset']=='Zinc15'):
        steps_per_epoch=int(250000/params['batch_size'])    
    elif(params['train_dataset']=='PubChem'):
        steps_per_epoch=int(250000/params['batch_size']) 
    else:
        steps_per_epoch=400
    
 
        
    # Add variational loss parameter    
    num_kl_loss_var = 0
    for i in range(0,len(model.non_trainable_variables)):
        if('Variational_layer' in model.non_trainable_variables[i].name):
            kl_loss_var = model.non_trainable_variables[i]
            num_kl_loss_var =num_kl_loss_var+1
    if(num_kl_loss_var>1):
        print('UNCERTAIN ABOUT WHAT VARIATIONAL LOSS PARAMTER TO USE IN TRAINING!!!!!! STOP')
        
        
    # Create all of the Callbacks
    vae_sig_schedule = partial(sigmoid_schedule, slope=anneal_sigmod_slope,   start=anneal_epoch_start)
    vae_anneal_callback = WeightAnnealer_epoch(vae_sig_schedule, kl_loss_var, kl_loss_weight, 'kl_loss_var' )
    csv_clb = CSVLogger(params['model_dir']+params['model_name']+'_hist.csv', append=True)
    nan_train = tf.keras.callbacks.TerminateOnNaN()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(params['model_dir']+params['model_name'] +"_best_CV.h5",     verbose=1,save_best_only=True)
    
    
    print('Model name : '+params['model_name'] )
    print('Experiment number : '+str(params['exp'] ) )
    
    for epoch in range(params['epoch_start'],params['max_epoch']):
        
        '''
        #Calculate error due to weights
        weights = FHO_encoder.get_weights()
        total = 0
        for weight in weights:
            total = total+np.sum(np.sum(weight*weight))
        print('encodeer weights reg loss: ' +str(total*params['NGF_reg']))
        weights = FHO_decoder.get_weights()
        total = 0
        for weight in weights:
            total = total+np.sum(np.sum(weight*weight))
        print('decoders weights reg loss: ' +str(total*params['FHO_decode_reg']))
        '''
        if(params['FHO']):
            g = FHO_data_generator(params,dataset = params['train_dataset'])
            CV_g = FHO_data_generator(params, dataset = params['CV_dataset'])
        else:
            g = data_generator(params,dataset = params['train_dataset'])
            CV_g = data_generator(params,dataset = params['CV_dataset']) 
            
        hist = model.fit_generator(g, steps_per_epoch=steps_per_epoch, epochs=epoch+1, verbose=int(1),   initial_epoch=epoch,
                                   callbacks = [cp_callback,vae_anneal_callback,csv_clb,nan_train],
                                   validation_data = CV_g, validation_steps = 10,
                                   use_multiprocessing=False)
        
        params['epoch_start'] = epoch
        
        if(params['FHO']):
            model.get_layer('FHO_encoder').save(params['model_dir']+'FHO_encoder.h5')
            model.get_layer('FHO_decoder').save(params['model_dir']+'FHO_decoder.h5') 
            
            model.get_layer('FHO_encoder').save(params['model_dir']+params['model_name']+'_encoder'+str(epoch)+'.h5')
            model.get_layer('FHO_decoder').save(params['model_dir']+params['model_name']+'_decoder'+str(epoch)+'.h5') 
            model.save(params['model_dir']+params['model_name']+'_FHO_train_epoch_'+str(epoch)+'.h5')
            model.save(params['model_dir']+params['model_name']+'_FHO_train_epoch_'+str(epoch)+'.h5')
        else:
            model.get_layer('Graphical_Encoder').save(params['model_dir']+'F1_Encoder.h5')
            model.get_layer('Atom_decoder').save(params['model_dir']+'F1_Atom_Decoder.h5')
            for degree_idx in range(0,params['max_degree'] ):
                model.get_layer('Decoder_degree_'+str(degree_idx)).save(params['model_dir']+'F1_NN_decoder_degree_'+str(degree_idx)+'.h5') 
                        
            model.save(params['model_dir']+params['model_name']+'_FHO_train_epoch_'+str(epoch)+'.h5')
            model.save(params['model_dir']+params['model_name']+'.h5')
            
        save_model_params(params, params['model_dir']+params['model_name']+'_params.json')
        

        
"""
# For debugging F1 training model
if(params['predict_model'] and not(params['FHO'])):
    print('Demonstrating F1 predictive model')
    from test_utils import test_training_model
    model = gen_F1_model(params)
    model.load_weights(params['model_dir']+params['model_name']+'_best_CV'+'.h5', by_name=False)
    test_training_model(model,params)
"""
    

    
