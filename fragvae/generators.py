
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:43:48 2019

@author: ja550
"""
from .hyperparameters import calc_num_atom_features
from .layers import Variational, NeuralGraphHidden,NeuralGraphSparsify, next_node_atom_error,node_atom_error,Hide_N_Drop,FHO_Error,Tanimoto
import random
from .fho_models import *
from .f1_models import *
from .display_funs import*
from .convert_mol_smile_tensor import *
from rdkit.Chem import MolFromSmiles
from random import randint
from rdkit.Chem import AllChem as Chem
import PIL
from PIL import Image
import keyboard
import os
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import math
from io import BytesIO
from PIL import Image, ImageDraw,ImageFont
import pandas as pd
import numpy as np
import copy 
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw



def f1_data_generator(params,libExamples,return_smiles = False):
    
    
    libExamples = libExamples.sample(frac=1).reset_index(drop=True)

    num_examples = len(libExamples)
    index = 1
    batch_size = params['F1_batch_size']
    params['num_atom_features'] = calc_num_atom_features(params)
    while 1:
        
        #df_subset = libExamples.sample(frac = batch_size / len(libExamples))
        df_subset = libExamples[(index-1)*batch_size:index*batch_size]

        df_subset = df_subset.reset_index(drop=True)
        molecules_atoms,  molecules_edges, molecules_bonds=  tensorise_smiles(df_subset, params ,printMe = False)


        NN_Tensor = []
        no_atoms = []
        for mol_idx in range(0,batch_size):
            NN_Tensor.append(gen_sparse_NN_Tensor(molecules_atoms[mol_idx],molecules_edges[mol_idx],molecules_bonds[mol_idx]))
            no_atoms.append([])
            for atom_idx in range(0,len(molecules_atoms[0])):
                no_atoms[mol_idx].append([(np.sum(molecules_atoms[mol_idx][atom_idx])==0)*1])
        no_atoms = np.array(no_atoms)
        NN_Tensor = np.array(NN_Tensor)
        
        
        
        
        #yield ({'molecules_atoms': molecules_atoms, 'molecules_edges':molecules_edges,'molecules_bonds':molecules_bonds,'NN_Tensor': NN_Tensor}, {'output_molecules_atoms': molecules_atoms, 'output_NN_Tensor': NN_Tensor})
        
        
        #add 1 to reference atoms if atoms
        
        total_error_dum = np.array([[0.0]]*batch_size)
        total_error_dum_out = np.array([[0.0]]*batch_size)
        if(return_smiles):
            yield ({'smiles':list(df_subset['smiles']),'atoms': molecules_atoms, 'edges':molecules_edges,'bonds':molecules_bonds,'NN_Tensor':NN_Tensor,'dummy_error_output':total_error_dum}, {'total_error_output': total_error_dum_out})

        else:
            yield ({'atoms': molecules_atoms, 'edges':molecules_edges,'bonds':molecules_bonds,'NN_Tensor':NN_Tensor}, {'total_error_output': total_error_dum_out})
        
        
        index = 1 + index
        if(index*batch_size>num_examples):
            index = 1
            libExamples = libExamples.sample(frac=1).reset_index(drop=True)
            
            
     

def fho_data_generator(params,libExamples,return_smiles = False,Dispaly_mol_fab=False):
    
    libExamples = libExamples.sample(frac=1).reset_index(drop=True)
    
    
    index = 1
    batch_size = params['FHO_batch_size']
    params['num_atom_features'] = calc_num_atom_features(params)
    params['Dispaly_mol_fab'] = Dispaly_mol_fab
    params['gifMe']  = False
    params['gif_idx'] = 0
    #params['max_atoms']= params['max_dangle_atoms']

    img_dict ={}
    img_dict['save_img']= True
    img_dict['molSize'] =(450,200)
    img_dict['single_space'] =17
    img_dict['font_size'] =15
    
    while 1:
        Gvalids_atoms = []
        Gvalids_bonds = []
        Gvalids_edges = []
        Gvalids_DG_atoms = []
        Gvalids_MSA = []
        Gvalids_MSB = []
      
        GInvalids_atoms = []
        GInvalids_bonds = []
        GInvalids_edges = []
        GInvalids_DG_atoms = []
        GInvalids_MSB = []
        GInvalids_MSA = []
        
        GInvalids2_atoms = []
        GInvalids2_bonds = []
        GInvalids2_edges = []
        GInvalids2_DG_atoms = []
        GInvalids2_MSB = []
        GInvalids2_MSA = []
        
        count_mini_bath = 0
        img_dict['gif_num'] =0
        
        batch_Gvalids_atoms = []
        batch_Gvalids_bonds = []
        batch_Gvalids_edges = []
        batch_Gvalids_DG_atoms = []
        batch_Gvalids_MSA = []
        batch_Gvalids_MSB = []
        
        batch_GInvalids_atoms = []
        batch_GInvalids_bonds = []
        batch_GInvalids_edges = []
        batch_GInvalids_DG_atoms = []
        batch_GInvalids_MSB = []
        batch_GInvalids_MSA = []
        
        
        batch_GInvalids2_atoms = []
        batch_GInvalids2_bonds = []
        batch_GInvalids2_edges = []
        batch_GInvalids2_DG_atoms = []
        batch_GInvalids2_MSB = []
        batch_GInvalids2_MSA = []
        
        G_atoms = []
        G_bonds = []
        G_edges  = []
        G_DG_atoms = []
        G_MSA = []
        G_MSB = []
        
        batch_count_idx = 0
        
        

        libExamples = libExamples.sample(frac=1).reset_index(drop=True) 
        
        for smile_idx in range(0,len(libExamples["smiles"])):
            if(params['Dispaly_mol_fab']):
                input("Press enter to continue")
            #likely need to remove charges on molecules to compare them    
            mol = smile_to_mol(libExamples["smiles"][smile_idx], params)



            #atoms, edges,bonds = mol_to_tensor(mol, params,FHO_Ring_feature=True,solving_bond=-2,suggested_bond=-2)
            atoms, edges,bonds = mol_to_tensor(mol, params,FHO_Ring_feature=True,dangling_atoms = np.zeros(params['max_dangle_atoms']),find_dangle_bonds=True)

            # Remove charges
            if(not(params['include_charges'])):
                mol = remove_charge(mol)
            
            
            NN_Tensor_non_sparse = gen_NN_Tensor(atoms,edges,bonds[:,:,0:params['num_bond_features']])
            #non_aromatic_NN = np.sum(NN_Tensor_non_sparse, axis = -2)[:,params['num_atom_features']+params['num_bond_features']]
            
            NN_Tensor = gen_sparse_NN_Tensor(atoms,edges,bonds[:,:,0:params['num_bond_features']])
            
            NN_Tensor = np.concatenate((NN_Tensor,np.array([NN_Tensor[0]*0])), axis =0)
            NN_Tensor_summed = np.sum(NN_Tensor, axis = -2)
            
            unqiue_frag = np.ones(mol.GetNumAtoms())
            

            for i in range(1,mol.GetNumAtoms()):
                for j in range(0,i):
                    if(all(NN_Tensor_summed[j]==NN_Tensor_summed[i]) and all(atoms[i]==atoms[j]) ):
                        '''
                        print(j)
                        print(NN_Tensor_summed[j])
                        print(NN_Tensor_summed[i])
                        print(atoms[j])
                        print(atoms[i])
                        '''
                        unqiue_frag[i]=0
                        break
            

            start_atom = randint(0, mol.GetNumAtoms()-1)
            
            
            #Generate a random fragment with num_frag_bonds start
            edmol_list,map_edmol_Idx_to_mol_list,cur_edmol_Idx_list, current_bond_list,P_remaining_DB = fragment_generator(mol,mol.GetNumBonds()-1 ,start_atom,params,True)
            
            params['FHO_attempts_per_batch']
            #for edmol_bond_idx in range(0,num_frag_bonds):
            num_batches_for_mol = int(mol.GetNumBonds()/params['FHO_attempts_per_batch'])
            start =  random.randint(0,np.mod(mol.GetNumBonds(),params['FHO_attempts_per_batch']))
            
            count_mini_bath = 0
            walk_idx = 0
            
            def unique_atoms_NN(atoms, unqiue_frag,NN_Tensor,mol):
                
                unique_atoms = []
                unqiue_NN_Tensor = []
                for i in range(len(unqiue_frag)):
                    if(unqiue_frag[i]==1):
                        unique_atoms.append(atoms[i])
                        unqiue_NN_Tensor.append(NN_Tensor[i])
                unqiue_NN_Tensor = np.array(unqiue_NN_Tensor)
                unique_atoms = np.array(unique_atoms)
                #print(unqiue_frag.shape)
                #print(unqiue_NN_Tensor.shape)
                #print(unique_atoms.shape)
                return unique_atoms, unqiue_NN_Tensor
            
            def unique_frag_map(unique_atoms, unqiue_NN_Tensor_summed,atoms,NN_Tensor_summed,mol):
                frag_unique_map = np.ones(len(atoms),np.int16)*-1
                for atom_idx in range(len(atoms)):
                    for unique_atom_idx in range(len(unique_atoms)):
                        if((atoms[atom_idx]==unique_atoms[unique_atom_idx]).all() and (NN_Tensor_summed[atom_idx]==unqiue_NN_Tensor_summed[unique_atom_idx]).all()):
                            frag_unique_map[atom_idx] = unique_atom_idx
                #print(frag_unique_map.shape)
                        
                return frag_unique_map
            
            unique_atoms, unqiue_NN_Tensor = unique_atoms_NN(atoms, unqiue_frag,NN_Tensor,mol)
            unqiue_NN_Tensor_summed = np.sum(unqiue_NN_Tensor, axis = -2)
            frag_unique_map = unique_frag_map(unique_atoms, unqiue_NN_Tensor_summed,atoms,NN_Tensor_summed,mol)
            G_mol_info = [atoms, NN_Tensor_summed,NN_Tensor,mol,params,unqiue_frag, unique_atoms, unqiue_NN_Tensor, frag_unique_map]
            
            if(params['Dispaly_mol_fab']):
                img,svg = gen_img_svg_mol(mol)
                #print('Complete mol')
                #display(img)
                #display(display_F1_graphs(params,len(unique_atoms),np.ones(mol.GetNumAtoms()),np.zeros(mol.GetNumAtoms()),img_dict['molSize'],unique_atoms, unqiue_NN_Tensor))

                    
                
            
            while(walk_idx<num_batches_for_mol*params['FHO_attempts_per_batch']+1):
                #if(params['Dispaly_mol_fab']):
                    #print('Start While Loop: '+libExamples["smiles"][smile_idx])
                    #print('Mini-batch: ' +str(count_mini_bath)+ ' Walks: '+ str(walk_idx)+' Num Walks: '+ str(num_batches_for_mol*params['FHO_attempts_per_batch']))

                
                if(count_mini_bath == params['FHO_attempts_per_batch']):
                    
                    if(walk_idx+start>=len(edmol_list)):
                        #print('Molecular reconstruction complete')
                        if(params['Dispaly_mol_fab']):
                            #print('Finial mini-batch graph')
                            #print('Molecular reconstruction complete')
                            img,svg = gen_img_svg_mol(mol)
                            #display(img)
                        Gvalids_atoms.append(copy.deepcopy(atoms))
                        Gvalids_bonds.append(copy.deepcopy(bonds))
                        Gvalids_edges.append(copy.deepcopy(edges))
                        Gvalids_DG_atoms.append( np.zeros(params['max_dangle_atoms']))
                        Gvalids_MSA.append( np.ones((params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1)))
                        Gvalids_MSB.append( np.ones((params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1)))
                        walk_idx = walk_idx+5
                    else:
                        #print('End of mini-batch')
                        #if(params['Dispaly_mol_fab']):
                            #print('Finial mini-batch graph')
                            
                        edmol = edmol_list[walk_idx+start]
                        map_edmol_Idx_to_mol =  copy.deepcopy(map_edmol_Idx_to_mol_list[walk_idx+start])
                        cur_edmol_Idx =  copy.deepcopy(cur_edmol_Idx_list[walk_idx+start])
                        current_bond =  copy.deepcopy(current_bond_list[walk_idx+start]    )
                            
                        temp_Gvalids_atoms,temp_Gvalids_bonds,temp_Gvalids_edges,temp_Gvalids_dangling_atoms,temp_Gvalids_MSA,temp_Gvalids_MSB,  temp_atoms, temp_edges,temp_bonds,temp_dangling_atoms,temp_MSA,temp_MSB,last_frag,img_dict = gen_valid_invalidGs(copy.deepcopy(G_mol_info),edmol,copy.deepcopy(map_edmol_Idx_to_mol),copy.deepcopy(cur_edmol_Idx),copy.deepcopy(current_bond),(),img_dict)
                        Gvalids_atoms.append(copy.deepcopy(temp_Gvalids_atoms))
                        Gvalids_bonds.append(copy.deepcopy(temp_Gvalids_bonds))
                        Gvalids_edges.append(copy.deepcopy(temp_Gvalids_edges))
                        Gvalids_DG_atoms.append(copy.deepcopy(temp_Gvalids_dangling_atoms))
                        Gvalids_MSA.append(copy.deepcopy(temp_Gvalids_MSA))
                        Gvalids_MSB.append(copy.deepcopy(temp_Gvalids_MSB))
                        
                        if(num_batches_for_mol*params['FHO_attempts_per_batch']+1==walk_idx+1):
                            walk_idx = walk_idx+1
                        
                    #add and display edmol


                    count_mini_bath = count_mini_bath+1
                elif(walk_idx+start<len(edmol_list)):

                    edmol = edmol_list[walk_idx+start]
                    map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol_list[walk_idx+start])
                    cur_edmol_Idx = copy.deepcopy(cur_edmol_Idx_list[walk_idx+start])
                    current_bond = copy.deepcopy(current_bond_list[walk_idx+start])
                    
                    img_dict['invalid_num'] ="1"
                    temp_Gvalids_atoms,temp_Gvalids_bonds,temp_Gvalids_edges,temp_Gvalids_dangling_atoms,temp_Gvalids_MSA,temp_Gvalids_MSB, temp_atoms, temp_edges,temp_bonds,temp_dangling_atoms,temp_MSA,temp_MSB,last_frag,img_dict = gen_valid_invalidGs(copy.deepcopy(G_mol_info),edmol,copy.deepcopy(map_edmol_Idx_to_mol),copy.deepcopy(cur_edmol_Idx),copy.deepcopy(current_bond),(),img_dict)

                    p_bonds = copy.deepcopy(P_remaining_DB[walk_idx+start])
                    
                    if(params['Dispaly_mol_fab']):
                        list_possible_roots = []
                    
                    p_bonds_with_same_connection = np.zeros_like(p_bonds)
                    bond = mol.GetBondWithIdx(current_bond)
                    atom1_idx = bond.GetBeginAtomIdx()
                    atom2_idx = bond.GetEndAtomIdx()    
                    if(any(map_edmol_Idx_to_mol==atom1_idx)*1+any(map_edmol_Idx_to_mol==atom2_idx)*1 <=1):
                        
                        p_bonds[current_bond] = 0
                    else:
                        
                        p_bonds_with_same_connection[current_bond] = 20
                        if(params['Dispaly_mol_fab']):
                            list_possible_roots.append(atom1_idx)
                            list_possible_roots.append(atom2_idx)
                        

                    
                    tested_root_atom = last_frag[0]
                    tested_leaf_atom = last_frag[1]
                    for test_bond_idx in range(0,len(p_bonds)):
                        if(p_bonds[test_bond_idx]==1):
                            test_bond = mol.GetBondWithIdx(test_bond_idx)
                            
                            test_atom1_idx = test_bond.GetBeginAtomIdx()
                            test_atom2_idx = test_bond.GetEndAtomIdx()
                            p_atom = np.array([any(map_edmol_Idx_to_mol==test_atom1_idx)*1,any(map_edmol_Idx_to_mol==test_atom2_idx)*1])
                            p_atom = p_atom/np.sum(p_atom)
                            rnd_atom = np.random.choice(2, 1, p=p_atom)[0]
                            possible_root_atom = int(np.array([test_atom1_idx,test_atom2_idx],dtype='int32')[rnd_atom])
                            not_rnd_atom = 1-rnd_atom
                            possible_leaf_atom = int(np.array([test_atom1_idx,test_atom2_idx],dtype='int32')[not_rnd_atom])
                            
                            if(not( all(NN_Tensor_summed[tested_leaf_atom] == NN_Tensor_summed[possible_leaf_atom] )) and all(atoms[tested_leaf_atom]==atoms[possible_leaf_atom] )):
                                if(test_bond.GetBondType() == mol.GetBondWithIdx(current_bond).GetBondType() and all(atoms[possible_root_atom]==atoms[tested_root_atom])):
                                    p_bonds_with_same_connection[test_bond_idx] = p_bonds_with_same_connection[test_bond_idx] +1

                                
                                NNs = mol.GetAtomWithIdx(tested_leaf_atom).GetNeighbors()
                                for NN in NNs:
                                    
                                    if( test_bond.GetBondType() ==mol.GetBondBetweenAtoms(tested_leaf_atom,NN.GetIdx()).GetBondType() and all(atoms[possible_root_atom]==atoms[NN.GetIdx()])):
                                        p_bonds_with_same_connection[test_bond_idx] =p_bonds_with_same_connection[test_bond_idx] +1 
                                        if(params['Dispaly_mol_fab']):
                                            list_possible_roots.append(possible_root_atom)
                                        break 
                    if(np.sum(p_bonds_with_same_connection)>=1):
                        #if(params['Dispaly_mol_fab']):
                            #print('Possible root frags')
                            #print(list_possible_roots)
                            #print(last_frag[1])
                        p_bonds = p_bonds_with_same_connection

                        
                    
                    if(np.sum(p_bonds)>=1):
                                                
                        #for 
                        all_invalid_bond = int(np.random.choice(params['max_bonds'], 1, p=p_bonds/np.sum(p_bonds))[0])
                    else:
                        all_invalid_bond = current_bond
                        
                    img_dict['invalid_num'] ="2"

                    dummy1,dummy2,dummy3,dummy4, dummy5,dummy6, temp_atoms2, temp_edges2,temp_bonds2,temp_dangling_atoms2,temp_MSA_2,temp_MSB_2,try_frag,img_dict = gen_valid_invalidGs(copy.deepcopy(G_mol_info),edmol,copy.deepcopy(map_edmol_Idx_to_mol),copy.deepcopy(cur_edmol_Idx),copy.deepcopy(all_invalid_bond),last_frag,img_dict)
               
            
                    
                    Gvalids_atoms.append(copy.deepcopy(temp_Gvalids_atoms))
                    Gvalids_bonds.append(copy.deepcopy(temp_Gvalids_bonds))
                    Gvalids_edges.append(copy.deepcopy(temp_Gvalids_edges))
                    Gvalids_DG_atoms.append(copy.deepcopy(temp_Gvalids_dangling_atoms))
                    Gvalids_MSA.append(copy.deepcopy(temp_Gvalids_MSA))
                    Gvalids_MSB.append(copy.deepcopy(temp_Gvalids_MSB))
                    
                    
                    
                    if(params['Dispaly_mol_fab']):
                        try:
                            edmol = Chem.EditableMol(copy.deepcopy(edmol_list[walk_idx+start+1].GetMol()))
                            map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol_list[walk_idx+start+1])
                            previous_map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol_list[walk_idx+start])
                            cur_edmol_Idx = copy.deepcopy(cur_edmol_Idx_list[walk_idx+start+1])
                            current_bond = copy.deepcopy(current_bond_list[walk_idx+start])
                            
                            # select root bond to add fragments
                            bond = mol.GetBondWithIdx(current_bond)
                            atom1_idx = bond.GetBeginAtomIdx()
                            atom2_idx = bond.GetEndAtomIdx()
                            p_atom = np.array([any(previous_map_edmol_Idx_to_mol==atom1_idx)*1,any(previous_map_edmol_Idx_to_mol==atom2_idx)*1])
    
                                
                            rnd_atom = np.random.choice(2, 1, p=p_atom/np.sum(p_atom))[0]
                            root_atom = int(np.array([atom1_idx,atom2_idx],dtype='int32')[rnd_atom])
                            not_rnd_atom = 1-rnd_atom
                            leaf_atom = int(np.array([atom1_idx,atom2_idx],dtype='int32')[not_rnd_atom])
                            root_atom_idx = int(np.argmax(previous_map_edmol_Idx_to_mol==root_atom))   
                            
                            cur_edmol_Idx = int(np.argmax(map_edmol_Idx_to_mol==leaf_atom))
    
    
                            leaf_atom_feature = gen_atom_features(mol.GetAtomWithIdx(leaf_atom),params)    
                            leaf_sparse_NN_feature =np.argmax(gen_sparse_NN_feature(leaf_atom_feature,gen_bond_features(bond,params)))
                            
                            img,svg = gen_img_svg_mol(edmol_list[walk_idx+start+1].GetMol())
                            #display(img)
                            dummy1, dummy2,dummy3,dummy4,G_Frag_fp,dummy5, dummy6  = gen_train_mol_tensor(edmol,NN_Tensor,map_edmol_Idx_to_mol,params,cur_edmol_Idx =cur_edmol_Idx,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=False,leaf_sparse_NN_feature=-1,dB_degree_mask=True,molSize=img_dict['molSize'])
                            img_dict['next_Gvalid']  = G_Frag_fp
                            #print('Next valid G')
                            #display(PIL.Image.open(G_Frag_fp))
                            
                            
                            flash_me = np.zeros(len(unqiue_NN_Tensor))
                            if(np.sum(p_atom)!=2):
                                flash_me[frag_unique_map[leaf_atom]] =1
    
                            img = display_F1_graphs(params,len(unqiue_NN_Tensor),np.ones(mol.GetNumAtoms()),flash_me,img_dict['molSize'],unique_atoms, unqiue_NN_Tensor)
                            frag_img_fp = BytesIO()
                            img.save( frag_img_fp,'PNG' )
                            img_dict['Valid_add_fragment'] = frag_img_fp
                            dict_frag = {}
                            for i in range(len(frag_unique_map)):
                                dict_frag[i] = frag_unique_map[i]
                            #print(dict_frag)
                            #print('leaf_atom ' +str(leaf_atom))

                            
                            def gif_gen_FHO(img_dict):
                                
                                def merge_imgs(img_overview, img_current, img_proposed,bag,valid,img_dict):
                                    font_size = img_dict['font_size']
                                    # overview
                                    text_size = (img_dict['molSize'][0], img_dict['single_space']*2)
                                    text_overview = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
                                    d = PIL.ImageDraw.Draw(text_overview)
        
                                    d.text((0,0), "FraGVAE: Molecular reconstruction training data (Zc)", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                                        
                                                                        
                                    # Bag
                                    text_size = (img_dict['molSize'][0], img_dict['single_space']*2)
                                    bag_text = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
                                    d = PIL.ImageDraw.Draw(bag_text)
        
                                    d.text((0,0), "Bag of fragments", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                                    
                                    
                                    img_left = vstack_img(vstack_img(text_overview,img_overview),vstack_img(bag_text,bag))
                                    
                                    
                                    # Current Fragment
                                    text_size = (img_dict['molSize'][0], img_dict['single_space']*2)
                                    text_current = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
                                    d = PIL.ImageDraw.Draw(text_current)
        
                                    d.text((0,0), "Previous valid fragment", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                                        
                                        
                                    
                                    # Proposed Fragment
                                    text_size = (img_dict['molSize'][0], img_dict['single_space']*2)
                                    text_proposed = PIL.Image.new('RGB', text_size, color = (255, 255, 255))
                                    d = PIL.ImageDraw.Draw(text_proposed)
        
                                    d.text((0,0), "Next proposed fragment. Fragment is valid/invalid if indicator (right)", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                                    d.text((0,img_dict['single_space']), "is green/red respectively", font=PIL.ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))                               
                                    img_middle = vstack_img(vstack_img(text_current,img_current),vstack_img(text_proposed,img_proposed))
                                
                                    
                                    #
                                    if(valid):
                                        fragmnet_indicator = PIL.Image.new('RGB', (40, img_dict['single_space']*4+2*img_dict['molSize'][1]), color = (0, 255, 0))
                                    else:
                                        fragmnet_indicator= PIL.Image.new('RGB', (40, img_dict['single_space']*4+2*img_dict['molSize'][1]), color = (255, 0, 0))
                                        
                                    d = PIL.ImageDraw.Draw(fragmnet_indicator)
                                    
                                    display_img= [img_left,img_middle,fragmnet_indicator]
                                    imgs_comb = [np.asarray( i ) for i in display_img]
    
    
                                    
                                    
                                    imgs_comb = PIL.Image.fromarray( np.hstack(imgs_comb))
                                    
                                    #if(z_test>-10E10 and (random.choice([True, False, False]) or best_z_test<z_test)):
                                    if(img_dict['save_img']):
                                        imgs_comb.save( 'imgs_gifs/imgs_to_gif/'+'FHO_training_gen'+str(img_dict['gif_num']).zfill(6)+'.png' )
                                    img_dict['gif_num']=img_dict['gif_num']+1
                                    
                                    display(imgs_comb)
                                            
                                    return img_dict
                                img_dict = merge_imgs(PIL.Image.open(img_dict['cur_overview']), PIL.Image.open(img_dict['cur_Gvalid']), PIL.Image.open(img_dict['cur_GInvalid1']),PIL.Image.open(img_dict['cur_GInvalid_frag1']),False,img_dict)
                                img_dict = merge_imgs(PIL.Image.open(img_dict['cur_overview']), PIL.Image.open(img_dict['cur_Gvalid']), PIL.Image.open(img_dict['cur_GInvalid2']) ,PIL.Image.open(img_dict['cur_GInvalid_frag2']),False,img_dict)
                                img_dict = merge_imgs(PIL.Image.open(img_dict['cur_overview']), PIL.Image.open(img_dict['cur_Gvalid']), PIL.Image.open(img_dict['next_Gvalid']) ,PIL.Image.open(img_dict['Valid_add_fragment']), True,img_dict)
                                return img_dict
                            img_dict = gif_gen_FHO(img_dict)
                        except:
                            failure = 1
                        
                    
                    if(count_mini_bath != params['FHO_attempts_per_batch']+1):
                        GInvalids_atoms.append(copy.deepcopy(temp_atoms))
                        GInvalids_bonds.append(copy.deepcopy(temp_bonds))
                        GInvalids_edges.append(copy.deepcopy(temp_edges))
                        GInvalids_DG_atoms.append(copy.deepcopy(temp_dangling_atoms))
                        GInvalids_MSA.append(copy.deepcopy(temp_MSA))
                        GInvalids_MSB.append(copy.deepcopy(temp_MSB))
                        

                        GInvalids2_atoms.append(copy.deepcopy(temp_atoms2))
                        GInvalids2_bonds.append(copy.deepcopy(temp_bonds2))
                        GInvalids2_edges.append(copy.deepcopy(temp_edges2))
                        GInvalids2_DG_atoms.append(copy.deepcopy(temp_dangling_atoms2))
                        GInvalids2_MSA.append(copy.deepcopy(temp_MSA_2))
                        GInvalids2_MSB.append(copy.deepcopy(temp_MSB_2))
                    walk_idx = walk_idx+1    
                    count_mini_bath = count_mini_bath+1


                
                if(count_mini_bath >= params['FHO_attempts_per_batch']+1):
                    #if(params['Dispaly_mol_fab']):
                        #print('Mini-Batch Complete')
                    
                    

                    
                    
                    G_atoms.append(copy.deepcopy(atoms))
                    G_bonds.append(copy.deepcopy(bonds))
                    G_edges.append(copy.deepcopy(edges))
                    G_DG_atoms.append( np.zeros(params['max_dangle_atoms']))
                    G_MSA.append( np.ones((params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1)))
                    G_MSB.append( np.ones((params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1)))
                    
                    batch_Gvalids_atoms.append(copy.deepcopy(np.array(Gvalids_atoms)))
                    batch_Gvalids_bonds.append(copy.deepcopy(np.array(Gvalids_bonds)))
                    batch_Gvalids_edges.append(copy.deepcopy(np.array(Gvalids_edges)))
                    batch_Gvalids_DG_atoms.append(copy.deepcopy(np.array(Gvalids_DG_atoms)))
                    batch_Gvalids_MSA.append(copy.deepcopy(np.array(Gvalids_MSA)))
                    batch_Gvalids_MSB.append(copy.deepcopy(np.array(Gvalids_MSB)))
                    
                    batch_GInvalids_atoms.append(copy.deepcopy(np.array(GInvalids_atoms)))
                    batch_GInvalids_bonds.append(copy.deepcopy(np.array(GInvalids_bonds)))
                    batch_GInvalids_edges.append(copy.deepcopy(np.array(GInvalids_edges)))
                    batch_GInvalids_DG_atoms.append(copy.deepcopy(np.array(GInvalids_DG_atoms)))
                    batch_GInvalids_MSA.append(copy.deepcopy(np.array(GInvalids_MSA)))
                    batch_GInvalids_MSB.append(copy.deepcopy(np.array(GInvalids_MSB)))
                    
                    batch_GInvalids2_atoms.append(copy.deepcopy(np.array(GInvalids2_atoms)))
                    batch_GInvalids2_bonds.append(copy.deepcopy(np.array(GInvalids2_bonds)))
                    batch_GInvalids2_edges.append(copy.deepcopy(np.array(GInvalids2_edges)))
                    batch_GInvalids2_DG_atoms.append(copy.deepcopy(np.array(GInvalids2_DG_atoms)))
                    batch_GInvalids2_MSA.append(copy.deepcopy(np.array(GInvalids2_MSA)))
                    batch_GInvalids2_MSB.append(copy.deepcopy(np.array(GInvalids2_MSB)))
                    
                    Gvalids_atoms = []
                    Gvalids_bonds = []
                    Gvalids_edges = []
                    Gvalids_DG_atoms = []
                    Gvalids_MSA = []
                    Gvalids_MSB = []
                  
                    GInvalids_atoms = []
                    GInvalids_bonds = []
                    GInvalids_edges = []
                    GInvalids_DG_atoms = []
                    GInvalids_MSB = []
                    GInvalids_MSA = []
                    
                    GInvalids2_atoms = []
                    GInvalids2_bonds = []
                    GInvalids2_edges = []
                    GInvalids2_DG_atoms = []
                    GInvalids2_MSB = []
                    GInvalids2_MSA = []
                    
                    count_mini_bath = 0
                            
                    batch_count_idx = batch_count_idx +1
                    if(batch_count_idx == batch_size):
                        #if(params['Dispaly_mol_fab']):
                            #print('Batch Complete')
                        batch_count_idx =0
                        
                        
                        G_atoms = np.array(G_atoms)
                        G_bonds = np.array( G_bonds)
                        G_edges =np.array(G_edges)
                        G_DG_atoms =np.array(G_DG_atoms)
                        G_MSA=np.array(G_MSA)
                        G_MSB=np.array(G_MSB)
                        
                        batch_Gvalids_atoms = np.array(batch_Gvalids_atoms)
                        batch_Gvalids_bonds = np.array(batch_Gvalids_bonds)
                        batch_Gvalids_edges = np.array(batch_Gvalids_edges)
                        batch_Gvalids_DG_atoms = np.array(batch_Gvalids_DG_atoms)
                        batch_Gvalids_MSA = np.array(batch_Gvalids_MSA)
                        batch_Gvalids_MSB = np.array(batch_Gvalids_MSB)
                        
                        
                        batch_GInvalids_atoms = np.array(batch_GInvalids_atoms)
                        batch_GInvalids_bonds = np.array(batch_GInvalids_bonds)
                        batch_GInvalids_edges = np.array(batch_GInvalids_edges)
                        batch_GInvalids_DG_atoms = np.array(batch_GInvalids_DG_atoms)
                        batch_GInvalids_MSA = np.array(batch_GInvalids_MSA)
                        batch_GInvalids_MSB = np.array(batch_GInvalids_MSB)
                        
                        batch_GInvalids2_atoms = np.array(batch_GInvalids2_atoms)
                        batch_GInvalids2_bonds = np.array(batch_GInvalids2_bonds)
                        batch_GInvalids2_edges = np.array(batch_GInvalids2_edges)
                        batch_GInvalids2_DG_atoms = np.array(batch_GInvalids2_DG_atoms)
                        batch_GInvalids2_MSA = np.array(batch_GInvalids2_MSA)
                        batch_GInvalids2_MSB = np.array(batch_GInvalids2_MSB)
                        

                        
                        
                        
                        
                        
                        
                        
                        
                        error = np.zeros((batch_size,1))
                        yield  ({'G_atoms':  G_atoms,  'G_bonds': G_bonds,  'G_edges': G_edges,'G_DG_atoms':G_DG_atoms,'G_MSA':G_MSA,'G_MSB':G_MSB,
                                'G_valid_atoms': batch_Gvalids_atoms, 'G_valid_bonds':batch_Gvalids_bonds,'G_valid_edges':batch_Gvalids_edges,'G_valid_DG_atoms':batch_Gvalids_DG_atoms, 'G_valid_MSA':batch_Gvalids_MSA,'G_valid_MSB':batch_Gvalids_MSB,
                                'G_INV1_atoms':batch_GInvalids_atoms, 'G_INV1_bonds':batch_GInvalids_bonds,'G_INV1_edges':batch_GInvalids_edges, 'G_INV1_DG_atoms':batch_GInvalids_DG_atoms,'G_INV1_MSA':batch_GInvalids_MSA,'G_INV1_MSB':batch_GInvalids_MSB,
                                'G_INV2_atoms':batch_GInvalids2_atoms,'G_INV2_bonds':batch_GInvalids2_bonds,'G_INV2_edges':batch_GInvalids2_edges,'G_INV2_DG_atoms':batch_GInvalids2_DG_atoms,'G_INV2_MSA':batch_GInvalids2_MSA,'G_INV2_MSB':batch_GInvalids2_MSB}, {'total_error_output':error})

    
                        batch_Gvalids_atoms = []
                        batch_Gvalids_bonds = []
                        batch_Gvalids_edges = []
                        batch_Gvalids_DG_atoms = []
                        batch_Gvalids_MSA = []
                        batch_Gvalids_MSB = []
                        
                        batch_GInvalids_atoms = []
                        batch_GInvalids_bonds = []
                        batch_GInvalids_edges = []
                        batch_GInvalids_DG_atoms = []
                        batch_GInvalids_MSB = []
                        batch_GInvalids_MSA = []
                        
                        
                        batch_GInvalids2_atoms = []
                        batch_GInvalids2_bonds = []
                        batch_GInvalids2_edges = []
                        batch_GInvalids2_DG_atoms = []
                        batch_GInvalids2_MSB = []
                        batch_GInvalids2_MSA = []
                        
                        G_atoms = []
                        G_bonds = []
                        G_edges  = []
                        G_DG_atoms = []
                        G_MSA = []
                        G_MSB = []

        

        
         
def gen_valid_invalidGs(G_mol_info,edmol,map_edmol_Idx_to_mol,cur_edmol_Idx,current_bond,last_frag,img_dict):
    [atoms, NN_Tensor_summed,NN_Tensor,mol,params,unqiue_frag, unique_atoms, unqiue_NN_Tensor,frag_unique_map]=G_mol_info 

    # select root bond to add fragments
    bond = mol.GetBondWithIdx(current_bond)
    atom1_idx = bond.GetBeginAtomIdx()
    atom2_idx = bond.GetEndAtomIdx()
    previous_map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol)



    
    p_atom = np.array([any(map_edmol_Idx_to_mol==atom1_idx)*1,any(map_edmol_Idx_to_mol==atom2_idx)*1])
    
    
        
    
    
    
    rnd_atom = np.random.choice(2, 1, p=p_atom/np.sum(p_atom))[0]
    root_atom = int(np.array([atom1_idx,atom2_idx],dtype='int32')[rnd_atom])
    not_rnd_atom = 1-rnd_atom
    leaf_atom = int(np.array([atom1_idx,atom2_idx],dtype='int32')[not_rnd_atom])
    
    
    if(np.sum(p_atom)==2 and len(last_frag) ==2):
        if(last_frag[0]==root_atom and last_frag[1] == leaf_atom):
            last_frag = (leaf_atom,root_atom)
        else:
            last_frag = (root_atom,leaf_atom)
        #if(params['Dispaly_mol_fab']):
            #print('Extending a ring with right frag too big')
    
    edmol_frag = copy.deepcopy(edmol.GetMol())
    edmol_frag = Chem.EditableMol(edmol_frag)
    edmol_frag.AddAtom(mol.GetAtomWithIdx(leaf_atom))

    #BOND already exsist error
    
    
    cur_edmol_Idx = cur_edmol_Idx+1
    map_edmol_Idx_to_mol[cur_edmol_Idx] = leaf_atom
    edmol_frag.AddBond(int(np.argmax(map_edmol_Idx_to_mol==root_atom)),cur_edmol_Idx,bond.GetBondType())
    mol_frag = edmol_frag.GetMol()
    
    root_atom_idx = int(np.argmax(map_edmol_Idx_to_mol==root_atom))
    
    matches = possible_frag_root_matches(mol_frag,mol,NN_Tensor_summed,map_edmol_Idx_to_mol)
    root_NN_feature = gen_sparse_NN_feature(gen_atom_features(mol.GetAtomWithIdx(root_atom),params),gen_bond_features(bond,params))
    root_NN_feature = np.concatenate((np.array([0]),root_NN_feature))
    
    leaf_atom_feature = gen_atom_features(mol.GetAtomWithIdx(leaf_atom),params)

    
    leaf_sparse_NN_feature =np.argmax(gen_sparse_NN_feature(leaf_atom_feature,gen_bond_features(bond,params)))
    # Gerate atom, bond and edge tensor for previous molecule before adding Fragment or bonding to itself.
    
    if(params['Dispaly_mol_fab'] and len(last_frag) == 0):
        #print('Current status')
        root_atom_color = (139/250,69/250,19/250)
        #if(valid_connection):
        root_bond_color = (205/250,133/250,63/250)
        highlights,colors,highlightBonds,bond_colors = highlight_map_edmol(map_edmol_Idx_to_mol,cur_edmol_Idx,edmol,edmol_frag,mol,bond)
        colors[root_atom] = root_atom_color
        #display_mol( mol,highlight=[int(map_edmol_Idx_to_mol[cur_edmol_Idx])],colors={int(map_edmol_Idx_to_mol[cur_edmol_Idx]):(0,1,0)})
    
        img, svg = gen_img_svg_mol(mol,highlight=highlights,colors =colors ,highlightBonds=highlightBonds,bond_colors=bond_colors,molSize = img_dict['molSize'])
        #display(img)
        #print(type(img))
        cur_img_fp = BytesIO()
        svg2png(bytestring=svg,write_to=cur_img_fp,scale=1)
        
        img_dict['cur_overview']  = cur_img_fp
    
    
    if(params['Dispaly_mol_fab']):
                                                                                                        
        Gvalids_atoms, Gvalids_edges,Gvalids_bonds,Gvalids_DG_atoms,G_Frag_fp,Gvalids_mask_symmetric_atoms, Gvalids_mask_symmetric_bonds  = gen_train_mol_tensor(edmol,NN_Tensor,previous_map_edmol_Idx_to_mol,params,cur_edmol_Idx =cur_edmol_Idx-1,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=False,leaf_sparse_NN_feature=leaf_sparse_NN_feature,dB_degree_mask=True,molSize=img_dict['molSize'])
        if(len(last_frag) == 0):
            #print('G - valid')
            #display(PIL.Image.open(G_Frag_fp))
            img_dict['cur_Gvalid']  = G_Frag_fp
    else:
        Gvalids_atoms, Gvalids_edges,Gvalids_bonds,Gvalids_DG_atoms,Gvalids_mask_symmetric_atoms, Gvalids_mask_symmetric_bonds  = gen_train_mol_tensor(edmol,NN_Tensor,previous_map_edmol_Idx_to_mol,params,cur_edmol_Idx =cur_edmol_Idx-1,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=False,leaf_sparse_NN_feature=leaf_sparse_NN_feature,dB_degree_mask=True)

    
        
    connection_verdicts = []
    
    for atom_idx in range(0,mol.GetNumAtoms()):
        potential_leaf_atom_feature = atoms[atom_idx] 
        NN_Tensor_slice = copy.deepcopy(NN_Tensor[atom_idx])
        connection_verdicts.append(check_frag_leaf_match(mol,matches,root_NN_feature,leaf_atom_feature,potential_leaf_atom_feature,NN_Tensor_slice,NN_Tensor_summed,cur_edmol_Idx))
    connection_verdicts = np.array(connection_verdicts)
    connection_verdicts_frag =connection_verdicts


    # switch to finding rings in structure
                    # switch to finding rings in structure
                
    #map_edmol_Idx_to_mol[cur_edmol_Idx] = -1
    #cur_edmol_Idx = int(np.argmax(map_edmol_Idx_to_mol==root_atom))
                      
    root_atom_feature = gen_atom_features(mol.GetAtomWithIdx(root_atom),params)
    required_NN = np.concatenate((np.array([0]),gen_sparse_NN_feature(root_atom_feature,gen_bond_features(bond,params))) )
    ring_connection_verdicts =[]
    mol_ring = edmol.GetMol()
    for atom_idx in range(0,mol.GetNumAtoms()):
        if(atom_idx<mol_ring.GetNumAtoms()):
            temp = ring_connection(mol,edmol,atom_idx,previous_map_edmol_Idx_to_mol,params,leaf_atom_feature,required_NN,NN_Tensor_summed,root_atom,bond)
        else:
            temp =[0,0]
        ring_connection_verdicts.append(temp)
        
    ring_connection_verdicts = np.array(ring_connection_verdicts)
    
    invalid_connection_frag = connection_verdicts_frag[:,0]*(1-connection_verdicts_frag[:,1])
    invalid_connection_rings = ring_connection_verdicts[:,0]*(1-ring_connection_verdicts[:,1])
    
    invlid_frags =  np.sum(invalid_connection_frag)>=1
    invalid_rings = np.sum(invalid_connection_rings)>=1
    
    
    invalid = True
    Add_frag = True
    if(invlid_frags and invalid_rings):
        Add_frag = random.choice([True, False])
    elif(invalid_rings):
        Add_frag = False
    elif(invlid_frags):
        Add_frag = True
    else:
        invalid = False
        if(params['Dispaly_mol_fab']):
            #print('No invalid frag can be added, used current edmol')
            #display(PIL.Image.open(G_Frag_fp))
            img_dict['cur_GInvalid'+img_dict['invalid_num']] = G_Frag_fp
            flash_me = np.zeros(len(unqiue_NN_Tensor))
            
            img = display_F1_graphs(params,len(unique_atoms),np.ones(mol.GetNumAtoms()),flash_me,img_dict['molSize'],unique_atoms, unqiue_NN_Tensor)
            frag_img_fp = BytesIO()
            img.save( frag_img_fp,'PNG' )
            img_dict['cur_GInvalid_frag'+img_dict['invalid_num']] = frag_img_fp

        
        G_invalid_atoms = copy.deepcopy(Gvalids_atoms)
        G_invalid_bonds = copy.deepcopy(Gvalids_bonds)
        G_invalid_edges = copy.deepcopy(Gvalids_edges)
        G_invalid_dangling_atoms = copy.deepcopy(Gvalids_DG_atoms)
        G_invalid_mask_symmetric_atoms = copy.deepcopy(Gvalids_mask_symmetric_atoms)
        G_invalid_mask_symmetric_bonds = copy.deepcopy(Gvalids_mask_symmetric_bonds)
        
    if(Add_frag and invalid):
        if(len(last_frag) ==2):
            possible_leaf_atom = last_frag[1]
            #if(params['Dispaly_mol_fab']):
                #print('last_frag_idx '+str(last_frag[1]))
            if(invalid_connection_frag[possible_leaf_atom] == 1):
                fragment_idx = possible_leaf_atom
                
            else:
                rosetta_set = set(previous_map_edmol_Idx_to_mol)
                possible_atoms_set = set(np.array(range(0,mol.GetNumAtoms())))
                other_atoms = list(possible_atoms_set.difference(rosetta_set))
                for temp_frag_idx in range(0,len(invalid_connection_frag)):
                    if(invalid_connection_frag[temp_frag_idx]==1):
                        for check_atom_idx in other_atoms:
                            if(all(atoms[temp_frag_idx]==atoms[check_atom_idx]) and  all(NN_Tensor_summed[temp_frag_idx]==NN_Tensor_summed[check_atom_idx])):
                                invalid_connection_frag[temp_frag_idx] = 2
                                break
                #print(invalid_connection_frag)
                fragment_idx = int(np.random.choice(mol.GetNumAtoms(), 1, p=(invalid_connection_frag*unqiue_frag)/(np.sum(( invalid_connection_frag*unqiue_frag) )))[0])
        else:
            rosetta_set = set(previous_map_edmol_Idx_to_mol)
            possible_atoms_set = set(np.array(range(0,mol.GetNumAtoms())))
            other_atoms = list(possible_atoms_set.difference(rosetta_set))
            for temp_frag_idx in range(0,len(invalid_connection_frag)):
                if(invalid_connection_frag[temp_frag_idx]==1):
                    for check_atom_idx in other_atoms:
                        if(all(atoms[temp_frag_idx]==atoms[check_atom_idx]) and  all(NN_Tensor_summed[temp_frag_idx]==NN_Tensor_summed[check_atom_idx])):
                            invalid_connection_frag[temp_frag_idx] = 2
                            break
            #print(invalid_connection_frag)
            fragment_idx = int(np.random.choice(mol.GetNumAtoms(), 1, p=(invalid_connection_frag*unqiue_frag)/(np.sum(( invalid_connection_frag*unqiue_frag) )))[0])
        
        if(params['Dispaly_mol_fab']):
            #print('G - In valid, leaf idx: ' +str(fragment_idx))
            flash_me = np.zeros(len(unqiue_NN_Tensor))
            flash_me[frag_unique_map[fragment_idx]] =1
            img = display_F1_graphs(params,len(unqiue_NN_Tensor),np.ones(mol.GetNumAtoms()),flash_me,img_dict['molSize'],unique_atoms, unqiue_NN_Tensor)
            #display(img)
            frag_img_fp = BytesIO()
            img.save( frag_img_fp,'PNG' )
            img_dict['cur_GInvalid_frag'+img_dict['invalid_num']] = frag_img_fp
            
        G_invalid_atoms, G_invalid_edges,G_invalid_bonds,G_invalid_dangling_atoms,G_invalid_mask_symmetric_atoms, G_invalid_mask_symmetric_bonds,img_dict = add_frag_edmol(edmol,previous_map_edmol_Idx_to_mol,fragment_idx,root_atom_idx,cur_edmol_Idx-1,mol.GetAtomWithIdx(leaf_atom),NN_Tensor,params,leaf_sparse_NN_feature,bond,img_dict)
    elif(not(Add_frag) and invalid):
        leaf_atom_idx = int(np.random.choice(mol.GetNumAtoms(), 1, p=invalid_connection_rings/np.sum(invalid_connection_rings))[0])
        #leaf_atom_idx =  int(np.argmax(previous_map_edmol_Idx_to_mol==leaf_atom))
        #if(params['Dispaly_mol_fab']):
            #print('G - In valid, leaf idx: '+str(leaf_atom_idx))
            
        G_invalid_atoms, G_invalid_edges,G_invalid_bonds,G_invalid_dangling_atoms,G_invalid_mask_symmetric_atoms, G_invalid_mask_symmetric_bonds,img_dict  = form_ring_edmol(edmol,root_atom_idx,leaf_atom_idx,previous_map_edmol_Idx_to_mol,bond,NN_Tensor,params,leaf_sparse_NN_feature,img_dict)
        
        
        if(params['Dispaly_mol_fab']):
            flash_me = np.zeros(len(unqiue_NN_Tensor))
            img = display_F1_graphs(params,len(unqiue_NN_Tensor),np.ones(mol.GetNumAtoms()),flash_me,img_dict['molSize'],unique_atoms, unqiue_NN_Tensor)
            frag_img_fp = BytesIO()
            img.save( frag_img_fp,'PNG' )
            img_dict['cur_GInvalid_frag'+img_dict['invalid_num']] = frag_img_fp

        
    return  Gvalids_atoms,Gvalids_bonds,Gvalids_edges,Gvalids_DG_atoms,Gvalids_mask_symmetric_atoms, Gvalids_mask_symmetric_bonds,G_invalid_atoms, G_invalid_edges,G_invalid_bonds,G_invalid_dangling_atoms,G_invalid_mask_symmetric_atoms, G_invalid_mask_symmetric_bonds,(root_atom,leaf_atom),img_dict




            
def form_ring_edmol(edmol,root_atom_idx,leaf_atom_idx,map_edmol_Idx_to_mol,cur_bond,NN_Tensor,params,leaf_sparse_NN_feature,img_dict):
    test_edmol = copy.deepcopy(edmol.GetMol())
    test_edmol =Chem.EditableMol(test_edmol)
    test_map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol)
    

    test_edmol.AddBond(int(root_atom_idx),int(leaf_atom_idx),cur_bond.GetBondType())
    
    if(params['Dispaly_mol_fab']):
        #print('forming ring')
        G_Frag_atoms, G_Frag_edges,G_Frag_bonds,G_Frag_dangling_atoms,G_Frag_fp,G_Frag_mask_symmetric_atoms, G_Frag_mask_symmetric_bonds =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,params,cur_edmol_Idx=  leaf_atom_idx,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=False,leaf_sparse_NN_feature=-1,dB_degree_mask=True,molSize=img_dict['molSize'])
        #display(PIL.Image.open(G_Frag_fp))
        img_dict['cur_GInvalid'+img_dict['invalid_num']] = G_Frag_fp
        
        
        
    else:
        G_Frag_atoms, G_Frag_edges,G_Frag_bonds,G_Frag_dangling_atoms,G_Frag_mask_symmetric_atoms, G_Frag_mask_symmetric_bonds =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,params,cur_edmol_Idx = leaf_atom_idx,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=False,leaf_sparse_NN_feature=leaf_sparse_NN_feature,dB_degree_mask=True)
                                    
    return G_Frag_atoms, G_Frag_edges,G_Frag_bonds,G_Frag_dangling_atoms,G_Frag_mask_symmetric_atoms, G_Frag_mask_symmetric_bonds,img_dict
            



def add_frag_edmol(edmol,map_edmol_Idx_to_mol,fragment_idx,root_atom_idx,cur_edmol_Idx,leaf_atom,NN_Tensor,params,leaf_sparse_NN_feature,cur_bond,img_dict):
    
    test_edmol = copy.deepcopy(edmol.GetMol())
    test_edmol = Chem.EditableMol(test_edmol)
    test_map_edmol_Idx_to_mol = copy.deepcopy(map_edmol_Idx_to_mol)
    test_map_edmol_Idx_to_mol[cur_edmol_Idx+1] = fragment_idx
    
    not_print = test_edmol.AddAtom(leaf_atom)
    not_print = test_edmol.AddBond(int(root_atom_idx),int(cur_edmol_Idx+1),cur_bond.GetBondType())
    if(params['Dispaly_mol_fab']):
        #print('adding frag: '+str(fragment_idx))
        G_Frag_atoms, G_Frag_edges,G_Frag_bonds,G_Frag_dangling_atoms,G_Frag_fp,G_Frag_mask_symmetric_atoms, G_Frag_mask_symmetric_bonds =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,params,cur_edmol_Idx =cur_edmol_Idx+1,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=True,leaf_sparse_NN_feature=leaf_sparse_NN_feature,dB_degree_mask=True,molSize=img_dict['molSize'])
        #display(PIL.Image.open(G_Frag_fp))
        img_dict['cur_GInvalid'+img_dict['invalid_num']] = G_Frag_fp
    else:
        G_Frag_atoms, G_Frag_edges,G_Frag_bonds,G_Frag_dangling_atoms,G_Frag_mask_symmetric_atoms, G_Frag_mask_symmetric_bonds =gen_train_mol_tensor(test_edmol,NN_Tensor,test_map_edmol_Idx_to_mol,params,cur_edmol_Idx =cur_edmol_Idx+1,edmol_root_idx =root_atom_idx,display_dangle=params['Dispaly_mol_fab'],updated_atom=True,leaf_sparse_NN_feature=leaf_sparse_NN_feature,dB_degree_mask=True)

    return G_Frag_atoms, G_Frag_edges,G_Frag_bonds,G_Frag_dangling_atoms,G_Frag_mask_symmetric_atoms, G_Frag_mask_symmetric_bonds,img_dict
            
def display_F1_tested(NN_Tensor_slice, node_atom_features):
    edmol = Chem.EditableMol(MolFromSmiles(''))
    
    highlights = [0]
    colors={0:(0,1,0)}
    
    idx = np.argmax(node_atom_features)
    if(params['atoms'][idx]=='H'):
        mol = Chem.MolFromSmiles('[H]')
    else:
        mol = Chem.MolFromSmiles(params['atoms'][idx])
    atom = mol.GetAtomWithIdx(0)
    
    edmol.AddAtom(atom)
    num = 1
    for i in range(len(NN_Tensor_slice)):
        if(NN_Tensor_slice[i][0]!=1):
            atom,bond = atom_bond_from_sparse_NN_feature(np.argmax(NN_Tensor_slice[i])-1,params)
            edmol.AddAtom(atom)
            edmol.AddBond(0,num,bond)
            highlights.append(num)
            colors[num] = (0,1,1)
            
            num = num+1
    display_mol(edmol.GetMol(),highlight=highlights,colors=colors)
            
def fragment_generator(mol,num_frag_bonds,start_atom,params,walk_through_atom):
    
    
    edmol_list=[]
    map_edmol_Idx_to_mol_list=[]
    cur_edmol_Idx_list=[]
    current_bond_list=[]
    P_remaining_DB = []
    
    edmol = Chem.EditableMol(MolFromSmiles(''))
    
    map_edmol_Idx_to_mol = np.ones(params['max_dangle_atoms'],dtype = int)*-1
    map_edmol_Idx_to_mol[0] = start_atom
    cur_edmol_Idx = 0
    
    edmol.AddAtom(mol.GetAtomWithIdx(start_atom))
    p_bonds = np.zeros(params['max_bonds'] )
    
    for bond in mol.GetAtomWithIdx(start_atom).GetBonds():
        p_bonds[bond.GetIdx()] =1
        
    current_bond = int(np.random.choice(params['max_bonds'], 1, p=p_bonds/np.sum(p_bonds))[0])
    
    if(walk_through_atom):
        edmol_list.append( Chem.EditableMol( copy.deepcopy(edmol.GetMol())))
        map_edmol_Idx_to_mol_list.append(copy.deepcopy(map_edmol_Idx_to_mol))
        cur_edmol_Idx_list.append(copy.deepcopy(cur_edmol_Idx))
        current_bond_list.append(copy.deepcopy(current_bond))
        P_remaining_DB.append(copy.deepcopy(p_bonds))
    
    for edmol_bond_idx in range(0,num_frag_bonds):
        
        bond = mol.GetBondWithIdx(current_bond)
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        

        p_atom = np.array([any(map_edmol_Idx_to_mol==atom1_idx)*1,any(map_edmol_Idx_to_mol==atom2_idx)*1])
        
        not_Ring = True
        if(np.sum(p_atom) == 2):
            not_Ring = False
        
        if(not_Ring):
            
            
            p_atom = p_atom/np.sum(p_atom)
            rnd_atom = np.random.choice(2, 1, p=p_atom)[0]
            root_atom = int(np.array([atom1_idx,atom2_idx])[rnd_atom])
            not_rnd_atom = 1-rnd_atom
            leaf_atom = int(np.array([atom1_idx,atom2_idx])[not_rnd_atom])
            
            cur_edmol_Idx = cur_edmol_Idx+1
            map_edmol_Idx_to_mol[cur_edmol_Idx] = leaf_atom
            
            edmol.AddAtom(mol.GetAtomWithIdx(leaf_atom))
            
            edmol_root_idx = int(np.argmax(map_edmol_Idx_to_mol==root_atom))
            
            edmol.AddBond(edmol_root_idx,cur_edmol_Idx,bond.GetBondType())
            
            
            
            
            
            leaf_bonds = mol.GetAtomWithIdx(leaf_atom).GetBonds()

            for leaf_bond in leaf_bonds:
                leaf_bond_idx = leaf_bond.GetIdx()
                                           
                p_bonds[leaf_bond_idx] = 1
               
                    
            # update p_bonds

            
        else:
            edmol.AddBond(int(np.argmax(map_edmol_Idx_to_mol==atom2_idx)),int(np.argmax(map_edmol_Idx_to_mol==atom1_idx)),bond.GetBondType())
            
        
        p_bonds[current_bond] =0
        
        rnd_walk = random.choice([True, False])
        if(rnd_walk or np.sum(p_bonds) ==1 ):
            current_bond = int(np.random.choice(params['max_bonds'], 1, p=p_bonds/np.sum(p_bonds))[0])
        else:
            
            found_rings = np.zeros(len(p_bonds))
            if(random.choice([True, False])):
                for bond_idx in range(0,len(p_bonds)):
                    if(p_bonds[bond_idx] ==1):
                        bond = mol.GetBondWithIdx(bond_idx)
                        atom1_idx = bond.GetBeginAtomIdx()
                        atom2_idx = bond.GetEndAtomIdx()
                        
                
                        p_atom = np.array([any(map_edmol_Idx_to_mol==atom1_idx)*1,any(map_edmol_Idx_to_mol==atom2_idx)*1])
                        
                        if(np.sum(p_atom) == 2):
                            found_rings[bond_idx] = 1
                        
            if(np.sum(found_rings)>0):
                current_bond = int(np.random.choice(params['max_bonds'], 1, p=found_rings/np.sum(found_rings))[0])
            else:
                # for each current bond. 
                bond_cicular_score = np.zeros(len(p_bonds))
                
                find_bonds =set()
                for bond_idx in range(0,len(p_bonds)):
                    if(p_bonds[bond_idx] == 1):
                        find_bonds.add(bond_idx)
                        
                NN = []
                #find distance to each dangling bond
                for atom_dis_idx in range(0,cur_edmol_Idx+1):
                    start_atom = mol.GetAtomWithIdx(int(map_edmol_Idx_to_mol[atom_dis_idx]))
                    bond_set = set([bond_lol.GetIdx() for bond_lol in start_atom.GetBonds()])
                    if(len(list(bond_set.intersection(find_bonds)))>0):
                        common_bonds = list(bond_set.intersection(find_bonds))
                        for common_bond in common_bonds: 
                            bond_cicular_score[common_bond]=bond_cicular_score[common_bond]+1
                    else:
                        searched_atoms = set([int(map_edmol_Idx_to_mol[atom_dis_idx])])
                        neighbors = set([neigh.GetIdx() for neigh in start_atom.GetNeighbors()])
                        for i in range(0,params['Num_Graph_Convolutions']+1):
                            neighbors_list= list(neighbors)
                            
                            new_neighbors = set()
                            for n in neighbors_list:
                                new_neighbors = new_neighbors.union([neigh.GetIdx() for neigh in mol.GetAtomWithIdx(n).GetNeighbors()])
                            searched_atoms = neighbors.union(searched_atoms)
                            neighbors = new_neighbors - searched_atoms
                            bond_set = set([])
                            for n in list(neighbors):                               
                                bond_set = bond_set.union( set([bond_lol.GetIdx() for bond_lol in mol.GetAtomWithIdx(n).GetBonds()]))
                            
                            common_bonds = list(bond_set.intersection(find_bonds))
                            if(len(common_bonds)>0):
                                for common_bond in common_bonds: 
                                    bond_cicular_score[common_bond]=bond_cicular_score[common_bond]+1
                                break
                        
                
               
                bond_cicular_score =   bond_cicular_score**2  
                
                current_bond = int(np.random.choice(params['max_bonds'], 1, p=bond_cicular_score/np.sum(bond_cicular_score))[0])
                            
                        

        edmol_list.append( Chem.EditableMol( copy.deepcopy(edmol.GetMol())))
        map_edmol_Idx_to_mol_list.append(copy.deepcopy(map_edmol_Idx_to_mol))
        cur_edmol_Idx_list.append(copy.deepcopy(cur_edmol_Idx))
        current_bond_list.append(copy.deepcopy(current_bond))
        P_remaining_DB.append(copy.deepcopy(p_bonds))
        
   
    return edmol_list,map_edmol_Idx_to_mol_list,cur_edmol_Idx_list, current_bond_list,P_remaining_DB          
                
                
                
    
def highlight_map_edmol(map_edmol_Idx_to_mol,cur_edmol_Idx,edmol,edmol_Frag,mol,bond):
    highlights =[]
    colors = {}
    highlightBonds=[]
    bond_colors={}
    root_atom_color = (139/250,69/250,19/250)
    root_bond_color = (205/250,133/250,63/250)
    leaf_color = (0,1,0)
    dangle_bond_color = (100/250,149/250,237/250)
    
    for bonds in edmol_Frag.GetMol().GetBonds():
        atom1_idx = bonds.GetBeginAtomIdx()
        atom2_idx = bonds.GetEndAtomIdx()
        if(map_edmol_Idx_to_mol[atom1_idx]!=-1 and map_edmol_Idx_to_mol[atom2_idx]!=-1):
            idx = mol.GetBondBetweenAtoms(int(map_edmol_Idx_to_mol[atom1_idx]),int(map_edmol_Idx_to_mol[atom2_idx])).GetIdx()
            highlightBonds.append(idx)
            bond_colors[idx] = dangle_bond_color
    for bonds in edmol.GetMol().GetBonds():
        atom1_idx = bonds.GetBeginAtomIdx()
        atom2_idx = bonds.GetEndAtomIdx()
        if(map_edmol_Idx_to_mol[atom1_idx]!=-1 and map_edmol_Idx_to_mol[atom2_idx]!=-1):
            idx = mol.GetBondBetweenAtoms(int(map_edmol_Idx_to_mol[atom1_idx]),int(map_edmol_Idx_to_mol[atom2_idx])).GetIdx()
            bond_colors[idx] = root_bond_color
            
            

    highlightBonds.append(bond.GetIdx())
    bond_colors[bond.GetIdx()] =(0,0,1)
        
        
    
    for i in range(0,len(map_edmol_Idx_to_mol)):
        if(int(map_edmol_Idx_to_mol[i])!=-1):
            highlights.append(int(map_edmol_Idx_to_mol[i]))
            colors[int(map_edmol_Idx_to_mol[i])] = root_bond_color
    colors[int(map_edmol_Idx_to_mol[cur_edmol_Idx])] =leaf_color
    
    
    return highlights,colors,highlightBonds,bond_colors


def ring_connection(mol,edmol,atom_idx,map_edmol_Idx_to_mol,params,leaf_atom_feature,required_NN,NN_Tensor_summed,root_atom,bond):
    connection_exists = 0
    valid_connection = 0
    mol_before_ring = copy.deepcopy(edmol.GetMol())
    if(int(np.argmax(map_edmol_Idx_to_mol==root_atom))!=atom_idx):
        if(all(leaf_atom_feature == gen_atom_features(mol_before_ring.GetAtomWithIdx(atom_idx),params))):
            
            #Check to see  if there is an avliable and compatiable bond between atoms
            atom = mol_before_ring.GetAtomWithIdx(atom_idx)
            NN = atom.GetNeighbors()
            edmol_NN_features = np.zeros(len(required_NN))
            
            
            for NN_atoms in NN:
                NN_example = np.concatenate((np.array([0]),gen_sparse_NN_feature(gen_atom_features(NN_atoms,params),gen_bond_features(mol_before_ring.GetBondBetweenAtoms(atom.GetIdx(),NN_atoms.GetIdx()),params)) ))
                edmol_NN_features = edmol_NN_features + NN_example
            
            from_ring_mol = copy.deepcopy(edmol.GetMol())
            from_ring_edmol = Chem.EditableMol(from_ring_mol)

          
            if(all((NN_Tensor_summed[map_edmol_Idx_to_mol[atom_idx]]-edmol_NN_features - required_NN)>=0) and (from_ring_mol.GetBondBetweenAtoms(int(np.argmax(map_edmol_Idx_to_mol==root_atom)),atom_idx) is None)):
                connection_exists = 1

                from_ring_edmol.AddBond(int(np.argmax(map_edmol_Idx_to_mol==root_atom)),atom_idx,bond.GetBondType())
                            
                mol_edited = from_ring_edmol.GetMol()
                matches = mol.GetSubstructMatches(mol_edited,uniquify=False)

                all_match = False
                    
                for match in matches:
                    all_match = True
                    for idx in range(0,len(match)):
                        if(any(NN_Tensor_summed[match[idx]] - NN_Tensor_summed[map_edmol_Idx_to_mol[idx]]!=0)):
                            all_match= False
                            break;
                    if(all_match):
                        valid_connection =1    
                        break;

    return [connection_exists,valid_connection]               

def possible_frag_root_matches(mol_frag,mol,NN_Tensor_summed,map_edmol_Idx_to_mol):
    matches = mol.GetSubstructMatches(mol_frag,uniquify=False)
    valid_matches = []
    for match in matches:
        match_found = True
        for sub_Idx in range(0,len(match)-1):
            if(not(all(NN_Tensor_summed[match[sub_Idx]] - NN_Tensor_summed[map_edmol_Idx_to_mol[sub_Idx]]==0))):
                match_found = False
                break
        
        if(match_found):
            valid_matches.append(match)
    
    
    
    
    return valid_matches




def check_frag_leaf_match(mol,matches,root_NN_feature,leaf_atom_feature,potential_leaf_atom_feature,NN_Tensor_slice,NN_Tensor_summed,rosetta_leaf_idx):  
    connection_exists = 0
    valid_connection = 0
    
    if(all(leaf_atom_feature - potential_leaf_atom_feature==0)):
        for edge_idx in range(0,len(  NN_Tensor_slice       )):
            if (all(root_NN_feature -NN_Tensor_slice[edge_idx] ==0)):
                connection_exists = 1
                                            
                for match in matches:
                    
                    
                    if(all(NN_Tensor_summed[match[rosetta_leaf_idx]] - np.sum(NN_Tensor_slice,axis=-2)==0)):
                        valid_connection =1
                        break
                break
    return [connection_exists, valid_connection]

def add_dangling_bonds(edmol2,NN_Tensor,map_edmol_Idx_to_mol,params,cur_edmol_Idx,edmol_root_idx,display_dangle,updated_atom,leaf_sparse_NN_feature,molSize):
    
    leaf_atom_solve_idx =-1
    mol = copy.deepcopy(edmol2.GetMol())
    edmol = Chem.EditableMol(mol)
    current_atom = mol.GetNumAtoms()
    
    NN_Tensor_summed = np.sum(NN_Tensor,axis=-2)
    dangling_atoms = np.zeros(params['max_dangle_atoms'])
    if(display_dangle):
        root_atom_color = (139/250,69/250,19/250)
        #if(valid_connection):
        root_bond_color = (205/250,133/250,63/250)
        #else:
           # root_bond_color = (1,128/250,114/250)
        dangle_atom_bond_color = (0,1,1)
        highlight=[cur_edmol_Idx,edmol_root_idx]
        colors={cur_edmol_Idx:(0,1,0),edmol_root_idx:root_atom_color}
        
        highlightBonds = []
        bond_colors = {}
        bond_HL_idx =0
    
        for bond_idx in range(0,mol.GetNumBonds()):
            highlightBonds.append(bond_idx)
            bond_colors[bond_idx] = root_bond_color
            bond_HL_idx =bond_HL_idx+1

    
    for atom_idx in range(0,mol.GetNumAtoms()):
        # for each atom add extra atoms that were not there before, 
        atom = mol.GetAtomWithIdx(atom_idx)
        root_idx = map_edmol_Idx_to_mol[atom_idx]
        NN_Tensor_slice = copy.deepcopy(NN_Tensor_summed[map_edmol_Idx_to_mol[atom_idx]]    )    
        for NN in atom.GetNeighbors(): 
            if(not(NN is None)):
                NN_feature = np.concatenate((np.array([0]),gen_sparse_NN_feature(gen_atom_features(NN,params),gen_bond_features(mol.GetBondBetweenAtoms(atom_idx,NN.GetIdx()),params))))
            
                NN_Tensor_slice = NN_Tensor_slice - NN_feature

                
        
        NN_Tensor_slice[0] = 0
        
        if(display_dangle):
            highlight.append(int(atom_idx))
            colors[int(atom_idx)]=root_bond_color
        
        while np.argmax(NN_Tensor_slice)!=0:
            if(leaf_sparse_NN_feature!=-1):
                if(not(updated_atom) and edmol_root_idx ==atom_idx and (np.argmax(NN_Tensor_slice) == (leaf_sparse_NN_feature+1))  ):
                    leaf_atom_solve_idx = current_atom
    
                    updated_atom = True

            sparse_NN_feature = np.argmax(NN_Tensor_slice)
            
            NN_Tensor_slice[sparse_NN_feature] = NN_Tensor_slice[sparse_NN_feature]-1
            
            atom_NN, bond_NN = atom_bond_from_sparse_NN_feature(int(sparse_NN_feature-1),params)
            edmol.AddAtom(atom_NN)
            #edmol.AddAtom(MolFromSmiles('I').GetAtomWithIdx(0))
            edmol.AddBond(atom_idx,current_atom,bond_NN)
            
            
            dangling_atoms[current_atom]=1
            
            if(display_dangle):
                highlight.append(int(current_atom))
                colors[current_atom]=dangle_atom_bond_color
                highlightBonds.append(bond_HL_idx)
                bond_colors[bond_HL_idx] = dangle_atom_bond_color
                bond_HL_idx =bond_HL_idx+1
            
            current_atom = current_atom+1
            
    if(display_dangle):           
        colors[cur_edmol_Idx]=(0,1,0)
        if(leaf_atom_solve_idx!=-1):
            colors[cur_edmol_Idx] =root_bond_color
            colors[leaf_atom_solve_idx] =(0,1,0)
            
            idx = edmol.GetMol().GetBondBetweenAtoms(edmol_root_idx, int(leaf_atom_solve_idx)).GetIdx()
        else:
        
            idx = edmol.GetMol().GetBondBetweenAtoms(cur_edmol_Idx,edmol_root_idx).GetIdx()
        colors[edmol_root_idx]=root_atom_color

        bond_colors[idx] =(0,0,1)
        mol_display = edmol.GetMol()
        Chem.RemoveHs(copy.deepcopy(mol_display),sanitize = False)
        img, svg =gen_img_svg_mol(mol_display,highlight=highlight,colors =colors,highlightBonds=highlightBonds,bond_colors=bond_colors,molSize=molSize)
        
        fp = BytesIO()

        svg2png(bytestring=svg,write_to=fp,scale=1)
        return edmol,dangling_atoms,leaf_atom_solve_idx, fp
    else:
        return edmol,dangling_atoms,leaf_atom_solve_idx

def gen_train_mol_tensor(edmol_frag,NN_Tensor,map_edmol_Idx_to_mol,params,cur_edmol_Idx =0,edmol_root_idx =0,display_dangle=False,updated_atom=False,leaf_sparse_NN_feature=-1,molSize=(450,150),dB_degree_mask = True):
    
    if(display_dangle):
        edmol_DB,dangling_atoms,atom_leaf_solving_idx,fp = add_dangling_bonds(edmol_frag,NN_Tensor,map_edmol_Idx_to_mol,params,cur_edmol_Idx,edmol_root_idx,display_dangle,updated_atom,leaf_sparse_NN_feature,molSize)
    else:    
        edmol_DB,dangling_atoms,atom_leaf_solving_idx= add_dangling_bonds(edmol_frag,NN_Tensor,map_edmol_Idx_to_mol,params,cur_edmol_Idx,edmol_root_idx,display_dangle,updated_atom,leaf_sparse_NN_feature,molSize)
    
    mol_DB = edmol_DB.GetMol()
    
    
    '''
    solving_bond =-2
    suggested_bond = -2
    if(updated_atom):
        suggested_bond = mol_DB.GetBondBetweenAtoms(cur_edmol_Idx,int(np.argmax(root_atom==map_edmol_Idx_to_mol))).GetIdx()
    else:
        solving_bond = mol_DB.GetBondBetweenAtoms(int(np.argmax(root_atom==map_edmol_Idx_to_mol)), int(atom_leaf_solving_idx)).GetIdx()
    '''
    #atoms, edges,bonds = mol_to_tensor(mol_DB, params,solving_bond = solving_bond,suggested_bond=suggested_bond, FHO_Ring_feature=True)
    atoms, edges,bonds = mol_to_tensor(mol_DB, params, FHO_Ring_feature=True,dangling_atoms = dangling_atoms,find_dangle_bonds=True)
    if(dB_degree_mask):

        mask_symmetric_atoms, mask_symmetric_bonds = DB_degree_masks(mol_DB, dangling_atoms,edges,params)
        #print(np.reshape(mask_symmetric_atoms,(7,60)))
        # print(np.reshape(mask_symmetric_bonds,(7,60,4))[1])
        #display(PIL.Image.open(fp))
        if(display_dangle):
            return atoms, edges,bonds,dangling_atoms,fp,mask_symmetric_atoms, mask_symmetric_bonds
        else:
            return atoms, edges,bonds,dangling_atoms,mask_symmetric_atoms, mask_symmetric_bonds
        
    else:
        if(display_dangle):
            return atoms, edges,bonds,dangling_atoms,fp
        else:
            return atoms, edges,bonds,dangling_atoms


def DB_degree_masks(mol, dangling_atoms,edge,params):
    
    new_mask_atoms = np.ones(params['max_dangle_atoms'])
    new_mask_bonds = np.ones((params['max_dangle_atoms'],params['max_degree']))
    
    mask_symmetric_atoms  = []
    mask_symmetric_bonds  = []
    
    NN = []
    for  atom_idx in range(0, len(dangling_atoms)):
        if(dangling_atoms[atom_idx]==1):
            NN.append(atom_idx)
            new_mask_atoms[atom_idx]  = 0.0
            new_mask_bonds[atom_idx,:] = 0.0
            
            for edge_idx in range(0,params['max_degree']):
                if(edge[atom_idx,edge_idx]!=-1):
                    sub_idx = np.argmax((edge[edge[atom_idx,edge_idx]] == atom_idx)*1)
                    new_mask_bonds[edge[atom_idx,edge_idx],sub_idx] = 0.0
                    
    mask_symmetric_atoms.append(new_mask_atoms)          
    mask_symmetric_bonds.append(new_mask_bonds)
    NN = set(NN)
    new_NN = set(NN)
    new_NN_list = list(new_NN)
    for convo_idx in range(1,params['Num_Graph_Convolutions']): 
        new_mask_atoms = np.ones(params['max_dangle_atoms'])
        new_mask_bonds = np.ones((params['max_dangle_atoms'],params['max_degree']))
        
        
        added_NN = set([])
        for NN_idx in new_NN_list:
            added_NN = added_NN.union([neigh.GetIdx() for neigh in mol.GetAtomWithIdx(NN_idx).GetNeighbors()])
        
        new_NN = added_NN.difference(NN)
        new_NN_list = list(new_NN)
        NN = NN.union(added_NN)
        
        for atom_idx in new_NN_list:
            new_mask_atoms[atom_idx]  = 0.0
            new_mask_bonds[atom_idx,:] = 0.0

            for edge_idx in range(0,params['max_degree']):
                if(edge[atom_idx,edge_idx]!=-1):
                    sub_idx = np.argmax((edge[edge[atom_idx,edge_idx]] == atom_idx)*1)
                    new_mask_bonds[edge[atom_idx,edge_idx],sub_idx] = 0.0
        mask_symmetric_atoms.append(mask_symmetric_atoms[convo_idx-1]*new_mask_atoms)
        mask_symmetric_bonds.append(mask_symmetric_bonds[convo_idx-1]*new_mask_bonds)
    
    mask_symmetric_atoms = np.array(mask_symmetric_atoms)
    mask_symmetric_atoms = np.reshape(mask_symmetric_atoms, (params['Num_Graph_Convolutions'],params['max_dangle_atoms'],1))
    mask_symmetric_bonds = np.array(mask_symmetric_bonds)
    mask_symmetric_bonds = np.reshape(mask_symmetric_bonds,(params['Num_Graph_Convolutions'],params['max_dangle_atoms']*params['max_degree'],1))
    return mask_symmetric_atoms, mask_symmetric_bonds
    
    

def gen_sparse_random_NN(num_features,params):
    
    
    p_sparse = np.ones(num_features+1)
    p_sparse[0] = 0
    p_sparse = p_sparse/(num_features)
    num_NN = np.array(range(0,params['max_degree']))
    p_edge = np.ones(params['max_degree'])/(params['max_degree'])
    
    NN = []
    num_edges = np.random.choice(params['max_degree'], 1, p=p_edge)[0]
    max_edge = num_NN[num_edges]
    
    for edge in range(0,params['max_degree']):
        zeros =  np.zeros(num_features+1)

        if(edge <=max_edge and edge!=0 ):
            zeros[np.random.choice(num_features+1, 1, p=p_sparse)[0]] =1
        elif(edge > max_edge):
            zeros[0] = 1
            
        NN.append(zeros)
        
    return np.array(NN)
    
    
       
def check_merge(mol, part,num_neighbours,current_atom):
    
    
    correct_merge = 0
    for match in matches:
        if(len(mol.GetAtomWithIdx(match[current_atom]).GetNeighbors())==num_neighbours):
            correct_merge = 1
            break
        
    return correct_merge
       


def remove_charge(mol):
    edmol = Chem.EditableMol(mol)
    for atom_idx in range(0,mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetFormalCharge(0)
        edmol.ReplaceAtom(atom_idx, atom)
    mol = edmol.GetMol()
    return mol




