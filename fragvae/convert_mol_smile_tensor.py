# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:54:00 2019

@author: ja550
"""
from __future__ import print_function
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem as Chem
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import random
import copy 
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import time
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

def tensorise_smiles(data, params, printMe = False):
    
    
    
    
    molecules_atom_tensor = []
    molecules_edge_tensor = []
    molecules_bond_tensor = []
    for i in range(len( data['smiles'])):
        
        
        atom_matrix, edge_matrix,bond_tensor =smile_to_tensor( data["smiles"][i], params)
        if not atom_matrix.any():
            print('failed to tensorize '+ data['smiles'][i] + ' at index ' + str(i))
        else:
            if(printMe):
                print(data['smiles'][i] + ' at index ' + str(i))
            molecules_atom_tensor.append(atom_matrix)
            molecules_edge_tensor.append(edge_matrix)
            molecules_bond_tensor.append(bond_tensor)
    return np.array(molecules_atom_tensor), np.array(molecules_edge_tensor), np.array(molecules_bond_tensor)
    
    # Tensorise data
    # atom matrix, size: (max_atoms, num_atom_features) This matrix defines the atom features.
def smile_to_tensor(smiles, params,FHO_Ring_feature=False,find_dangle_bonds = False):
    mol = smile_to_mol(smiles, params)
    
    atom_matrix, edge_matrix,bond_tensor = mol_to_tensor(mol, params,FHO_Ring_feature=FHO_Ring_feature,find_dangle_bonds=find_dangle_bonds) 
            
            
    return atom_matrix, edge_matrix,bond_tensor
    '''
    except: 
        x = np.array([])
        
        return x,x,x 
    '''
    
def smile_to_mol(smiles, params):
    mol = Chem.MolFromSmiles(smiles)
    reshuffle_atms = list(range(mol.GetNumAtoms()))
    
    if(params["random_shuffle"]):
        random.shuffle(reshuffle_atms)
    mol = Chem.RenumberAtoms(mol,reshuffle_atms)
    
    if(params['max_dangle_atoms']>0):
        max_atoms = params['max_dangle_atoms']
    else:
        max_atoms = params['max_atoms']
    non_aromatic_set = find_custom_Kekulize_set(mol, max_atoms,  params['max_degree'],printMe = False)
    
    if(params['KekulizeBonds']):
        Chem.Kekulize(mol)
    else:
        mol = custom_kekulize(mol,non_aromatic_set)
    
    return mol
    
    
def mol_to_tensor(mol, params,solving_bond=-1,suggested_bond=-1,FHO_Ring_feature=False,dangling_atoms = -1,find_dangle_bonds = False):    

    '''
    atom matrix, size: (max_atoms, num_atom_features) This matrix defines the atom features.

    Each column in the atom matrix represents the feature vector for the atom at the index of that column.
    '''
    
    if(params['max_dangle_atoms']>0):
        max_atoms = params['max_dangle_atoms']
    else:
        max_atoms = params['max_atoms']
    

     
    
        
    
    atom_matrix = np.array([[0 for i in range(params['num_atom_features'])] for j in range(max_atoms)])
    
    
    '''
    edge matrix, size: (max_atoms, max_degree) This matrix defines the connectivity between atoms.
    
    Each column in the edge matrix represent the neighbours of an atom. The neighbours are encoded by an integer representing the index of their feature vector in the atom matrix.
    
    As atoms can have a variable number of neighbours, not all rows will have a neighbour index defined. These entries are filled with the masking value of -1. (This explicit edge matrix masking value is important for the layers to work)
    
    '''
    edge_matrix = np.array([[-1 for i in range(params['max_degree'])] for j in range(max_atoms)])
    
    '''
       
    bond tensor size: (max_atoms, max_degree, num_bond_features) This matrix defines the atom features.
    
    The first two dimensions of this tensor represent the bonds defined in the edge tensor. The column in the bond tensor at the position of the bond index in the edge tensor defines the features of that bond.
    
    Bonds that are unused are masked with 0 vectors.
    '''
    bond_tensor = np.array([[[0 for k in range(int(FHO_Ring_feature)+int(solving_bond!=-1)+int(find_dangle_bonds)+int(suggested_bond!=-1)+params['num_bond_features'])] for i in range(params['max_degree'])] for j in range(max_atoms)])
    
    #try:
    
    
    
    
    
    min_atoms = np.minimum(max_atoms,mol.GetNumAtoms())
    
    for atom_index in range(0,min_atoms):
        atom = mol.GetAtomWithIdx(atom_index)
        #IsAromatic = int(atom.GetIsAromatic())
        

               
        

        atom_matrix[atom.GetIdx()][0:params['num_atom_features']] = gen_atom_features(atom,params)
        if(params['printMe']):
            print('atom Symbol: ' +str(atom.GetSymbol()))
            print('atom Degree: ' +str(atom.GetDegree()))
            print('atom GetTotalNumHs: ' +str(atom.GetTotalNumHs()))
            print('atom GetImplicitValence: ' +str(atom.GetImplicitValence()))
            print('atom GetIsAromatic: ' +str(bool(IsAromatic)))
            print('')
    min_bonds = np.minimum(max_atoms,len(mol.GetBonds()))
    for bond_index in range(0,min_bonds):
        
        
        bond = mol.GetBonds()[bond_index]
        atom1_Idx = bond.GetBeginAtom().GetIdx()
        atom2_Idx = bond.GetEndAtom().GetIdx()
        
        if(find_dangle_bonds):
            dangle_bond  = dangling_atoms[atom1_Idx] or  dangling_atoms[atom2_Idx]
        else:
            dangle_bond =-1
        bond_features = gen_bond_features(bond,params,FHO_Ring_control=FHO_Ring_feature,solving_bond=solving_bond,suggested_bond=suggested_bond,dangle_bond=dangle_bond)

            
        if(params['printMe']):
            print('bond type: '+str(bond.GetBondType()))
            print('bond in ring: ' + str(bond.IsInRing()))
            print('bond is conjugated: ' + str(bond.GetIsConjugated()))
            print('')
        
        i = 0
        while i < params['max_degree']:
            if(edge_matrix[atom1_Idx,i]==-1):
                edge_matrix[atom1_Idx][i] =atom2_Idx
                bond_tensor[atom1_Idx][i][0:len(bond_features)] = bond_features
                i = params['max_degree']*2+100
            i = i+1
        i = 0
        while i < params['max_degree']:
            if(edge_matrix[atom2_Idx,i]==-1):
                edge_matrix[atom2_Idx][i] =atom1_Idx
                bond_tensor[atom2_Idx][i][0:len(bond_features)] = bond_features
                i = params['max_degree']*2+100
            i = i+1
            
            
            
            
    return atom_matrix, edge_matrix,bond_tensor
    

def gen_atom_features(atom,params):
    
    atom_Features =  np.array([s == atom.GetSymbol()  for s in params["atoms"]])
    if(params['include_charges']):
        atom_Features = np.concatenate((atom_Features,np.array([s == atom.GetFormalCharge()  for s in params["charges"]])))
    if(params['include_degrees']):
        atom_Features = np.concatenate((atom_Features,np.array([s == atom.GetDegree()  for s in params["degrees"]])))
    if(params['include_valence']):
        atom_Features = np.concatenate((atom_Features,np.array([s == atom.GetImplicitValence()  for s in params["valence"]])))

    atom_Features = atom_Features*1
    '''
    atom_Features =  np.concatenate([    atom_Features,np.array([s == atom.GetTotalNumHs() for s in [0, 1, 2, 3, 4]])*1,
                                                 np.array([s == atom.GetImplicitValence() for s in [0, 1, 2, 3, 4, 5]])*1,
                                                 np.array([s == atom.GetDegree() for s in [0, 1, 2, 3, 4, 5]])*1,
                                                 np.array([IsAromatic]),np.array([atom.GetFormalCharge()])])
            if(EtraBond_feature):
                atom_Features =  np.concatenate([    atom_Features,np.array([atom.GetTotalValence()- atom.GetTotalDegree()])])
    '''
    return atom_Features


def gen_bond_features(bond,params,FHO_Ring_control=False,solving_bond=-1,suggested_bond=-1,dangle_bond=-1):
    
    bond_Features = np.array([s == bond.GetBondType() for s in params['bonds']])*1

    if(solving_bond!=-1):
        bond_Features = np.concatenate((bond_Features,np.array([bond.GetIdx() == solving_bond],float)))
    if(suggested_bond!=-1):
        bond_Features = np.concatenate((bond_Features,np.array([bond.GetIdx() == suggested_bond],float)))
    if(dangle_bond!=-1):
        bond_Features = np.concatenate((bond_Features,np.array([dangle_bond == 1],float)))
    if(FHO_Ring_control==True):
        bond_Features = np.concatenate((bond_Features,np.array([bond.IsInRing() == True],float)))
    return bond_Features
    '''
    if(not(min_features)):
        bond_features = np.concatenate([bond_features,np.array([bond.GetIsConjugated(), bond.IsInRing()])*1])
    return bond_Features
    '''

def find_custom_Kekulize_set(mol,  max_atoms,  max_degree,printMe = False):
    '''
    rdkit does not correclty label molecules such as TCNQ as nonaromatic 
    
    check_force_Kekulize determines if each ring in the molecules is aromatic
    if one of the rings which claims and is possible to be aromatic is not 
    aromatic.  
    
    
    
    
    '''
    
    mol = copy.deepcopy(mol)  
    Chem.Kekulize(mol)

    mol_rings = mol.GetRingInfo().AtomRings()
    
    non_aromatic_ring_set = set([])
    other_ring = set([])
    for ring in mol_rings:
        
        #check if all atoms in ring are considered aromatic
        if(isEven(len(ring))):
            is_aromatic = True
            for atom_IDX in ring:
                if( not(mol.GetAtomWithIdx(atom_IDX).GetIsAromatic())):
                    is_aromatic = False
                    
                    
            if(is_aromatic):
                # check is the ring is conjugated
                # check if the first bond in the ring is double or single
                consitent = True
                lenRing = len(ring)
                skip = False
                index = 0
                while index < lenRing and not(skip):
                    atom_IDX = ring[index]
                    for Neighbor_atom in mol.GetAtomWithIdx(atom_IDX).GetNeighbors():
                        
                        if(Chem.rdchem.BondType.DOUBLE == mol.GetBondBetweenAtoms(atom_IDX,Neighbor_atom.GetIdx()).GetBondType() and not( Neighbor_atom.IsInRing())):            
                            if(Neighbor_atom.GetNeighbors()==1 ):
                                consitent = False
                                skip = True
                                break
                            #check if any of the next neighbours are in rings
                            isNextNeighbourAromatic = False
                            for next_Neighbor in Neighbor_atom.GetNeighbors():
                                if(next_Neighbor.GetIdx()!=atom_IDX): 
                                    if(next_Neighbor.GetIsAromatic()):
                                        isNextNeighbourAromatic = False
                                    
                            if(not(isNextNeighbourAromatic)):
                                consitent = False
                                skip = True
                                break
                    index = index +1
                if(consitent):
                    
                    other_ring = set(ring)|other_ring
                else:
                    
                    non_aromatic_ring_set = set(ring)|non_aromatic_ring_set
        
            else:
                other_ring = set(ring)|other_ring
    
    #find atoms that are only in non_aromatic_rings
                                                        
            
    return non_aromatic_ring_set  -other_ring    







def gen_NN_Tensor(atoms,edges,bonds):
  
    '''
    atom matrix, size: (max_atoms, num_atom_features) This matrix defines the atom features.

    Each column in the atom matrix represents the feature vector for the atom at the index of that column.
    '''
    #atom_matrix = np.array([[0 for i in range(params['num_atom_features'])] for j in range(params['max_atoms'])])
    
    
    '''
    edge matrix, size: (max_atoms, max_degree) This matrix defines the connectivity between atoms.
    
    Each column in the edge matrix represent the neighbours of an atom. The neighbours are encoded by an integer representing the index of their feature vector in the atom matrix.
    
    As atoms can have a variable number of neighbours, not all rows will have a neighbour index defined. These entries are filled with the masking value of -1. (This explicit edge matrix masking value is important for the layers to work)
    
    '''
    #edge_matrix = np.array([[-1 for i in range(params['max_degree'])] for j in range(params['max_atoms'])])
    
    '''
       
    bond tensor size: (max_atoms, max_degree, num_bond_features) This matrix defines the atom features.
    
    The first two dimensions of this tensor represent the bonds defined in the edge tensor. The column in the bond tensor at the position of the bond index in the edge tensor defines the features of that bond.
    
    Bonds that are unused are masked with 0 vectors.
    '''
    #    bond_tensor = np.array([[[0 for k in range(params['num_bond_features'])] for i in range(params['max_degree'])] for j in range(params['max_atoms'])])
    
    '''
       
    NN tensor size: (max_nodes, max_edge, num_atom_features+bond features) This matrix defines the atom features.
    
    The first two dimensions of this tensor represent atoms connected to a set node. The column in the nearest neighbour tensor at the position of the bond index in the edge tensor defines the features of that bond.
    
    Bonds that are unused are masked with 0 vectors.
    '''
    #NN_tensor = np.array([[[0 for k in range(num_atom_features)] for i in range(max_edge)] for j in range(max_nodes)])
    NN_tensor = np.array([[[0 for k in range(atoms.shape[-1]+bonds.shape[-1]+1)] for i in range(bonds.shape[-2])] for j in range(atoms.shape[-2])])
    
    one = np.concatenate((np.array([1]),np.zeros(bonds.shape[-1]+atoms.shape[-1])))
    zero = np.array([0])
    for atom_idx in range(0,atoms.shape[-2]):
        for edge_idx in range(0,edges.shape[-1]):
            if(edges[atom_idx][edge_idx]!=-1):
                
                atom_features = atoms[edges[atom_idx][edge_idx]]
                bond_features = bonds[atom_idx][edge_idx]
                NN_tensor[atom_idx][edge_idx] =  np.concatenate((zero,atom_features,bond_features))
            else:

                NN_tensor[atom_idx][edge_idx] = one
                

            
            
    return NN_tensor


def gen_sparse_NN_Tensor(atoms,edges,bonds):
  
    '''
    atom matrix, size: (max_atoms, num_atom_features) This matrix defines the atom features.

    Each column in the atom matrix represents the feature vector for the atom at the index of that column.
    '''
    #atom_matrix = np.array([[0 for i in range(params['num_atom_features'])] for j in range(params['max_atoms'])])
    
    
    '''
    edge matrix, size: (max_atoms, max_degree) This matrix defines the connectivity between atoms.
    
    Each column in the edge matrix represent the neighbours of an atom. The neighbours are encoded by an integer representing the index of their feature vector in the atom matrix.
    
    As atoms can have a variable number of neighbours, not all rows will have a neighbour index defined. These entries are filled with the masking value of -1. (This explicit edge matrix masking value is important for the layers to work)
    
    '''
    #edge_matrix = np.array([[-1 for i in range(params['max_degree'])] for j in range(params['max_atoms'])])
    
    '''
       
    bond tensor size: (max_atoms, max_degree, num_bond_features) This matrix defines the atom features.
    
    The first two dimensions of this tensor represent the bonds defined in the edge tensor. The column in the bond tensor at the position of the bond index in the edge tensor defines the features of that bond.
    
    Bonds that are unused are masked with 0 vectors.
    '''
    #    bond_tensor = np.array([[[0 for k in range(params['num_bond_features'])] for i in range(params['max_degree'])] for j in range(params['max_atoms'])])
    
    '''
       
    NN tensor size: (max_nodes, max_edge, num_atom_features+bond features) This matrix defines the atom features.
    
    The first two dimensions of this tensor represent atoms connected to a set node. The column in the nearest neighbour tensor at the position of the bond index in the edge tensor defines the features of that bond.
    
    Bonds that are unused are masked with 0 vectors.
    '''
    #NN_tensor = np.array([[[0 for k in range(num_atom_features)] for i in range(max_edge)] for j in range(max_nodes)])
    NN_tensor = np.array([[[0 for k in range(atoms.shape[-1]*bonds.shape[-1]+1)] for i in range(bonds.shape[-2])] for j in range(atoms.shape[-2])])

    one = np.concatenate((np.array([1]),np.zeros(atoms.shape[-1]*bonds.shape[-1])))
    zero = np.array([0])
    for atom_idx in range(0,atoms.shape[-2]):
        for edge_idx in range(0,edges.shape[-1]):
            if(edges[atom_idx][edge_idx]!=-1):
                
                
                atom_features = atoms[edges[atom_idx][edge_idx]]
                bond_features = bonds[atom_idx][edge_idx]
                NN_feature = gen_sparse_NN_feature(atom_features,bond_features)

                NN_tensor[atom_idx][edge_idx] =  np.concatenate((zero,NN_feature))
            else:

                NN_tensor[atom_idx][edge_idx] = one
                

            
            
    return NN_tensor



def gen_sparse_NN_feature(atom_features,bond_features):
    
    NN_feature = np.zeros(atom_features.shape[-1]*bond_features.shape[-1])
    atom_feature_num = np.argmax(atom_features)
    bond_feature_num =  np.argmax(bond_features)
    NN_feature[bond_feature_num*atom_features.shape[-1]+atom_feature_num] =1
    
    return NN_feature

def atom_bond_from_sparse_NN_feature(sparse_NN_feature,params):
    
    bond_feature_num = int(sparse_NN_feature/params['num_atom_features'])
    atom_feature_num = int(sparse_NN_feature-bond_feature_num*params['num_atom_features'])
    if(params['atoms'][atom_feature_num]=='H'):
        mol = Chem.MolFromSmiles('[H]')
    else:
        mol = Chem.MolFromSmiles(params['atoms'][atom_feature_num])
    atom = mol.GetAtomWithIdx(0)
    
        
    bond = params["bonds"][bond_feature_num]
    return atom, bond 

def bond_from_bond_feature(bond_feature_num,params):
    
       
    bond = params["bonds"][bond_feature_num]
    return  bond 

def atom_from_atom_feature(atom_feature_num,params):
    
    if(params['atoms'][atom_feature_num]=='H'):
        mol = Chem.MolFromSmiles('[H]')
    else:
        mol = Chem.MolFromSmiles(params['atoms'][atom_feature_num])
    atom = mol.GetAtomWithIdx(0)
    
        
    return atom 

def isEven(num):
    return not(num%2)


def test_force_Kekulize():
    df = pd.read_csv('All_Moles_Tested_Data.csv')
    i= 0
    mol_list = []
    for smile in df['smiles']:
        mol = MolFromSmiles(smile)
        x = find_custom_Kekulize_set(smile,  max_atoms= 60,  max_degree= 5,printMe = False)
        for index in x:
            mol.GetAtomWithIdx(index).SetAtomicNum(32)
            
        mol_list.append(mol)
        
    df['mol'] = pd.DataFrame({'mol':mol_list})
    
    unit = 5
    for i in range(0,len(df)//unit):
        display(PandasTools.FrameToGridImage(df.iloc[i*unit:i*unit+unit],column='mol', legendsCol='',molsPerRow=unit))
    if((len(df)%unit>0)*1):
        display(PandasTools.FrameToGridImage(df.iloc[len(df)//unit*unit:len(df)],column='mol', legendsCol='',molsPerRow=unit))
#test_force_Kekulize()
        

def custom_kekulize(mol,non_aromatic_atoms):
    edmol = Chem.EditableMol(mol)
    kekmol = copy.deepcopy(mol)  
    Chem.Kekulize(kekmol)
    
    
    for atom_idx in list(non_aromatic_atoms):
        atom_idx = int(atom_idx)
        edmol.ReplaceAtom(atom_idx,kekmol.GetAtomWithIdx(atom_idx))
        bonds = mol.GetAtomWithIdx(atom_idx).GetBonds()
        

        
        
        for bond in bonds:
            edmol.ReplaceBond(bond.GetIdx(),kekmol.GetBondWithIdx(bond.GetIdx()))
    kekulize_mol = edmol.GetMol()
    return kekulize_mol
    
    
    
    
def test_custom_kekulize():
    smiles = 'CC=C1c2ccccc2C(=CC)c3ccccc13'
    smiles = 'N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2'
    mol = MolFromSmiles(smiles)
    
    display(mol)
    for atom_idx in range(0,mol.GetNumAtoms()):
       bonds =  mol.GetAtomWithIdx(atom_idx).GetBonds()
       for bond in bonds:
           print(bond.GetBondType())
           
        
    non_aromatic_atoms = find_custom_Kekulize_set(mol,  60,  5)
    
    mol = custom_kekulize(mol,non_aromatic_atoms)
    
    display(mol)
    for atom_idx in range(0,mol.GetNumAtoms()):
       bonds =  mol.GetAtomWithIdx(atom_idx).GetBonds()
       for bond in bonds:
           print(bond.GetBondType())
    
    
    
    



    

    
    
    
    