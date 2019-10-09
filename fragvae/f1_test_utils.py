# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:33:10 2019

@author: ja550
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:43:48 2019

@author: ja550
"""
from utils import *
from convert_mol_smile_tensor import *
import keyboard
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles, rdDepictor
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import * 
from IPython.display import SVG
import pandas as pd
import numpy as np
from load_model_parameters import *
import matplotlib.image as mpimg
from cairosvg import svg2png
import imageio
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from cairosvg import svg2png
import imageio
import datetime
import ctypes
from io import BytesIO
import copy 
from PIL import Image, ImageDraw,ImageFont,ImageFont


def test_F1_training_model(model,params):
    '''
    function to visualize the decoding of all fragments compared to ground truth of decoding
    
    iputs
        - predicitive training model as input
    
    '''
    params["random_shuffle"] = False
    params['gifMe'] = True
    params['max_atoms'] = 60
    params['batch_size'] = 1
    gif_counter = 0
    g = data_generator(params,return_smiles = True)
    data = next(g)[0]

    
    mol_atoms =    data['atoms']
    mol_edge = data['edges']
    mol_bonds = data['bonds']
    mol_NN_Tensor = data['NN_Tensor'] 
    total_error_in = data['dummy_error_output']

    
    
    out_model= model.predict([mol_atoms,mol_edge,mol_bonds,mol_NN_Tensor,total_error_in])
    [NN_prob,match,N_prob,AI_N,AI_NN,selected_NN,selected_N,AI_N_out,AI_NN_out,N_match_out] = out_model
    
    
    
        
    select_me = ['0']+['-'+i for i in params["atoms"]]+['='+i for i in params["atoms"]]+['#'+i for i in params["atoms"]]+['≈'+i for i in params["atoms"]]
    select_atoms = ['0']+params["atoms"]
    
    
    orange = (255/255,165/255,0)
    single_space = 26
    quarter_pic_size = (675, 337-single_space*2)
    text_size = (675, single_space*2)
    full_quarter_pic_size = (675, 337)
    font_size = 22
        
    atoms =mol_atoms[0]
    edges = mol_edge[0]
    bonds = mol_bonds[0]
    NN_Tensor = mol_NN_Tensor[0]
    
    
    
    mol_idx =0
    [NN_prob,match,N_prob,AI_N,AI_NN,selected_NN,selected_N,AI_N_out,AI_NN_out,N_match_out] = out_model
    [NN_prob,match,N_prob,AI_N,AI_NN,selected_NN,selected_N,AI_N_out,AI_NN_out,N_match_out] =[NN_prob[mol_idx],match[mol_idx],N_prob[mol_idx],AI_N[mol_idx],AI_NN[mol_idx],selected_NN[mol_idx],selected_N[mol_idx],AI_N_out[mol_idx],AI_NN_out[mol_idx],N_match_out[mol_idx]]
    
    smiles = data['smiles'][0]
    print(smiles)
    #img = rdkit.Chem.Draw.MolsToImage([Chem.MolFromSmiles(smiles)])
    #display(img)

    
    m = Chem.MolFromSmiles(smiles)

    highlight=[]
    colors={}
    num_atoms = m.GetNumAtoms()


    
    
    print('num atoms '+str(num_atoms))
    selected_F1 = np.zeros(num_atoms)
    if(num_atoms>=-1):
        for atom_idx in range(0, num_atoms):  
            #plt.figure(figsize=(20,11))
            
            display_F1 = np.ones(num_atoms) - selected_F1
            flash_F1 =display_F1*0
            bag_frags = display_F1_graphs(params,m.GetNumAtoms()-1,display_F1,flash_F1, quarter_pic_size,atoms, NN_Tensor)
            selected_frags = display_F1_graphs(params,num_atoms,selected_F1,flash_F1,quarter_pic_size,atoms, NN_Tensor)
    
    
            text_selected_frags= Image.new('RGB', text_size, color = (255, 255, 255))
            d = ImageDraw.Draw(text_selected_frags)
            d.text((0,0), "Molecular structure with previously sampled frags (top, red)", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            d.text((0,single_space), "Bag of previously sampled molecular frags (bottom)", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            vstack_img(text_selected_frags,selected_frags)        
            
            plt.figure(figsize=(quarter_pic_size[0]/100,(quarter_pic_size[1]+text_size[1])/100), dpi=100)
            plt.rcParams.update({'font.size': 13})
            plt.gcf().subplots_adjust(bottom=0.15)
            axes = plt.gca()
            axes.set_ylim([0,1])
    
                        
            
            
            plt.bar(select_atoms, AI_N[atom_idx], width=0.8, bottom=None,alpha = 0.5,color ='r',label= 'AI_P(n)')
            plt.bar(select_atoms, N_prob[atom_idx], width=0.8, bottom=None,alpha = 0.5,color ='c',label= 'P(n)')
            plt.bar(select_atoms, N_prob[atom_idx]*0, width=0.8, bottom=None,alpha = 0.5,color =orange,label= 'Sampling')
            plt.ylabel('Prob. of sampling a specific node',wrap=True)
            plt.xlabel('Possible nodes to sample',wrap=True)
            '''
            plt.title('P(atom|Fingerprint - '+str(atom_idx)+' atoms) AI Selected [ '+
                          select_atoms[np.argmax(AI_N_out[atom_idx])]+' ], Training Selected [ '+
                          select_atoms[np.argmax(selected_N[atom_idx])]+' ]')
            '''
            plt.title('P(Node|Z , Z_sampled_frags)',wrap=True)
        
        
            plt.legend()
            
            P_F1_N = BytesIO()
            plt.savefig(P_F1_N, dpi=100)
            P_F1_N = PIL.Image.open(P_F1_N)
            plt.close()
    
            select_bar_mag = np.max([AI_N[atom_idx][int(np.argmax(selected_N[atom_idx]))],N_prob[atom_idx][int(np.argmax(selected_N[atom_idx]))]])
            temp_AI = copy.deepcopy(AI_N[atom_idx])
            temp_AI[int(np.argmax(selected_N[atom_idx]))] =0
            
            temp_N_prob = copy.deepcopy(N_prob[atom_idx])
            temp_N_prob[int(np.argmax(selected_N[atom_idx]))] =0
            select_bar = copy.deepcopy(N_prob[atom_idx])*0.0
            select_bar[int(np.argmax(selected_N[atom_idx]))] =select_bar_mag
            
            plt.figure(figsize=(quarter_pic_size[0]/100,(quarter_pic_size[1]+text_size[1])/100), dpi=100)
            plt.bar(select_atoms, temp_AI, width=0.8, bottom=None,alpha = 0.5,color ='r',label= 'AI_P(n)')
            plt.bar(select_atoms, temp_N_prob, width=0.8, bottom=None,alpha = 0.5,color ='c',label= 'P(n)')
            plt.bar(select_atoms, select_bar, width=0.8, bottom=None,color =orange,label= 'Sampling')
            plt.ylabel('Prob. of sampling a specific node',wrap=True)
            plt.xlabel('Possible nodes to sample',wrap=True)
            plt.title('P(Node|Z , Z_sampled_frags',wrap=True)
            axes = plt.gca()
            axes.set_ylim([0,1])
            plt.legend()
            plt.gcf().subplots_adjust(bottom=0.15)
            P_F1_flash = BytesIO()
            plt.savefig(P_F1_flash, dpi=100)
            P_F1_flash = PIL.Image.open(P_F1_flash)
            plt.gcf().subplots_adjust(bottom=0.05)
            plt.close()
            
            #print('press space to continue...')
            #keyboard.wait(' ') 
            
            
            flash_F1 = np.zeros(num_atoms)
            next_display_F1 = copy.deepcopy(display_F1)
            for atom_idx2 in range(0,num_atoms):
                if(selected_F1[atom_idx2]==0):
                    if(all(atoms[atom_idx2] == atoms[int(match[atom_idx])]) ):
                        flash_F1[atom_idx2] =1
                    else:
                        next_display_F1[atom_idx2] =0
                else:
                    next_display_F1[atom_idx2] =0
            
            
            img_mol, svg = gen_img_svg_mol(m,highlight=highlight,colors=colors,highlightBonds=[],bond_colors={},molSize=quarter_pic_size)
            fp = BytesIO()
            svg2png(bytestring=svg,write_to=fp,scale=1)
            img_mol = PIL.Image.open(fp)
            
            text_img_mol= Image.new('RGB', text_size, color = (255, 255, 255))
            d = ImageDraw.Draw(text_img_mol)
            d.text((0,0), "FragVAE - Smallest Molecular Fragment Generation Process.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0));
            d.text((0,single_space), "Status: Plotting Fragment Node", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            
            flash_text_img_mol= Image.new('RGB', text_size, color = (255, 255, 255))
            d = ImageDraw.Draw(flash_text_img_mol)
            d.text((0,0), "FragVAE - Smallest Molecular Fragment Generation Process.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0));
            d.text((0,single_space), "Status: Sampling Fragment Node", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            
    
            flash_bag_frags = display_F1_graphs(params,m.GetNumAtoms(),display_F1,flash_F1, quarter_pic_size,atoms, NN_Tensor)
            text_flash_bag_frags= Image.new('RGB', text_size, color = (255, 255, 255))
            d = ImageDraw.Draw(text_flash_bag_frags)
            d.text((0,0), "Bag of molecular fragments to sample, given previous restraints.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            d.text((0,single_space), "Status: Sampling", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            text_bag_frags= Image.new('RGB', text_size, color = (255, 255, 255))
            d = ImageDraw.Draw(text_bag_frags)
            d.text((0,0), "Bag of molecular fragments to sample, given previous restraints.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            d.text((0,single_space), "Status: Plotting", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            
            
    
            img_no_flash = merge_F1_train_example(vstack_img(text_bag_frags,bag_frags),vstack_img(text_selected_frags,selected_frags), P_F1_N,vstack_img(text_img_mol,img_mol),full_quarter_pic_size)
            
            img_flash = merge_F1_train_example(vstack_img(text_flash_bag_frags,flash_bag_frags),vstack_img(text_selected_frags,selected_frags), P_F1_flash,vstack_img(flash_text_img_mol,img_mol),full_quarter_pic_size)
            
            
    
            
            gif_counter = image_flash(img_no_flash,img_flash,gif_counter)
            
            given = ' '
    
            for degree in range(0,params['max_degree'] ):
    
             
                proposed_NN = list( AI_NN[atom_idx][degree])
                #proposed_NN = [proposed_NN[0]]+[0]+proposed_NN[1:params['num_atom_features']+1 ]+[0]+proposed_NN[1+params['num_atom_features']:params['num_atom_features']*2+1 ]+[0]+proposed_NN[1+params['num_atom_features']*2:params['num_atom_features']*3+1]+[0]+proposed_NN[1+params['num_atom_features']*3:params['num_atom_features']*4+1]
                
                
                
                prob = list(NN_prob[atom_idx][degree])
                #true_NN = [prob[0]]+[0]+prob[1:params['num_atom_features']+1 ]+[0]+prob[1+params['num_atom_features']:params['num_atom_features']*2+1 ]+[0]+prob[1+params['num_atom_features']*2:params['num_atom_features']*3+1]+[0]+prob[1+params['num_atom_features']*3:params['num_atom_features']*4+1]
                true_NN = prob
                #NN_categories = ['0']+['']+['-'+i for i in params["atoms"]]+[' ']+['='+i for i in params["atoms"]]+['  ']+['#'+i for i in params["atoms"]]+['   ']+['≈'+i for i in params["atoms"]]
                NN_categories = ['0']+['-'+i for i in params["atoms"]]+['='+i for i in params["atoms"]]+['#'+i for i in params["atoms"]]+['≈'+i for i in params["atoms"]]
    
                if(( degree == 0 ) or ( np.argmax(selected_NN[atom_idx][degree-1]) != 0 )):
    
                    
                    plt.figure(figsize=(quarter_pic_size[0]/100,(quarter_pic_size[1]+text_size[1])/100), dpi=100)
                    plt.xticks(rotation='vertical')
                    plt.bar(NN_categories, proposed_NN, width=0.8, bottom=None,alpha = 0.5,color ='r',label= 'AI_P(NN)')
                    plt.bar(NN_categories, true_NN, width=0.8, bottom=None,alpha = 0.5,color ='c',label= 'P(NN)')
                    plt.bar(NN_categories, np.zeros_like(true_NN), width=0.8, bottom=None,alpha = 0.5,color =orange,label= 'Sampling')
                    plt.xticks(rotation='vertical')
                    plt.gcf().subplots_adjust(bottom=0.2,top=0.9)
                    
                    axes = plt.gca()
                    axes.set_ylim([0,1])
    
                    plt.ylabel('Prob. of sampling a nearest neighbour node',wrap=True)
                    plt.xlabel('Possible nearest neighbour nodes to sample',wrap=True)
                    plt.title('P(Node | Z , Z_sampled_frags, N.N. bag [' + given+'])',wrap=True)
                    
                    
                    plt.legend()
                    P_F1_NN = BytesIO()
                    plt.savefig(P_F1_NN, dpi=100)
                    plt.close()
                    P_F1_NN = PIL.Image.open(P_F1_NN)
                    
                    
                    next_selection = int(np.argmax(selected_NN[atom_idx][degree]))
                    
                    select_bar_mag = np.max([proposed_NN[next_selection],true_NN[next_selection]])
                    temp_proposed_NN= copy.deepcopy(proposed_NN)
                    temp_proposed_NN[next_selection] = 0
                    
                    temp_N_true_NN = copy.deepcopy(true_NN)
                    temp_N_true_NN[next_selection] = 0
                    
                    select_bar = np.zeros_like(true_NN)
                    select_bar[next_selection] = select_bar_mag
                    
                    
                    plt.figure(figsize=(quarter_pic_size[0]/100,(quarter_pic_size[1]+text_size[1])/100), dpi=100)
                    plt.bar(NN_categories, temp_proposed_NN, width=0.8, bottom=None,alpha = 0.5,color ='r',label= 'AI_P(NN)')
                    plt.bar(NN_categories, temp_N_true_NN, width=0.8, bottom=None,alpha = 0.5,color ='c',label= 'P(NN)')
                    plt.bar(NN_categories, select_bar, width=0.8, bottom=None,alpha = 0.5,color =orange,label= 'Sampling')
                    plt.xticks(rotation='vertical')
                    plt.gcf().subplots_adjust(bottom=0.2,top=0.9)
                    
                    axes = plt.gca()
                    axes.set_ylim([0,1])
    
                    plt.ylabel('Prob. of sampling a nearest neighbour node',wrap=True)
                    plt.xlabel('Possible nearest neighbour nodes to sample',wrap=True)
                    plt.title('P(Node | Z , Z_sampled_frags, N.N. bag [' + given+'])',wrap=True)
    
                    
                    
                    plt.legend()
                    P_F1_NN_flash = BytesIO()
                    plt.savefig(P_F1_NN_flash, dpi=100)
                    P_F1_NN_flash = PIL.Image.open(P_F1_NN_flash)
                    plt.close()
                    
                    
                    
                    
                    text_img_mol= Image.new('RGB', text_size, color = (255, 255, 255))
                    d = ImageDraw.Draw(text_img_mol)
                    d.text((0,0), "FragVAE - Smallest Molecular Fragment Generation Process.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0));
                    d.text((0,single_space), "Status: Plotting nearest neighbour node #"+str(degree), font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                    
                    flash_text_img_mol= Image.new('RGB', text_size, color = (255, 255, 255))
                    d = ImageDraw.Draw(flash_text_img_mol)
                    d.text((0,0), "FragVAE - Smallest Molecular Fragment Generation Process.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0));
                    d.text((0,single_space), "Status: Sampling nearest neighbour node #"+str(degree), font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                    
                    
                    flash_bag_frags = display_F1_graphs(params,m.GetNumAtoms(),display_F1,flash_F1, quarter_pic_size,atoms, NN_Tensor)
                    text_flash_bag_frags= Image.new('RGB', text_size, color = (255, 255, 255))
                    d = ImageDraw.Draw(text_flash_bag_frags)
                    d.text((0,0), "Bag of molecular fragments to sample, given previous sampling.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                    d.text((0,single_space), "Status: Sampling", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                    text_bag_frags= Image.new('RGB', text_size, color = (255, 255, 255))
                    d = ImageDraw.Draw(text_bag_frags)
                    d.text((0,0), "Bag of molecular fragments to sample, given previous sampling.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                    d.text((0,single_space), "Status: Plotting", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                    
                    
                    
                    flash_F1 = np.zeros(num_atoms)
                    display_F1 = next_display_F1
                    next_display_F1 = copy.deepcopy(display_F1)
                    for atom_idx2 in range(0,num_atoms):
                        if(selected_F1[atom_idx2]==0 and (all(atoms[atom_idx2] == atoms[int(match[atom_idx])]))):
                            if(all(( np.sum(NN_Tensor[atom_idx2][0:params['max_degree'] ],axis=0) - np.sum(selected_NN[atom_idx][0:degree+1],axis=0) )>=0) and( np.argmax(selected_NN[atom_idx][degree]) != 0 ) ):
                                flash_F1[atom_idx2] =1
                            elif(all(( np.sum(NN_Tensor[atom_idx2][: ,1:len(NN_Tensor[0,0,:]) ],axis=0) - np.sum(selected_NN[atom_idx][:,1:len(NN_Tensor[0,0,:])],axis=0) )==0)):
                                flash_F1[atom_idx2] =1
                                
                                if(atom_idx2 ==atom_idx):
                                    text_selected_frags= Image.new('RGB', text_size, color = (255, 255, 255))
                                    d = ImageDraw.Draw(text_selected_frags)
                                    d.text((0,0), "Molecular structure with previously sampled frags (top, red)", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                                    d.text((0,single_space), "Bag of previously sampled molecular frags (bottom). Updating....", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                                    vstack_img(text_selected_frags,selected_frags)       
                                
                            else:
                                next_display_F1[atom_idx2] =0
                        else:
                            next_display_F1[atom_idx2] =0
                    
                    flash_bag_frags = display_F1_graphs(params,m.GetNumAtoms(),display_F1,flash_F1, quarter_pic_size,atoms, NN_Tensor)
                    bag_frags = display_F1_graphs(params,m.GetNumAtoms(),display_F1,flash_F1*0, quarter_pic_size,atoms,NN_Tensor)
                            
    
    
                    img_no_flash = merge_F1_train_example(vstack_img(text_bag_frags,bag_frags),vstack_img(text_selected_frags,selected_frags)  , P_F1_NN,vstack_img(text_img_mol,img_mol),full_quarter_pic_size)
            
            
                
            
                    img_flash = merge_F1_train_example(vstack_img(text_flash_bag_frags,flash_bag_frags),vstack_img(text_selected_frags,selected_frags)  , P_F1_NN_flash,vstack_img(flash_text_img_mol,img_mol),full_quarter_pic_size)
                    gif_counter = image_flash(img_no_flash,img_flash,gif_counter)
                    given = given+NN_categories[next_selection] +' '
                    
            if(match[atom_idx]<m.GetNumAtoms()):
                if atom_idx !=0:
                    colors[previous_match]=(1,0,0) 
                if not(int(match[atom_idx]) in colors.keys()):
                    colors[int(match[atom_idx])]=(1,0,0)
                    
                    if atom_idx !=0:
                        colors[int(match[atom_idx])]=(1,0,0)
    
                        
                    
                    highlight.append(int(match[atom_idx]))
                    previous_match = int(match[atom_idx])
                    selected_F1[int(match[atom_idx])]=1
            if(atom_idx==num_atoms-1):
                
                text_img_mol= Image.new('RGB', text_size, color = (255, 255, 255))
                d = ImageDraw.Draw(text_img_mol)
                d.text((0,0), "FragVAE - Smallest Molecular Fragment Generation Process."+str(degree), font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0));
                d.text((0,single_space), "Status: Complete!", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                
                img_mol, svg = gen_img_svg_mol(m,highlight=highlight,colors=colors,highlightBonds=[],bond_colors={},molSize=quarter_pic_size)
                fp = BytesIO()
                svg2png(bytestring=svg,write_to=fp,scale=1)
                img_mol = PIL.Image.open(fp)
                
                text_selected_frags= Image.new('RGB', text_size, color = (255, 255, 255))
                d = ImageDraw.Draw(text_selected_frags)
                d.text((0,0), "Molecular structure with previously sampled frags (top, red)", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                d.text((0,single_space), "Bag of previously sampled molecular frags (bottom). Complete!", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                 
                text_bag_frags= Image.new('RGB', text_size, color = (255, 255, 255))
                d = ImageDraw.Draw(text_bag_frags)
                d.text((0,0), "Bag of molecular fragments to sample, given previous sampling.", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
                d.text((0,single_space), "Status: Empty", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            
                
                bag_frags = display_F1_graphs(params,m.GetNumAtoms(),display_F1*0,flash_F1, quarter_pic_size,atoms, NN_Tensor)
                selected_frags = display_F1_graphs(params,m.GetNumAtoms(),selected_F1,flash_F1*0, quarter_pic_size,atoms, NN_Tensor)
                img = merge_F1_train_example(vstack_img(text_bag_frags,bag_frags),vstack_img(text_selected_frags,selected_frags)  , P_F1_NN,vstack_img(text_img_mol,img_mol),full_quarter_pic_size)
                img.save('gif_decode/gif_image'+str(gif_counter).zfill(6)+'.png', 'PNG')
                gif_counter = gif_counter+1
                

def merge_F1_train_example(bag_frags_img,sel_frags_img, plot_img,sel_mol_img, img_size):
    transparent = transparent = np.ones([img_size[1],img_size[0],1], dtype=np.asarray(sel_mol_img).dtype)*255
    temp =  np.concatenate((np.asarray(sel_mol_img), transparent), axis=-1)
    imgs_comb = [np.hstack([temp,np.asarray(plot_img)])]
    imgs_comb.append(np.hstack([np.concatenate((np.asarray(sel_frags_img), transparent), axis=-1),np.concatenate((np.asarray(bag_frags_img), transparent), axis=-1)]))
    imgs_comb = np.vstack(imgs_comb)
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    
    return imgs_comb
def vstack_img(img_top,img_bottom):
    
    imgs_comb = np.vstack( (np.asarray( i ) for i in [img_top,img_bottom] ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb    
def image_flash(img_no_flash,img_flash,gif_counter):
    img_no_flash.save('gif_decode/gif_image'+str(gif_counter).zfill(6)+'.png', 'PNG')
    gif_counter = gif_counter+1
    '''
    img_no_flash.save('gif_decode/gif_image'+str(gif_counter).zfill(6)+'.png', 'PNG')
    gif_counter = gif_counter+1
    
    
    img_flash.save('gif_decode/gif_image'+str(gif_counter).zfill(6)+'.png', 'PNG')
    gif_counter = gif_counter+1
    
    img_no_flash.save('gif_decode/gif_image'+str(gif_counter).zfill(6)+'.png', 'PNG')
    gif_counter = gif_counter+1
    '''
    img_flash.save('gif_decode/gif_image'+str(gif_counter).zfill(6)+'.png', 'PNG')
    gif_counter = gif_counter+1
    
    return gif_counter
#test_training_model(model,params)

def demonstrate_F1_training(model,params):

    
    '''
    requires predicitive model as input
    
    '''
    
    params['gifMe'] = True
    gif_counter = 0
    g = data_generator(params,return_smiles = True)
    data = next(g)[0]

    
    mol_atoms =    data['atoms']
    mol_edge = data['edges']
    mol_bonds = data['bonds']
    mol_NN_Tensor = data['NN_Tensor'] 
    total_error_in = data['dummy_error_output']

    
    
    out_model= model.predict([mol_atoms,mol_edge,mol_bonds,mol_NN_Tensor,total_error_in])
    [NN_prob,match,N_prob,AI_N,AI_NN,selected_NN,selected_N,AI_N_out,AI_NN_out,N_match_out] = out_model
    
    
    
    drawer = rdMolDraw2D.MolDraw2DSVG(400,200)
    
    select_me = ['0']+['-'+i for i in params["atoms"]]+['='+i for i in params["atoms"]]+['#'+i for i in params["atoms"]]+['≈'+i for i in params["atoms"]]
    select_atoms = ['0']+params["atoms"]
    for mol_idx in range(0,len(mol_atoms[0])):
        '''
            atoms =np.array( [mol_atoms[0]])
            edge =np.array( [ mol_edge[0]])
            bonds = np.array( [mol_bonds[0]])
            NN_Tensor = np.array( [mol_NN_Tensor[0]])
        
        '''
        
        
        [NN_prob,match,N_prob,AI_N,AI_NN,selected_NN,selected_N,AI_N_out,AI_NN_out,N_match_out] = out_model
        [NN_prob,match,N_prob,AI_N,AI_NN,selected_NN,selected_N,AI_N_out,AI_NN_out,N_match_out] =[NN_prob[mol_idx],match[mol_idx],N_prob[mol_idx],AI_N[mol_idx],AI_NN[mol_idx],selected_NN[mol_idx],selected_N[mol_idx],AI_N_out[mol_idx],AI_NN_out[mol_idx],N_match_out[mol_idx]]
        
        smiles = data['smiles'][mol_idx]
        print(smiles)
        #img = rdkit.Chem.Draw.MolsToImage([Chem.MolFromSmiles(smiles)])
        #display(img)

        
        m = Chem.MolFromSmiles(smiles)

        highlight=[]
        colors={}
        
        for atom_idx in range(0, params['max_atoms']):  
            #plt.figure(figsize=(20,11))
            plt.figure(figsize=(19.5,11))
            plt.subplots_adjust(left=0.05, right=0.95 ,bottom=0.05, top=0.95, wspace=0.06, hspace=0.25 )
            plt.rcParams.update({'font.size': 13})
            plt.tight_layout()

                        
            
            
            plt.subplot(3,2,2)
            plt.bar(select_atoms, AI_N[atom_idx], width=0.8, bottom=None,alpha = 0.5,color ='r',label= 'AI_P(n)')
            plt.bar(select_atoms, N_prob[atom_idx], width=0.8, bottom=None,alpha = 0.5,color ='c',label= 'P(n)')
            '''
            plt.title('P(atom|Fingerprint - '+str(atom_idx)+' atoms) AI Selected [ '+
                          select_atoms[np.argmax(AI_N_out[atom_idx])]+' ], Training Selected [ '+
                          select_atoms[np.argmax(selected_N[atom_idx])]+' ]')
            '''
            plt.title('P(atom|Fingerprint - '+str(atom_idx)+' atoms) AI Selected [ '+
              select_atoms[np.argmax(AI_N_out[atom_idx])]+' ]')
            plt.legend()

            #print('press space to continue...')
            #keyboard.wait(' ') 



            
            
            given = ' '+select_atoms[np.argmax(AI_N_out[atom_idx])]
            train_given = ' '+select_atoms[np.argmax(selected_N[atom_idx])]
            for degree in range(0,params['max_degree'] ):

             
                proposed_NN = list( AI_NN[atom_idx][degree])
                proposed_NN = [proposed_NN[0]]+[0]+proposed_NN[1:params['num_atom_features']+1 ]+[0]+proposed_NN[1+params['num_atom_features']:params['num_atom_features']*2+1 ]+[0]+proposed_NN[1+params['num_atom_features']*2:params['num_atom_features']*3+1]+[0]+proposed_NN[1+params['num_atom_features']*3:params['num_atom_features']*4+1]

                prob = list(NN_prob[atom_idx][degree])
                true_NN = [prob[0]]+[0]+prob[1:params['num_atom_features']+1 ]+[0]+prob[1+params['num_atom_features']:params['num_atom_features']*2+1 ]+[0]+prob[1+params['num_atom_features']*2:params['num_atom_features']*3+1]+[0]+prob[1+params['num_atom_features']*3:params['num_atom_features']*4+1]
                
                NN_categories = ['0']+['']+['-'+i for i in params["atoms"]]+[' ']+['='+i for i in params["atoms"]]+['  ']+['#'+i for i in params["atoms"]]+['   ']+['≈'+i for i in params["atoms"]]
                if(degree != params['max_degree'] -1):
                    plt.subplot(3,2,3+degree)
                    plt.bar(NN_categories, proposed_NN, width=0.8, bottom=None,alpha = 0.5,color ='r',label= 'AI_P(NN)')
                    plt.bar(NN_categories, true_NN, width=0.8, bottom=None,alpha = 0.5,color ='c',label= 'P(NN)')
                    plt.xticks(rotation='vertical')
                    '''
                    plt.title('P(NN|Fingerprint - '+str(atom_idx)+' F1 - '+ str(degree)+' NN), AI Selected [ '+
                          select_me[np.argmax(AI_NN_out[atom_idx][degree])]+'|'+given+' ],'+' Training Selected [ '+
                          select_me[np.argmax(selected_NN[atom_idx][degree])]+'|'+train_given+' ]')
                    '''
                    
                    plt.title('P(NN|Fingerprint - '+str(atom_idx)+' F1 - '+ str(degree)+' NN), AI Selected [ '+
                          select_me[np.argmax(AI_NN_out[atom_idx][degree])]+'|'+given+' ]')
                    train_given= train_given+', '+select_me[np.argmax(selected_NN[atom_idx][degree])]
                    given = given+', '+select_me[np.argmax(AI_NN_out[atom_idx][degree])]
                    
                    plt.legend()
                    
            if(match[atom_idx]<m.GetNumAtoms()):
                if atom_idx !=0:
                    colors[previous_match]=(1,0,0) 
                if not(int(match[atom_idx]) in colors.keys()):
                    colors[int(match[atom_idx])]=(0,1,0)
                    
                    if atom_idx !=0:
                        colors[int(match[atom_idx])]=(0,1,0)
    
                        
                    
                    highlight.append(int(match[atom_idx]))
                    previous_match = int(match[atom_idx])
    
                
                    
            rdDepictor.Compute2DCoords(m)
            #SVG(moltosvg(m))
            
            plt.subplot(3,2,1)
            drawer = rdMolDraw2D.MolDraw2DSVG(400,200)
            drawer.DrawMolecule(m,highlightAtoms=highlight,highlightAtomColors=colors)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace('svg:','')
            svg2png(bytestring=svg,write_to='output.png',scale=5)
            img = imageio.imread('output.png')
                
            #display(SVG(svg))
            plt.axis('off')
            plt.imshow(img)
            
            if(params['gifMe']):
                plt.savefig('gif_decode/F1_Sample'+str(gif_counter).zfill(6)+'.png')
                gif_counter = gif_counter+1
            plt.show()  

            if(not(params['gifMe'])):
            
                print('press space to continue...')
                keyboard.wait(' ')
                
        if(params['gifMe']):

            print('press space to continue...')
            keyboard.wait(' ')
            
            
