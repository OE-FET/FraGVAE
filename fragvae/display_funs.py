# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:13:23 2019

@author: ja550
"""
import PIL
from io import BytesIO
from PIL import Image, ImageDraw,ImageFont
from rdkit.Chem import AllChem as Chem
import copy 
from .convert_mol_smile_tensor import *
from rdkit.Chem import MolFromSmiles
from random import randint
from rdkit.Chem import AllChem as Chem
import matplotlib.pyplot as plt
from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
import random

def gen_img_svg_mol(mol,molSize=(450,150),highlight=[],colors={},highlightBonds=[],bond_colors={},sanitize= False):
    def moltosvg(mol,molSize,highlight,colors,highlightBonds,highlightBondColors):
        mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        include_atoms_num = False
        if(include_atoms_num):
            opts = drawer.drawOptions()
    
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)

        drawer.DrawMolecule(mc,highlightAtoms=highlight,highlightAtomColors=colors,highlightBonds=highlightBonds,highlightBondColors=highlightBondColors)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # It seems that the svg renderer used doesn't quite hit the spec.
        # Here are some fixes to make it work in the notebook, although I think
        # the underlying issue needs to be resolved at the generation step
        return svg.replace('svg:','')
    mol = Chem.RemoveHs(mol,sanitize = sanitize)
    svg = moltosvg(mol,molSize,highlight,colors,highlightBonds,bond_colors)
    img = SVG(svg)
    return img, svg      
#display_F1_graphs      
def display_F1_graphs(params,num_F1,display_F1,flash_F1,pic_size,atoms, NN_Tensor):
    orange = (255/255,165/255,0)
    green = (0,1,0)
    light_blue = (0,1,1)
   


    png_files =[]
    
                    
    
    w_len = 1 
    stop_while = True
    while stop_while:
        
        h_len_min = int(num_F1/w_len)+int(np.mod(num_F1,w_len)>0)
        
        h_temp = pic_size[0]/w_len*h_len_min
        if(h_temp>pic_size[1]) :
            w_len = w_len+1
        else:
            stop_while = False
    h_len =  h_len_min
    molSize= (int(pic_size[0]/w_len),int(pic_size[0]/w_len))
    for atom_idx in range(0,h_len*w_len):
        if( atom_idx < len(atoms)and np.sum(atoms[atom_idx])!=0 and display_F1[atom_idx]==1):
            edmol = MolFromSmiles('')
            edmol = Chem.EditableMol(edmol)
            
            
            atom_feature_num = np.argmax(atoms[atom_idx])
            if(params['atoms'][atom_feature_num]=='H'):
                mol = Chem.MolFromSmiles('[H]')
            else:
                mol = Chem.MolFromSmiles(params['atoms'][atom_feature_num])
            atom = mol.GetAtomWithIdx(0)
            edmol.AddAtom(atom)
            if(flash_F1[atom_idx]==1):
                bond_colour = (0,0,1)
                atom_colour = (0,1,0)
            else:
                bond_colour = light_blue
                atom_colour = green
            highlight=[0]
            colors={0:atom_colour} 
            highlightBonds=[]
            bond_colors={}
            for edge_idx in range(0,len(NN_Tensor[0]))  :  
                sparse_NN_feature = np.argmax(NN_Tensor[atom_idx][edge_idx])
                if(sparse_NN_feature!=0):
                    atom_NN, bond_NN = atom_bond_from_sparse_NN_feature(sparse_NN_feature-1,params)
                    edmol.AddAtom(atom_NN)
                    #edmol.AddAtom(MolFromSmiles('I').GetAtomWithIdx(0))
                    edmol.AddBond(edge_idx+1,0,bond_NN)
                    highlightBonds.append(edge_idx)
                    bond_colors[edge_idx] = bond_colour
                    highlight.append(edge_idx+1)
                    colors[edge_idx+1]= bond_colour
                    
    
            mol = edmol.GetMol()
            
            img, svg = gen_img_svg_mol(mol,highlight=highlight,colors=colors,highlightBonds=highlightBonds,bond_colors=bond_colors,molSize= molSize)
            fp = BytesIO()
            png_files.append(fp)
    
            svg2png(bytestring=svg,write_to=fp,scale=1)
        else:
            fp = BytesIO()
            png_files.append(fp)
            
            
            image = Image.new('RGB', molSize, color = 'white')
            draw = ImageDraw.Draw(image)
            image.save(fp, 'PNG')
            
            
        
    imgs = [ PIL.Image.open(i) for i in png_files ]
    imgs_comb = []
    for y_idx in range(0,h_len):
        y_idx_list = png_files[y_idx*w_len:w_len+y_idx*w_len]
        imgs_comb.append( np.hstack( (np.asarray(PIL.Image.open(i) ) for i in y_idx_list ) ))
        
    imgs_comb = np.vstack(imgs_comb)
    
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb_size = imgs_comb.size
    if(imgs_comb_size[0]!=pic_size[0]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0]-imgs_comb_size[0],imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.hstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        
        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb_size = imgs_comb.size
    if(imgs_comb_size[1]!=pic_size[1]):
        fp = BytesIO()
        image = PIL.Image.new('RGB', (pic_size[0],pic_size[1]-imgs_comb_size[1]), color = 'white')
        draw = PIL.ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.vstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        imgs_comb = PIL.Image.fromarray( imgs_comb)
    fp = BytesIO()
    imgs_comb.save( fp,'PNG' )
    #display(PIL.Image.open(fp))
    #print(imgs_comb.size)
    '''   
                    
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    display(imgs_comb)
    
    
                imgs_comb = np.hstack( (np.asarray( i ) for i in [imgs_comb,img] ) )
            imgs_comb = PIL.Image.fromarray( imgs_comb)
            display(imgs_comb)
            
            #os.remove(list_im[0])
            #os.remove(list_im[1])
            imgs_comb.save( 'gif_decode/gif_molx'+str(str(params['gif_idx']-1).zfill(6))+'.png' )
            params['gif_idx'] = params['gif_idx']-1
    '''
    return imgs_comb
    #os.remove(list_im[0])
    #os.remove(list_im[1])
    #imgs_comb.save( 'gif_decode/gif_molx'+str(str(params['gif_idx']-1).zfill(6))+'.png' )
    

    
    
def display_F2_graphs(params,num_F2,pic_size,mol):
    orange = (255/255,165/255,0)
    green = (0,1,0)
    light_blue = (0,1,1)

    png_files =[]

    w_len = 1 
    stop_while = True
    while stop_while:
        
        h_len_min = int(num_F2/w_len)+int(np.mod(num_F2,w_len)>0)
        
        h_temp = pic_size[0]/w_len*h_len_min
        if(h_temp>pic_size[1]) :
            w_len = w_len+1
        else:
            stop_while = False
    h_len =  h_len_min
    molSize= (int(pic_size[0]/w_len),int(pic_size[0]/w_len))
    for bond_idx in range(0,h_len*w_len):
        
        
        if( bond_idx < mol.GetNumBonds()):
            
            bond = mol.GetBondWithIdx(bond_idx)
            keep_atoms = [bond.GetBeginAtomIdx()]
            keep_atoms.append( bond.GetEndAtomIdx())
            
            keep_atoms=keep_atoms+list( set( [neigh.GetIdx() for neigh in bond.GetEndAtom().GetNeighbors()]+ [neigh.GetIdx() for neigh in bond.GetBeginAtom().GetNeighbors()]))
            keep_atoms2 = set(keep_atoms)
            
            all_atoms = set(np.array(range(0,mol.GetNumAtoms())))
            remove = list(all_atoms -keep_atoms2)
            edmol = Chem.EditableMol(mol)
            remove.sort(reverse=True)
            for atom_idx in remove:
                edmol.RemoveAtom(int(atom_idx))
            
            
            for indx in range(0,len(keep_atoms)):
                keep_atoms[indx] = keep_atoms[indx] - np.sum((remove<np.ones_like(remove)*keep_atoms[indx])*1)
    
            edmol_mol = edmol.GetMol()

            highlight=[]
            colors={} 
            for idx in  keep_atoms:
                highlight.append(int(idx))
                colors[int(idx)] = light_blue
                
            colors[keep_atoms[0]] = (139/250,69/250,19/250)
            colors[keep_atoms[1]] = green
            
            highlightBonds=[]
            bond_colors ={}
            for idx in range(edmol_mol.GetNumBonds()):
                highlightBonds.append(int(idx))
                bond_colors[idx] = light_blue
            sym_bond = edmol_mol.GetBondBetweenAtoms(int(keep_atoms[0]),int(keep_atoms[1]))
            
            bond_colors[sym_bond.GetIdx()] = (0,0,1)
    
    
    
            
            img, svg = gen_img_svg_mol(edmol_mol,highlight=highlight,colors=colors,highlightBonds=highlightBonds,bond_colors=bond_colors,molSize= molSize)
            fp = BytesIO()
            png_files.append(fp)
    
            svg2png(bytestring=svg,write_to=fp,scale=1)
        else:
            fp = BytesIO()
            png_files.append(fp)
            
            
            image = Image.new('RGB', molSize, color = 'white')
            draw = ImageDraw.Draw(image)
            image.save(fp, 'PNG')
            
            
        
    imgs = [ PIL.Image.open(i) for i in png_files ]
    imgs_comb = []
    for y_idx in range(0,h_len):
        y_idx_list = png_files[y_idx*w_len:w_len+y_idx*w_len]
        imgs_comb.append( np.hstack( (np.asarray(PIL.Image.open(i) ) for i in y_idx_list ) ))
        
    imgs_comb = np.vstack(imgs_comb)
    
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb_size = imgs_comb.size
    if(imgs_comb_size[0]!=pic_size[0]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0]-imgs_comb_size[0],imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.hstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        
        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb_size = imgs_comb.size
    if(imgs_comb_size[1]!=pic_size[1]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0],pic_size[1]-imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.vstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        imgs_comb = PIL.Image.fromarray( imgs_comb)
    fp = BytesIO()
    imgs_comb.save( fp,'PNG' )
    #display(PIL.Image.open(fp))
    #print(imgs_comb.size)
    '''   
                    
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    display(imgs_comb)
    
    
                imgs_comb = np.hstack( (np.asarray( i ) for i in [imgs_comb,img] ) )
            imgs_comb = PIL.Image.fromarray( imgs_comb)
            display(imgs_comb)
            
            #os.remove(list_im[0])
            #os.remove(list_im[1])
            imgs_comb.save( 'gif_decode/gif_molx'+str(str(params['gif_idx']-1).zfill(6))+'.png' )
            params['gif_idx'] = params['gif_idx']-1
    '''
    return imgs_comb
    #os.remove(list_im[0])
    #os.remove(list_im[1])
    #imgs_comb.save( 'gif_decode/gif_molx'+str(str(params['gif_idx']-1).zfill(6))+'.png' )
    
    
    
def display_training_data(params,num_F2=80,pic_size=(int(1190*0.9),int(1684*0.9)),train=True):
    text_size =20
    text_font_size = 20
    if(train==True):
        libExamples = pd.read_csv(params['model_dir']+'Experimental_Training_set_NN.csv')
        libExamples = libExamples.reset_index(drop=True)
    else:
        libExamples = pd.read_csv(params['model_dir']+'Experimental_Test_NN.csv')
        libExamples = libExamples.reset_index(drop=True)
    
    
    orange = (255/255,165/255,0)
    green = (0,1,0)
    light_blue = (0,1,1)

    png_files =[]

    w_len = 1 
    stop_while = True
    while stop_while:
        
        h_len_min = int(num_F2/w_len)+int(np.mod(num_F2,w_len)>0)
        
        h_temp = pic_size[0]/w_len*h_len_min
        if(h_temp>pic_size[1]) :
            w_len = w_len+1
        else:
            stop_while = False
    h_len =  h_len_min
    molSize= (int(pic_size[0]/w_len),int(pic_size[0]/w_len))
    
    x = np.array(range(len(libExamples)))
    #random.shuffle(x)
    for mol_idx1 in range(h_len*w_len):
       
        
        if( mol_idx1 < num_F2):
            mol_idx = x[mol_idx1]
            highlight =[]
            colors={}
            highlightBonds=[]
            highlightAtomRadii ={}
            bond_colors={}
            mol =smile_to_mol(libExamples['smiles'][mol_idx], params)
            mol = Chem.RemoveHs(mol,sanitize = False)
            if(libExamples['metrics'][mol_idx]==1):
                for i in range(mol.GetNumAtoms()):
                    highlight.append(i)
                    if(libExamples['metrics'][mol_idx]==1):
                        colors[i] = (0.75,1,0.75)
                    else:
                        colors[i] = (1,0.75,0.75)
                    highlightAtomRadii[i] =1000000
                        
                for i in range(mol.GetNumBonds()):
                    highlightBonds.append(i)
                    if(libExamples['metrics'][mol_idx]==1):
                        bond_colors[i] = (0.75,1,0.75)
                    else:
                        bond_colors[i] = (1,0.75,0.75)

            
            Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
            drawer.DrawMolecule(mol,highlightAtoms=highlight,highlightAtomColors=colors,highlightBonds=highlightBonds,highlightBondColors=bond_colors,highlightAtomRadii=highlightAtomRadii)

            drawer.FinishDrawing()
            fp = BytesIO()
            #with open(fp,'wb') as f:
            fp.write(drawer.GetDrawingText())
            
            
            img_top = PIL.Image.open(fp)
            if(libExamples['metrics'][mol_idx]==1):
                text_works = Image.new('RGB', (molSize[0],text_size), color = (191,255,191))
            else:
                text_works = Image.new('RGB', (molSize[0],text_size), color = (255, 255, 255))
            d = ImageDraw.Draw(text_works)

            if((libExamples['ECFP'][mol_idx]==0 and libExamples['metrics'][mol_idx]==1)or(libExamples['ECFP'][mol_idx]==1 and libExamples['metrics'][mol_idx]==0)):
                symbol = 'E'
            else:
                symbol =  ""
            d.text((int(molSize[0]/2)+text_size*0.5,0), symbol, font=ImageFont.truetype("arial.ttf", text_font_size),fill=(0,0,255))
            
            if((libExamples['ChemVAE'][mol_idx]==0 and libExamples['metrics'][mol_idx]==1)or(libExamples['ChemVAE'][mol_idx]==1 and libExamples['metrics'][mol_idx]==0)):
                symbol =  'C'
            else:
                symbol =  ""
            d.text((int(molSize[0]/2)+text_size*1.5,0), symbol, font=ImageFont.truetype("arial.ttf", text_font_size),fill=(0,0,255))
            
            if((libExamples['rnd_FragVAE'][mol_idx]==0 and libExamples['metrics'][mol_idx]==1)or(libExamples['rnd_FragVAE'][mol_idx]==1 and libExamples['metrics'][mol_idx]==0)):
                symbol =  'R'
            else:
                symbol =  ""
            d.text((int(molSize[0]/2)-text_size*0.5,0), symbol, font=ImageFont.truetype("arial.ttf", text_font_size),fill=(0,0,255))
            
            if((libExamples['FragVAE'][mol_idx]==0 and libExamples['metrics'][mol_idx]==1)or(libExamples['FragVAE'][mol_idx]==1 and libExamples['metrics'][mol_idx]==0)):
                symbol =  'F'
            else:
                symbol =  ""
            d.text((int(molSize[0]/2)-text_size*1.5,0),symbol, font=ImageFont.truetype("arial.ttf", text_font_size),fill=(0,0,255))
            d.text((0,0),str(mol_idx1+1), font=ImageFont.truetype("arialbd.ttf", text_font_size),fill=(0,0,0))


            
            comb_img = vstack_img(img_top,text_works)
            fp1 = BytesIO()
            comb_img.save(fp1, 'PNG')
            png_files.append(fp1)
 
            '''
            img, svg = gen_img_svg_mol(mol,molSize=(500,250))
            fp = BytesIO()
            
            display(img)
            print(libExamples['smiles'][mol_idx])
            svg2png(bytestring=svg,write_to=fp,scale=1)
            
                text_selected_frags_flash = Image.new('RGB', (half_pic_size[0],single_space*2), color = (255, 255, 255))
    d = ImageDraw.Draw(text_selected_frags_flash)
    d.text((0,0), "Bag of sampled first order fragments", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), "Status: Sampling node", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            
            '''
            
        else:
            fp2 = BytesIO()
            png_files.append(fp2)
            

            image = Image.new('RGB', (molSize[0],molSize[1]+text_size), color = 'white')
            draw = ImageDraw.Draw(image)
            image.save(fp2, 'PNG')
            
    imgs = [ PIL.Image.open(i) for i in png_files ]

    imgs_comb = []
    for y_idx in range(0,h_len):
        y_idx_list = png_files[y_idx*w_len:w_len+y_idx*w_len]
        imgs_comb.append( np.hstack( (np.asarray(PIL.Image.open(i) ) for i in y_idx_list ) ))
        
    imgs_comb = np.vstack(imgs_comb)
    
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb_size = imgs_comb.size
    if(imgs_comb_size[0]!=pic_size[0]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0]-imgs_comb_size[0],imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.hstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        
        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb_size = imgs_comb.size
    '''
    if(imgs_comb_size[1]!=pic_size[1]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0],pic_size[1]-imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.vstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        imgs_comb = PIL.Image.fromarray( imgs_comb)
    '''
    fp = BytesIO()
    imgs_comb.save( fp,'PNG' )

    return imgs_comb






def display_encoded_decoded(params,tanimoto_values,encode_decode_mols,num_F2=100,pic_size=(1190/2,1684/2),start_num=1):
    

    text_size =12
    text_font_size = 12

    
    orange = (255/255,165/255,0)
    green = (0,1,0)
    light_blue = (0,1,1)

    png_files =[]

    w_len = 1 
    '''
    stop_while = True
    while stop_while:
        
        h_len_min = int(num_F2/w_len)+int(np.mod(num_F2,w_len)>0)
        
        h_temp = pic_size[0]/w_len*h_len_min
        if(h_temp>pic_size[1]) :
            w_len = w_len+1
        else:
            stop_while = False
    '''
    w_len = 5
    h_len_min = int(num_F2/w_len)+int(np.mod(num_F2,w_len)>0)

    h_len =  h_len_min
    molSize= (int(pic_size[0]/w_len)*2,int(pic_size[0]/w_len))
    
    x = np.array(range(len(libExamples)))
    #random.shuffle(x)
    for mol_idx1 in range(h_len*w_len):
       
        
        if( mol_idx1 < num_F2):
            #mol = Chem.RemoveHs(mol,sanitize = False)


            mol = Chem.RemoveHs(encode_decode_mols[mol_idx1][0],sanitize = False)
            Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],int(molSize[1]/2)*2)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            fp = BytesIO()
            #with open(fp,'wb') as f:
            fp.write(drawer.GetDrawingText())
            img_top = PIL.Image.open(fp)
            
            mol = Chem.RemoveHs(encode_decode_mols[mol_idx1][1],sanitize = False)
            Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],int(molSize[1]/2)*2)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            fp = BytesIO()
            #with open(fp,'wb') as f:
            fp.write(drawer.GetDrawingText())
            
            
            img_bottom = PIL.Image.open(fp)
            

            text_works = Image.new('RGB', (molSize[0],text_size), color = (255, 255, 255))
            d = ImageDraw.Draw(text_works)

            
            d.text((0,0),str(mol_idx1+start_num), font=ImageFont.truetype("arialbd.ttf", text_font_size),fill=(0,0,0))
            d.text((0,0),'              Tanimoto '+str(int(tanimoto_values[mol_idx1]*100)/100), font=ImageFont.truetype("arial.ttf", text_font_size),fill=(0,0,0))

            
            comb_img = vstack_img(text_works,vstack_img(img_top,img_bottom))
            fp1 = BytesIO()
            comb_img.save(fp1, 'PNG')
            png_files.append(fp1)
            #display(comb_img)
            #print(comb_img.size)
            '''
            img, svg = gen_img_svg_mol(mol,molSize=(500,250))
            fp = BytesIO()
            
            display(img)
            print(libExamples['smiles'][mol_idx])
            svg2png(bytestring=svg,write_to=fp,scale=1)
            
                text_selected_frags_flash = Image.new('RGB', (half_pic_size[0],single_space*2), color = (255, 255, 255))
    d = ImageDraw.Draw(text_selected_frags_flash)
    d.text((0,0), "Bag of sampled first order fragments", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
    d.text((0,single_space), "Status: Sampling node", font=ImageFont.truetype("arial.ttf", font_size),fill=(0,0,0))
            
            '''
            
        else:
            fp2 = BytesIO()
            png_files.append(fp2)
            
            #print((int(molSize[1]/2.0)*4))
            image = Image.new('RGB', (molSize[0],(int(molSize[1]/2.0)*4)+text_size), color = 'white')
            draw = ImageDraw.Draw(image)
            image.save(fp2, 'PNG')
            #print(image.size)
            
    imgs = [ PIL.Image.open(i) for i in png_files ]

    imgs_comb = []
    for y_idx in range(0,h_len):
        y_idx_list = png_files[y_idx*w_len:w_len+y_idx*w_len]
        imgs_comb.append( np.hstack( (np.asarray(PIL.Image.open(i) ) for i in y_idx_list ) ))
        
    imgs_comb = np.vstack(imgs_comb)
    
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb_size = imgs_comb.size
    '''
    if(imgs_comb_size[0]!=pic_size[0]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0]-imgs_comb_size[0],imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.hstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        
        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb_size = imgs_comb.size
    '''
    '''
    if(imgs_comb_size[1]!=pic_size[1]):
        fp = BytesIO()
        image = Image.new('RGB', (pic_size[0],pic_size[1]-imgs_comb_size[1]), color = 'white')
        draw = ImageDraw.Draw(image)
        image.save(fp, 'PNG')
        imgs_comb=( np.vstack( (np.asarray(i) for i in [imgs_comb,PIL.Image.open(fp)] ) ))
        imgs_comb = PIL.Image.fromarray( imgs_comb)
    '''
    fp = BytesIO()
    imgs_comb.save( fp,'PNG' )
    
    return imgs_comb
'''
start_num =1
max_batch = 35
for i in range(0,4):
    print(i)
    imgs_comb=display_encoded_decoded(params,tanimoto_values[max_batch*i:max_batch*(i+1)],encode_decode_mols[max_batch*i:max_batch*(i+1)],num_F2=max_batch,pic_size=(1190/2*1.2,1684/2),start_num=max_batch*i+1)
    display(imgs_comb)

'''
def plot_ven_diagram():
    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib_venn import venn3, venn3_circles
     
    # Make a Basic Venn
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
     
    # Custom it
    v.get_patch_by_id('100').set_alpha(1.0)
    v.get_patch_by_id('100').set_color('white')
    v.get_label_by_id('100').set_text('Unknown')
    v.get_label_by_id('A').set_text('Set "A"')
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
    c[0].set_lw(1.0)
    c[0].set_ls('dotted')
     
    # Add title and annotation
    plt.title("Sample Venn diagram")
    plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
    ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
     
    # Show it
    plt.show()






def vstack_img(img_top,img_bottom):
    imgs_comb = np.vstack( (np.asarray( i ) for i in [img_top,img_bottom] ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb 

def gif_rebuild(duration = 1.2,name=''):
    import os
    import imageio
    
    png_dir = 'imgs_gifs/imgs_to_gif'
    images = []
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('imgs_gifs/animation'+name+'2.gif', images,duration = duration)    
    

def display_main_figure(params):
    import PIL
    from io import BytesIO
    from PIL import Image, ImageDraw,ImageFont
    from matplotlib.collections import PatchCollection
    
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem.AllChem import Compute2DCoords
    from rdkit.Chem.Draw import rdMolDraw2D
    import random
    from io import BytesIO
    
    
    fig, ax = plt.subplots(figsize=(30,30))
    
    #fig, ax = plt.subplots()
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (-100, 100)),
        
        (Path.CURVE4, (-100, -100)),
        (Path.CURVE4, (100, -100)),
        (Path.CURVE4, (100,100)),
        (Path.CURVE4, (-100, 100) )]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    plt.axis('off')
    patch = mpatches.PathPatch(path, facecolor=(1,1,0.925),color=(1,1,0.925) ,alpha=1,fill=True)
    ax.add_patch(patch)
    
    #fig, ax = plt.subplots()
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (500/480, -(840+480-480)/480)),
        
        (Path.CURVE4, (900/480, (880-480)/480)),
        (Path.CURVE4, (290/480, -(388-480)/480)),
        (Path.CURVE4, (212/480,-(325-480) /480)),
        (Path.CURVE4, (144/480, -(272-480)/480)),
    
        (Path.CURVE4, (26/480, -(197-480)/480))
        ,(Path.CURVE4, (46/480, -(139-480)/480))
        ,(Path.CURVE4, (77/480, -(66-480)/480))
        ,(Path.CURVE4, (260/480, -(0-480-50)/480))
        ,(Path.CURVE4, (0/480, -(0-480)/480))
        ,(Path.CURVE4, (0/480, -(0-480)/480))
        ,(Path.CURVE4, (0/480, -(0-480)/480))
        ,(Path.MOVETO, (0/480, (0)/480)) ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    plt.axis('off')
    patch = mpatches.PathPatch(path, facecolor=(0.925,1,0.925),color=(0.925,1,0.925) ,alpha=1,fill=True)
    ax.add_patch(patch)
    
    # plot control points and connecting lines
    x, y = zip(*path.vertices)
    #@line, = ax.plot(x, y)
    
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (480/480, -(0-620)/480)),
        (Path.CURVE4, (480/480, -(0-480)/480)),
        (Path.CURVE4, (1402/480, (261-480)/480)),
        (Path.CURVE4, (300/480, -(209-480)/480)),
        (Path.CURVE4, (273/480,-(143-480) /480)),
        (Path.CURVE4, (291/480, -(-163-480)/480)) ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=(0.925,0.925,1),color=(0.925,0.925,1) ,alpha=1,fill=True)
    
    
    ax.add_patch(patch)
    
    
    
    ellipse = mpatches.Ellipse( (0.5,0.155), 0.35,0.225,color=(1,1,0.925) ,alpha=1,fill=True)
    ax.add_patch(ellipse)
    
    

    ellipse = mpatches.Ellipse(path, 0.2, 0.1)

    #collection.set_array(np.array(colors))
    ax.add_patch(patch)
    
    
    
    # plot control points and connecting lines
    x, y = zip(*path.vertices)
    #@line, = ax.plot(x, y)
    
    
    
    #ax.grid()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    def display_smiles(smiles_list,colour_type,params,molSize=(450,150)):
        for smile in smiles_list:
            highlight =[]
            colors={}
            highlightBonds=[]
            highlightAtomRadii ={}
            bond_colors={}
            mol = smile_to_mol(smile, params)
            mol = Chem.RemoveHs(mol,sanitize = False)
            if(colour_type=='t'):
                color = (1,1,0.925)
            elif(colour_type=='g'):
                
                color = (0.925,1,0.925)
            elif(colour_type=='w'):
                color = (1,1,1)
            else:
                color = (0.925,0.925,1)
            
                
            for i in range(mol.GetNumAtoms()):
                highlight.append(i)
                colors[i] = color
                highlightAtomRadii[i] =1000000
                
                
            for i in range(mol.GetNumBonds()):
                highlightBonds.append(i)
                bond_colors[i] = color
    
            
            Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
            drawer.DrawMolecule(mol,highlightAtoms=highlight,highlightAtomColors=colors,highlightBonds=highlightBonds,highlightBondColors=bond_colors,highlightAtomRadii=highlightAtomRadii)
    
            drawer.FinishDrawing()
            fp = BytesIO()
            #with open(fp,'wb') as f:
            fp.write(drawer.GetDrawingText())
            
            
            img_top = PIL.Image.open(fp)
            display(img_top)
    
    smile_unknown = ['C1CC(=C(C#N)C#N)CCC1=C(C#N)C#N','CC1=C(C(=O)C(=C(C1=O)[N+](=O)[O-])C)[N+](=O)[O-]','C(#N)C1=C(C(=O)C(=C(C1=O)Cl)Cl)C#N','C1=CC=C2C(=C1)C(=O)C3=C(C2=O)SC(=C(S3)C#N)C#N','CC(=O)C(=CC1=CC=CC=C1)C(=O)C']
    smile_known_work = ['C1=CC(=C(C#N)C#N)C=CC1=C(C#N)C#N','N#C/C(C#N)=C(/C/1=C(C#N)/C#N)\C1=C(C#N)\C#N','N#C/C(C(OC)=O)=C1C2=C(C=CC=C2)C3=C/1C(C(C=CC=C4)=C4/C5=C(C(OC)=O)/C#N)=C5C(C6=C/7C=CC=C6)=C3C7=C(C#N)\C(OC)=O']
    smile_known_notwork  = ['C1=CC(=C(C=C1[N+](=O)[O-])Cl)N','C1=CC=C(C=C1)C(=O)C2=CC=CC=C2','C(C#N)C#N','CCOC(=O)C(=C)C#N','C(=CC#N)C#N','CC(C)(C)C1=CC2=C(C=C1)C=C(C(=C2)C#N)C#N','CC1=CC=CC=C1C2=CC=CC=C2C']            
    
    
    display_smiles(smile_unknown,'t',params)
    display_smiles(smile_known_work,'g',params)
    display_smiles(smile_known_notwork,'e',params)
        

    display_smiles(smile_unknown,'w',params)
    display_smiles(smile_known_work,'w',params)
    display_smiles(smile_known_notwork,'w',params)
        
    
    
    
    fig, ax = plt.subplots(figsize=(20, 20))
    input_dim = 1
    output_dim =0.6
    len_dim = 0.5
    #fig, ax = plt.subplots()
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (0, 0)),
        
        (Path.LINETO, (len_dim, output_dim/2)),
        (Path.LINETO, (len_dim, 1-output_dim/2)),
        (Path.LINETO, (0,1)),
        (Path.LINETO, (0, 0) )]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    plt.axis('off')
    patch = mpatches.PathPatch(path, facecolor=(1,0,0),color=(1,0,0) ,alpha=1,fill=True)
    ax.add_patch(patch)
    

    
    
    #ax.grid()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    
    plt.show()


    fig, ax = plt.subplots(figsize=(20,20))
    input_dim = 1
    output_dim =0.6
    len_dim = 0.5
    #fig, ax = plt.subplots()
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (0, 0)),
        
        (Path.LINETO, (len_dim, output_dim/2)),
        (Path.LINETO, (len_dim, 1-output_dim/2)),
        (Path.LINETO, (0,1)),
        (Path.LINETO, (0, 0) )]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    plt.axis('off')
    patch = mpatches.PathPatch(path, facecolor=(0,0,1),color=(0,0,1) ,alpha=1,fill=True)
    ax.add_patch(patch)
    

    
    
    #ax.grid()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    
    plt.show()
    
#display_main_figure(params)