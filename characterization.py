# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:04:51 2019
@author: ja550

Python file contains all functions used to Charactersize FraGVAE in our paper
This file is not required to run FraGVAE

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import copy 
import fragvae as fg
import sys 
import os 

from rdkit.Chem import AllChem as Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from pathlib import Path
from tensorflow.keras.layers import Layer, Input, InputSpec, Dense
from tensorflow.keras import models, optimizers, regularizers

# requires Chemvae be installed in GitHub directory (Directorys are mangaged using GitHub Desktop, currently trouble shooting git installation error)
sys.path.append('..')
from chemical_vae import *
from chemical_vae.chemvae.vae_utils import VAEUtils

def compared_fingerprints_general(model_num_frag,y_predict_name):
    
    #Generate a FraGVAE object with experiment number 5 
    fragvae_obj = fg.FraGVAE(model_num_frag)
    params = fragvae_obj.params
    fragvae_obj.load_models(rnd=False,testing = False)
    
    
    print("Loaded previous FraGVAE model")

    
    rnd_fragvae_obj = fg.FraGVAE(model_num_frag)
    rnd_fragvae_obj.load_models(rnd=True,testing = False)
    rnd_fragvae_obj.reset_weights()


    print("Loaded previous random FraGVAE model")

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    Expt_results = pd.DataFrame()
    

    Rnd_parameters = {'min_samples_leaf':[1,2],'n_estimators':[200],'max_features':[0.25,0.5,0.75],'max_depth':[6,7,None]}
    
    with_chemVAE=True
    
    
    libExamples = pd.read_csv('data_lib/'+params['train_dataset']+'.csv')
    libExamples = libExamples.sample(frac=1).reset_index(drop=True)
    num_repeats = [100, 100,100,100,100,100]
    num_samples = [10, 16,25,40,63,100]
    
    if(params['train_dataset'] == 'ESOL_Delaneyfiltered'):

        num_features_FragVAE = [10,20,30,40,50,60,70,80,90,100]
        num_features_ChemVAE = [10,20,30,40,50,60,70,90,100,110,140,150,160,180,200]
        num_features_ECFP =    [10,20,30,40,50,60,70,90,100,110,140,150,160,180,200,220,240,260,300,340,360,400,240,480]
        test_num=700
        y_predict_name ='logP'
        chem_vae_model_dir = '/chemical_vae/models/zinc'

    elif(params['train_dataset']=='Zinc15filtered'):

        num_features_FragVAE = [10,20,30,40,50,60,70,90,100,110,140,150,160,180,200,220,240,260,300,340,360,400]
        num_features_ChemVAE = [10,20,30,40,50,60,70,90,100,110,140,150,160,180,200]
        num_features_ECFP =    [10,20,30,40,50,60,70,90,100,110,140,150,160,180,200,220,240,260,300,340,360,400,240,480]
        num_features_ECFP =    num_features_FragVAE
        test_num=1000
        
        chem_vae_model_dir = '/chemical_vae/models/zinc'

 
    Expt_results['Number_Training_Samples'] = pd.Series(num_samples)


    Expt_results['ECFP_MSE_mean'] = pd.Series( np.ones([len(num_samples)])*np.inf)
    Expt_results['ECFP_MSE_std'] = pd.Series( np.ones([len(num_samples)])*np.inf)

        
    Expt_results['FragVAE_MSE_mean'] = pd.Series( np.ones([len(num_samples)])*np.inf)
    Expt_results['FragVAE_MSE_std'] = pd.Series( np.ones([len(num_samples)])*np.inf)
    
    Expt_results['rnd_FragVAE_MSE_mean'] = pd.Series( np.ones([len(num_samples)])*np.inf)
    Expt_results['rnd_FragVAE_MSE_std'] = pd.Series( np.ones([len(num_samples)])*np.inf)
    
    if(with_chemVAE):
        Expt_results['ChemVAE_MSE_mean'] = pd.Series( np.ones([len(num_samples)])*np.inf)
        Expt_results['ChemVAE_MSE_std'] = pd.Series( np.ones([len(num_samples)])*np.inf)
    

    if(with_chemVAE):
        curdir = os.getcwd()
        parent_path = Path(curdir).parent.as_posix()
        vae = VAEUtils(directory=parent_path+chem_vae_model_dir)
        
        
    # iterate through the number of samples used to make a prediction.
    for sample_idx in range(0,len(num_samples)):
    
            
        
        ECFP_MSE_sample = []
        FragVAE_MSE_sample = []
        rnd_FragVAE_MSE_sample = []
        ChemVAE_MSE_sample = []
        
        # reap the experiment num_repeats of times
        for repeat_idx in range(0,num_repeats[sample_idx]):
            
            rnd_fragvae_obj.reset_weights()
            
            sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' ECFP'+params['excess_space'])
            sys.stdout.flush()
            

            libExamples = libExamples.sample(frac=1).reset_index(drop=True)
            data_train_subset = libExamples[0:num_samples[sample_idx]]
            data_train_subset = data_train_subset.reset_index(drop=True)
            #data_train_subset.to_csv(params['model_dir']+y_predict_name+'_train_data_samp_'+str(num_samples[sample_idx]).zfill(4)+'_'+str(repeat_idx).zfill(2)+'.csv',index=False)
            
            data_test_subset = libExamples[num_samples[sample_idx]:num_samples[sample_idx]+test_num]
            #data_test_subset.to_csv(params['model_dir']+y_predict_name+'_test_data_samp_'+str(num_samples[sample_idx]).zfill(4)+'_'+str(repeat_idx).zfill(2)+'.csv',index=False)
            

            


            ''''
            ECFP preditionss 
            
            '''
            best_score = -np.inf
            best_num_features = -1
            best_ECFP_degree = -1
            for ECFP_degree in [2,3,4]:
                list_ECFPs = find_ECFP(data_train_subset['smiles'], data_train_subset[y_predict_name], params,max(num_features_ECFP), ECFP_degree=ECFP_degree)
                
                ECFP_features_train = gen_features_from_ECFPS(data_train_subset['smiles'], list_ECFPs, max(num_features_ECFP),params)
                for num_features in num_features_ECFP:
    
                    # Find valid mlecular fingerprints
    
                    Train_temp = ECFP_features_train[:,len(ECFP_features_train[0])-num_features:len(ECFP_features_train[0])]

                    reg = RandomForestRegressor( )                
                    clf = GridSearchCV(reg, Rnd_parameters,scoring='neg_mean_squared_error', cv=3)
                    clf.fit(Train_temp,data_train_subset[y_predict_name])
                    
                    if(repeat_idx==0):
                        sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' ECFP'+' degree '+str(ECFP_degree)+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score)+params['excess_space'])
                    else:
                        sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' ECFP'+' degree '+str(ECFP_degree)+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score) +' MSE '+str(np.mean(np.array(ECFP_MSE_sample)))+params['excess_space'])
                    sys.stdout.flush()
                    if(-clf.best_score_ < -best_score):
                        rnd_model = clf.best_estimator_ 
                        best_num_features = num_features
                        best_score=clf.best_score_
                        best_ECFP_degree =ECFP_degree

            list_ECFPs = find_ECFP(data_train_subset['smiles'], data_train_subset[y_predict_name], params,best_num_features, ECFP_degree=best_ECFP_degree)
            ECFP_features_test = gen_features_from_ECFPS(data_test_subset['smiles'], list_ECFPs, best_num_features,params)
            
            test_predictions = rnd_model.predict(ECFP_features_test)
            iteration_mse =  np.sum((test_predictions - data_test_subset[y_predict_name])**2)/test_num

                
            ECFP_MSE_sample.append(iteration_mse)
                
            
            ''''
            FragVAE preditionss 
            
            '''
            sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' FragVAE'+params['excess_space'])
            sys.stdout.flush()
            
            rel_features = find_Frag_VAE_features(data_train_subset['smiles'], data_train_subset[y_predict_name], params, fragvae_obj,max(num_features_FragVAE),with_F1 = True)    
            FragVAE_features_train = gen_features_from_Frag_VAE(data_train_subset['smiles'], fragvae_obj, rel_features,params)
            
            
            early_stop = -100
            best_score = -np.inf
            best_num_features = -1
            for num_features in num_features_FragVAE:

                # Find valid mlecular fingerprints

                Train_temp = FragVAE_features_train[:,len(FragVAE_features_train[0])-num_features:len(FragVAE_features_train[0])]
                reg = RandomForestRegressor( )                
                clf = GridSearchCV(reg, Rnd_parameters,scoring='neg_mean_squared_error', cv=3)             
                clf.fit(Train_temp,data_train_subset[y_predict_name])
                
                if(repeat_idx==0):
                    sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' FragVAE'+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score)+params['excess_space'])
                else:
                    sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' FragVAE'+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score) +' MSE '+str(np.mean(np.array(FragVAE_MSE_sample)))+params['excess_space'])
                sys.stdout.flush()
                if(-clf.best_score_ < -best_score):
                    rnd_model = clf.best_estimator_ 
                    best_num_features = num_features
                    best_score=clf.best_score_
                    early_stop = 0
                else:
                    early_stop = early_stop+1
                    if(early_stop==2):
                        break
        
        
            
            FragVAE_features_test = gen_features_from_Frag_VAE(data_test_subset['smiles'], fragvae_obj, rel_features,params)
            FragVAE_features_test = FragVAE_features_test[:,len(FragVAE_features_test[0])-best_num_features:len(FragVAE_features_test[0])]

            test_predictions = rnd_model.predict(FragVAE_features_test)
            iteration_mse =  np.sum((test_predictions - data_test_subset[y_predict_name])**2)/test_num
            
            
            
            FragVAE_MSE_sample.append(iteration_mse)
            
            ''''
            Rnd FragVAE preditionss 
            
            '''
            sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' RND FragVAE'+params['excess_space'])
            sys.stdout.flush()
            
            

            rel_features = find_Frag_VAE_features(data_train_subset['smiles'], data_train_subset[y_predict_name], params,rnd_fragvae_obj,max(num_features_FragVAE),with_F1 = True)    
            FragVAE_features_train = gen_features_from_Frag_VAE(data_train_subset['smiles'], rnd_fragvae_obj, rel_features,params)
            
            best_score = -np.inf
            best_num_features = -1
            early_stop = -100
            for num_features in num_features_FragVAE:

                # Find valid mlecular fingerprints

                Train_temp = FragVAE_features_train[:,len(FragVAE_features_train[0])-num_features:len(FragVAE_features_train[0])]

                reg = RandomForestRegressor( )                
                clf = GridSearchCV(reg, Rnd_parameters,scoring='neg_mean_squared_error', cv=3)             
                clf.fit(Train_temp,data_train_subset[y_predict_name])
                
                if(repeat_idx==0):
                    sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' RND_FraGVAE'+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score)+params['excess_space'])
                else:
                    sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' RND_FraGVAE'+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score) +' MSE '+str(np.mean(np.array(rnd_FragVAE_MSE_sample)))+params['excess_space'])
                sys.stdout.flush()
                
                if(-clf.best_score_ < -best_score):
                    rnd_model = clf.best_estimator_ 
                    best_num_features = num_features
                    best_score=clf.best_score_
                    early_stop = 0
                else:
                    early_stop = early_stop+1
                    if(early_stop==2):
                        break
            
        
            
            FragVAE_features_test = gen_features_from_Frag_VAE(data_test_subset['smiles'], rnd_fragvae_obj, rel_features,params)
            FragVAE_features_test = FragVAE_features_test[:,len(FragVAE_features_test[0])-best_num_features:len(FragVAE_features_test[0])]

            test_predictions = rnd_model.predict(FragVAE_features_test)
            iteration_mse = np.sum((test_predictions - data_test_subset[y_predict_name])**2)/test_num
            
            rnd_FragVAE_MSE_sample.append(iteration_mse)
            
            
            ''''
            ChemVAE preditionss 
            
            '''
            if(with_chemVAE):
                sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' ChemVAE'+params['excess_space'])
                sys.stdout.flush()
                            
                Z_ChemVAE = vae.encode(vae.smiles_to_hot(data_train_subset['smiles'],canonize_smiles=True))
                # Find valid mlecular fingerprints
                rel_features = find_Chem_VAE_features(Z_ChemVAE, data_train_subset[y_predict_name], params,num_features)    
                
                ChemVAE_features_train = gen_features_from_ChemVAE(Z_ChemVAE, rel_features)     
                
                early_stop = -100
                best_score = -np.inf
                best_num_features = -1
                for num_features in num_features_ChemVAE:
    
                    # Find valid mlecular fingerprints
                    Train_temp = ChemVAE_features_train[:,len(ChemVAE_features_train[0])-num_features:len(ChemVAE_features_train[0])]
                    reg = RandomForestRegressor( )                
                    clf = GridSearchCV(reg, Rnd_parameters,scoring='neg_mean_squared_error', cv=3)               
                    clf.fit(Train_temp,data_train_subset[y_predict_name])


                    if(repeat_idx==0):
                        sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' ChemVAE'+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score)+params['excess_space'])
                    else:
                        sys.stdout.write("\r" + 'Training. Num Samples: '+ str(num_samples[sample_idx])+' Repeat idx: '+str(repeat_idx)+' ChemVAE'+' New/BestScore '+str(-clf.best_score_)+'/'+str(-best_score) +' MSE '+str(np.mean(np.array(ChemVAE_MSE_sample)))+params['excess_space'])
                    sys.stdout.flush()
                    if(-clf.best_score_ < -best_score):
                        rnd_model = clf.best_estimator_ 
                        best_num_features = num_features
                        best_score=clf.best_score_
                        early_stop = 0
                    else:
                        early_stop = early_stop+1
                        if(early_stop==2):
                            break
                 
                Z_ChemVAE = vae.encode(vae.smiles_to_hot(data_test_subset['smiles'],canonize_smiles=True))
                ChemVAE_features_test = gen_features_from_ChemVAE(Z_ChemVAE, rel_features)
                ChemVAE_features_test = ChemVAE_features_test[:,len(ChemVAE_features_test[0])-best_num_features:len(ChemVAE_features_test[0])]
    
                test_predictions = rnd_model.predict(ChemVAE_features_test)
                iteration_mse = np.sum((test_predictions - data_test_subset[y_predict_name])**2)/test_num
                
                ChemVAE_MSE_sample.append(iteration_mse)
            
            
        ECFP_MSE_sample = np.array(ECFP_MSE_sample)
        FragVAE_MSE_sample  = np.array(FragVAE_MSE_sample)
        rnd_FragVAE_MSE_sample =  np.array(rnd_FragVAE_MSE_sample)
        if(with_chemVAE):
            ChemVAE_MSE_sample =  np.array(ChemVAE_MSE_sample)
        
        Expt_results.at[sample_idx,'ECFP_RMSE_mean'] = np.sqrt(np.mean(ECFP_MSE_sample))
        Expt_results.at[sample_idx,'ECFP_RMSE_std'] =0.5*1/( np.sqrt(np.mean(ECFP_MSE_sample)))*np.std(ECFP_MSE_sample)

            
        Expt_results.at[sample_idx,'FragVAE_RMSE_mean'] =np.sqrt(np.mean(FragVAE_MSE_sample))
        Expt_results.at[sample_idx,'FragVAE_RMSE_std'] =0.5*1/( np.sqrt(np.mean(FragVAE_MSE_sample)))*np.std(FragVAE_MSE_sample)
        
        Expt_results.at[sample_idx,'rnd_FragVAE_RMSE_mean'] = np.sqrt(np.mean(rnd_FragVAE_MSE_sample))
        Expt_results.at[sample_idx,'rnd_FragVAE_RMSE_std'] = 0.5*1/( np.sqrt(np.mean(rnd_FragVAE_MSE_sample)))*np.std(rnd_FragVAE_MSE_sample)
        if(with_chemVAE):
            Expt_results.at[sample_idx,'ChemVAE_RMSE_mean']= np.sqrt(np.mean(ChemVAE_MSE_sample))
            Expt_results.at[sample_idx,'ChemVAE_RMSE_std']= 0.5*1/( np.sqrt(np.mean(ChemVAE_MSE_sample)))*np.std(ChemVAE_MSE_sample)
        
        plt.figure(figsize=(10,5))
    
        
        plt.errorbar(np.array(num_samples), Expt_results['ECFP_RMSE_mean'], yerr=Expt_results['ECFP_RMSE_std'],label= 'ECFP')
        plt.errorbar(np.array(num_samples), Expt_results['rnd_FragVAE_RMSE_mean'], yerr=Expt_results['rnd_FragVAE_RMSE_std'],label= 'Random FragVAE')
        plt.errorbar(np.array(num_samples), Expt_results['FragVAE_RMSE_mean'], yerr=Expt_results['FragVAE_RMSE_std'],label= 'FragVAE')
        if(with_chemVAE):
            plt.errorbar(np.array(num_samples), Expt_results['ChemVAE_RMSE_mean'], yerr=Expt_results['ChemVAE_RMSE_std'],label= 'ChemVAE')

        
        
        plt.xscale('log', nonposx='clip')
    
       
        plt.ylabel('Root Mean squared error',wrap=True)
        plt.xlabel('Number of training data samples provided',wrap=True)
        
        plt.title('Efficiency of fingerprint space',wrap=True)
        plt.legend()
        
        
        plt.savefig('Fingerprint_Efficiency', dpi=100)
        plt.show()
        plt.close()
    
    Expt_results.to_csv(params['model_dir']+y_predict_name+'_experiment_Predict_Results2.csv',index=False)
    return  Expt_results


def compared_fingerprints_additives(model_num_frag = 3):
    
    #Generate a FraGVAE object with experiment number 5
    fragvae_obj = fg.FraGVAE(model_num_frag)
    params = fragvae_obj.params
    
    Training_data = pd.read_csv('data_lib/PolymerOSCAdditives_Training.csv')
    Test_data = pd.read_csv('data_lib/PolymerOSCAdditives_Test.csv')
    
    Training_data = Training_data.reset_index(drop=True)
    Test_data = Test_data.reset_index(drop=True)
    
    params['CFP_additive_regl2'] = [1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9]

    params['CFP_additive_num_features'] = [15,17,19,21,23,25,26,28,30]
    
    def LOOCV_TF(X, Y,input_Data_len,params,name=''):
        import gc
        import sys
        import os

        regl2_opt =0
        
        clear = lambda: os.system('cls')
        training_LOOCV_prediction_invalid = []
        training_LOOCV_prediction_valid = []
        training_LOOCV_prediction_invalid_temp =[]
        training_LOOCV_prediction_valid_temp =[]
        best_correct = 0
        count = 0
        
        df = pd.DataFrame([{'SE':[],'valid':[],'invalid':[],'num_features':[],'regl2':[],'roc_area':[]}])
        model_numbers = len(X)*len(params['CFP_additive_num_features'])*len(params['CFP_additive_regl2'])
        modelnum = -1
        min_error = np.inf


        for regl2 in params['CFP_additive_regl2']:                        
            for num_features in params['CFP_additive_num_features']:
                list_x = copy.deepcopy(list(X[:,len(X[0])-num_features:len(X[0])]))
                list_y = copy.deepcopy(list(Y))
                modelnum = modelnum+1
                SE =0
                training_LOOCV_prediction_invalid_temp =[]
                training_LOOCV_prediction_valid_temp =[]
                total_correct = 0
                for i in range(len(X)):
                    #i = len(X)-j-1
                    count = count+1
                    sys.stdout.write("\r" + 'model_'  + ': '+str(count)+'/' +str(model_numbers)+' total_correct: '+str(total_correct)+'/'+str(i))
                    sys.stdout.flush()
                    
                    x = np.array(list_x[0:i] + list_x[i+1:len(list_x)])
                    y = np.array(list_y[0:i] + list_y[i+1:len(list_y)])
                    test_x = np.array(list_x[i])
                    cur_model = gen_pred_TF(x, y,num_features,regl2=regl2  )
                    #result = cur_model.predict(np.array([test_x]))[0]
                    f = gen_dropout_fun(cur_model)
                    result, sigma = predict_with_uncertainty(f, [np.array([test_x])], 2, n_iter=1)

                    
                    #print(' '+str(Y[i])+' '+str(result[0][1]-result[0][0]))
                    training_LOOCV_prediction_invalid_temp.append(result[0][0])
                    training_LOOCV_prediction_valid_temp.append(result[0][1])
                    if(Y[i]==1):
                        SE = SE + (1-result[0][1])**2 + (result[0][0])**2
                    else:
                        SE = SE + (result[0][1])**2 + (1-result[0][0])**2
                    del cur_model
                    tf.contrib.keras.backend.clear_session()
                    gc.collect()
                    total_correct = total_correct+(((result[0][1]-result[0][0])>=0) and Y[i])*1+ (((result[0][1]-result[0][0])<0) and not(Y[i]))*1
                
                area = ROC_curve(np.array(training_LOOCV_prediction_valid_temp) -np.array(training_LOOCV_prediction_invalid_temp),Y)
                
                
                mol_class = ((np.array(training_LOOCV_prediction_valid_temp)-np.array(training_LOOCV_prediction_invalid_temp))>0)*1
                
                total_correct = np.sum(mol_class*Y)+np.sum((1-mol_class)*(1-Y))
                
                if(total_correct>best_correct or (total_correct==best_correct and SE<min_error)  ):
  
                    
                    
                    
                    
                    
                    n_checks = 4
                    area =area/(n_checks+1)
                    SE=SE/(n_checks+1)
                    total_correct= total_correct/(n_checks+1)
                    print()
                    print('Error is less than best: Checking addtional '+str(n_checks)+' times' +' Model Error ' + str(int(100*SE*(n_checks+1)/(i+1))/100.0) +' valid/invalid: '+str(np.sum(mol_class*Y))+'/'+str(np.sum((1-mol_class)*(1-Y)))+' num_features ' +str(num_features) +' regl2 ' +str(regl2))
                    print()
                    count2= 0
                    for j in range(n_checks): 
                        training_LOOCV_prediction_invalid_temp1 =[]
                        training_LOOCV_prediction_valid_temp1 =[]
                        for i in range(len(X)):
                            sys.stdout.write("\r" + 'Regression_Checking_best SE:'  +str(int(100*SE*(n_checks+1)/(j+1+i/len(X)))/100.0 )+ ' '+str(count2)+'/' +str(n_checks*len(X)) +' total_correct '+str(int(100*total_correct*(n_checks+1)/(j+1))/100.0 ))
                            sys.stdout.flush()
                            count2=count2+1

                            
                            x = np.array(list_x[0:i] + list_x[i+1:len(list_x)])
                            y = np.array(list_y[0:i] + list_y[i+1:len(list_y)])
                            test_x = np.array(list_x[i])
                            cur_model = gen_pred_TF(x, y,num_features,regl2=regl2  )

                            f = gen_dropout_fun(cur_model)
                            result, sigma = predict_with_uncertainty(f, [np.array([test_x])], 2, n_iter=1)
  
                            if(Y[i]==1):
                                SE = SE + ((1-result[0][1])**2 + (result[0][0])**2)/(n_checks+1)
                            else:
                                SE = SE + ((result[0][1])**2 + (1-result[0][0])**2)/(n_checks+1)
                            del cur_model
                            tf.contrib.keras.backend.clear_session()
                            gc.collect()
                            training_LOOCV_prediction_invalid_temp1.append(result[0][0])
                            training_LOOCV_prediction_valid_temp1.append(result[0][1])
                        
                        mol_class = ((np.array(training_LOOCV_prediction_valid_temp1)-np.array(training_LOOCV_prediction_invalid_temp1))>0)*1

                        total_correct =total_correct+ np.sum(mol_class*Y)/(n_checks+1)+np.sum((1-mol_class)*(1-Y))/(n_checks+1)
                        area =area+ ROC_curve(np.array(training_LOOCV_prediction_valid_temp1) -np.array(training_LOOCV_prediction_invalid_temp1),Y)/(n_checks+1)
                            
                    if(total_correct>best_correct or (total_correct==best_correct and SE<min_error) ):
                        

                        print('Update Model')
                        

                        regl2_opt=regl2 
                        min_error= SE
                        num_features_opt = num_features
                        training_LOOCV_prediction_valid = copy.deepcopy(training_LOOCV_prediction_valid_temp)
                        training_LOOCV_prediction_invalid = copy.deepcopy(training_LOOCV_prediction_invalid_temp)
                        best_correct = total_correct
                    
                print()
                print('Model Error ' + str(SE) +' TotalCorrect '+str(total_correct)+' valid/invalid: '+str(np.sum(mol_class*Y))+'/'+str(np.sum((1-mol_class)*(1-Y)))+' num_features ' +str(num_features) +' regl2 ' +str(regl2)+' area '+str(area))
                df.loc[modelnum,'SE']=SE
                df.loc[modelnum,'valid']=str(np.sum(mol_class*Y))
                df.loc[modelnum,'invalid']=str(np.sum((1-mol_class)*(1-Y)))
                df.loc[modelnum,'num_features']=num_features
                df.loc[modelnum,'regl2']=regl2
                df.loc[modelnum,'total_correct']=total_correct
                df.loc[modelnum,'roc_area']=area
            clear()  
        regression_params ={}
        regression_params['num_features_opt']=num_features_opt
        regression_params['regl2_opt']=regl2_opt 
        df.to_csv(params['model_dir']+'regression_hyper_'+name+'.csv',index=False)

        
        return training_LOOCV_prediction_invalid,training_LOOCV_prediction_valid, regression_params
    
    
    def ensemble_models(reg_params,features_test,features_train,Test_data,Training_data,num_average):
        test_valid=np.zeros(len(features_test))
        MSE = np.zeros(len(features_test))
        Z_score = np.zeros(len(features_test))
        Z_avg = np.zeros((len(features_test),2))
        Z_sig = np.zeros((len(features_test),2))
        features_test = np.array(features_test[:,len(features_test[0])-reg_params['num_features_opt']:len(features_test[0])])
        features_train = np.array(features_train[:,len(features_train[0])-reg_params['num_features_opt']:len(features_train[0])])
        for j in range(num_average):
            cur_model = gen_pred_TF(features_train, Training_data['metrics'],reg_params['num_features_opt'],regl2=reg_params['regl2_opt']  )
            f = gen_dropout_fun(cur_model)
            Z, sigma = predict_with_uncertainty(f, [features_test], 2, n_iter=100)
            for i in range(len(features_test)):
                if(Test_data.at[i,'metrics']==0):
                    MSE[i] = MSE[i] + ((1-Z[i,0])**2+ (Z[i,1])**2)/num_average
                else:
                    MSE[i] = MSE[i] + ((Z[i,0])**2+ (1-Z[i,1])**2)/num_average
                
                Z_avg[i,0] =Z[i,0]+Z_avg[i,0]
                Z_avg[i,1] =Z[i,1]+Z_avg[i,1]
                Z_sig[i,0] = Z_sig[i,0]+sigma[i,0]**2/num_average
                Z_sig[i,1] = Z_sig[i,1]+sigma[i,1]**2/num_average            
            import gc
            del cur_model
            tf.contrib.keras.backend.clear_session()
            gc.collect()  
        for i in range(len(features_test)):
            if(Z_sig[i,1]==0 or Z_sig[i,0]==0):
                Z_sig[i,1] =1
                Z_sig[i,0]==1
            
            Z_score[i] =Z_score[i]+ (-Z_avg[i,0]+Z_avg[i,1])/num_average
            
 
        for i in range(len(features_test)):          
            test_valid[i]=(Z_score[i]>0)*1
        for i in range(len(features_test)):
            print('Test/Model '+str(Test_data.at[i,'metrics'])+'/' +str(int(test_valid[i]))+'     Z_score ' +str(int(Z_score[i]*100)/100.0)+' MSE ' +str(int(MSE[i]*100)/100.0))
        area = ROC_curve(Z_score,np.array(Test_data['metrics']))

        return test_valid, Z_score, MSE,area

    
    '''
    ChemVAE preditionss 
    
    '''
    num_average =20 
    print('ChemVAE')

    chem_vae_model_dir = '/chemical_vae/models/zinc'
    curdir = os.getcwd()
    parent_path = Path(curdir).parent.as_posix()
    vae = VAEUtils(directory=parent_path+chem_vae_model_dir)
    
    
    
    Z_ChemVAE = vae.encode(vae.smiles_to_hot(Training_data['smiles'],canonize_smiles=True))
    # Find valid mlecular fingerprints
    rel_features = find_Chem_VAE_features(Z_ChemVAE, Training_data['metrics'], params,max(params['CFP_additive_num_features']))    
    ChemVAE_features_train = gen_features_from_ChemVAE(Z_ChemVAE, rel_features)
     
    Z_ChemVAE = vae.encode(vae.smiles_to_hot(Test_data['smiles'],canonize_smiles=True))
    ChemVAE_features_test = gen_features_from_ChemVAE(Z_ChemVAE, rel_features)
   
    
    training_LOOCV_prediction_invalid_chemVAE,training_LOOCV_prediction_valid_chemVAE, reg_params = LOOCV_TF(ChemVAE_features_train, Training_data['metrics'],max(params['CFP_additive_num_features']),params,name='ChemVAE')
    Training_data['ChemVAE_invalid'] = pd.Series(training_LOOCV_prediction_invalid_chemVAE)
    Training_data['ChemVAE_valid'] = pd.Series(training_LOOCV_prediction_valid_chemVAE)

    test_valid, Z_score, MSE,area = ensemble_models(reg_params,ChemVAE_features_test,ChemVAE_features_train,Test_data,Training_data,num_average)
    Test_data['ChemVAE'] = pd.Series(test_valid)
    Test_data['ChemVAE_Z_Score'] = pd.Series(Z_score)
    Test_data['ChemVAE_MSE'] = pd.Series(MSE)
    Test_data['ChemVAE_area'] = pd.Series(area)
    
    '''
    FragVAE preditionss 
    '''
    #Generate a FraGVAE object with experiment number 5
    fragvae_obj = fg.FraGVAE(model_num_frag)
    fragvae_obj.load_models(rnd=False,testing = False)
    
    print()
    print('FragVAE')
    
    rel_features = find_Frag_VAE_features(Training_data['smiles'], np.array(Training_data['metrics']), params,fragvae_obj,max(params['CFP_additive_num_features']),with_F1 = True)    
    features_train = gen_features_from_Frag_VAE(Training_data['smiles'], fragvae_obj, rel_features,params)    
    features_test = gen_features_from_Frag_VAE(Test_data['smiles'], fragvae_obj, rel_features,params)

    training_LOOCV_prediction_invalid_FragVAE,training_LOOCV_prediction_valid_FragVAE, reg_params = LOOCV_TF(features_train, Training_data['metrics'],max(params['CFP_additive_num_features']),params,name='FraGVAE')
    Training_data['FragVAE_invalid'] = pd.Series(training_LOOCV_prediction_invalid_FragVAE)
    Training_data['FragVAE_valid'] = pd.Series(training_LOOCV_prediction_valid_FragVAE)
    print(reg_params)
    

    
    test_valid, Z_score, MSE,area = ensemble_models(reg_params,features_test,features_train,Test_data,Training_data,num_average)
    Test_data['FragVAE'] = pd.Series(test_valid)
    Test_data['FragVAE_Z_Score'] = pd.Series(Z_score)
    Test_data['FragVAE_MSE'] = pd.Series(MSE)
    Test_data['FragVAE_area'] = pd.Series(area)

    Training_data.to_csv(params['model_dir']+'Experimental_Training_set_reg.csv',index=False)
    Test_data.to_csv(params['model_dir']+'Experimental_Test_reg.csv',index=False)


    
    '''
    
    Rnd_FragVAE preditionss 
    
    '''
    print()
    print('Rnd_FragVAE')
    
    rnd_fragvae_obj = fg.FraGVAE(3)
    rnd_fragvae_obj.load_models(rnd=True,testing = False) 
    
    rel_features = find_Frag_VAE_features(Training_data['smiles'], np.array(Training_data['metrics']), params,rnd_fragvae_obj,max(params['CFP_additive_num_features']),with_F1 = True)    

    Rnd_FragVAE_features_train = gen_features_from_Frag_VAE(Training_data['smiles'], rnd_fragvae_obj, rel_features,params)
    Rnd_FragVAE_features_test = gen_features_from_Frag_VAE(Test_data['smiles'], rnd_fragvae_obj, rel_features,params)

    training_LOOCV_prediction_invalid_rnd_FragVAE,training_LOOCV_prediction_valid_rnd_FragVAE,reg_params = LOOCV_TF(Rnd_FragVAE_features_train, Training_data['metrics'],max(params['CFP_additive_num_features']),params,name='rnd_FraGVAE')
    Training_data['rnd_FragVAE_invalid'] = pd.Series(training_LOOCV_prediction_invalid_rnd_FragVAE)
    Training_data['rnd_FragVAE_valid'] = pd.Series(training_LOOCV_prediction_valid_rnd_FragVAE)
    
    
    
    test_valid, Z_score, MSE,area = ensemble_models(reg_params,Rnd_FragVAE_features_test,Rnd_FragVAE_features_train,Test_data,Training_data,num_average)
    Test_data['rnd_FragVAE'] = pd.Series(test_valid)
    Test_data['rnd_FragVAE_Z_Score'] = pd.Series(Z_score)
    Test_data['rnd_FragVAE_MSE'] = pd.Series(MSE)
    Test_data['rnd_FragVAE_area'] = pd.Series(area)
    Training_data.to_csv(params['model_dir']+'Experimental_Training_set_reg.csv',index=False)
    Test_data.to_csv(params['model_dir']+'Experimental_Test_reg.csv',index=False)   
    ''''
    ECFP preditionss 
    
    '''
    print()
    print('ECFP')
    # Find valid mlecular fingerprints
    list_ECFPs = find_ECFP(np.array(Training_data['smiles']), np.array(Training_data['metrics']), params,max(params['CFP_additive_num_features']), ECFP_degree=3)
    
    ECFP_features_train = gen_features_from_ECFPS(np.array(Training_data['smiles']), list_ECFPs, max(params['CFP_additive_num_features']),params)
    ECFP_features_test = gen_features_from_ECFPS(np.array(Test_data['smiles']), list_ECFPs, max(params['CFP_additive_num_features']),params)
 
    training_LOOCV_prediction_invalid_ECFP,training_LOOCV_prediction_valid_ECFP,reg_params  = LOOCV_TF(ECFP_features_train, Training_data['metrics'],max(params['num_features_expt']),params,name='ECFP')
    Training_data['ECFP_invalid'] = pd.Series(training_LOOCV_prediction_invalid_ECFP)
    Training_data['ECFP_valid'] = pd.Series(training_LOOCV_prediction_valid_ECFP)
    
    
    
    test_valid, Z_score, MSE,area = ensemble_models(reg_params,ECFP_features_test,ECFP_features_train,Test_data,Training_data,num_average)
    Test_data['ECFP'] = pd.Series(test_valid)
    Test_data['ECFP_Z_Score'] = pd.Series(Z_score)
    Test_data['ECFP_MSE'] = pd.Series(MSE)
    Test_data['ECFP_area'] = pd.Series(area)
    


    Training_data.to_csv(params['model_dir']+'Experimental_Training_set_reg.csv',index=False)
    Test_data.to_csv(params['model_dir']+'Experimental_Test_reg.csv',index=False)
  

    Training_data.to_csv(params['model_dir']+'Experimental_Training_set_reg.csv',index=False)
    Test_data.to_csv(params['model_dir']+'Experimental_Test_reg.csv',index=False)   
    #hyper_optimization.to_csv('data_lib/Experimental_RND_Forrest_hyper_optimization.csv',index=False)
    
    return  

def RND_forrest_uncertainty(RND_forrect, X):
    RND_trees_list = RND_forrect.estimators_
    tree_perdictions = []
    for tree in RND_trees_list:
        tree_perdictions.append(tree.predict(X))
    tree_perdictions=np.array(tree_perdictions)
    pred_mean = np.mean(tree_perdictions)
    pred_std = np.std(tree_perdictions)
    return pred_mean,pred_std
                
                
def select_rel_features(Z_1 , Z_HO,rel_feature,params=[]):
    rel_features=copy.deepcopy(rel_feature)
    Z=list(Z_1)+list( Z_HO)
    Z=np.array(Z)
    features = []
    #print(rel_features)
    for i in range(0,int(len(Z))):
        index = np.argmin(rel_features)
        if(rel_features[index]!=np.inf):
            features.append(Z[index])
            rel_features[index]=np.inf

    features = np.array(features)
    return features

def find_ECFP(smiles, y, params,num_finger, ECFP_degree=3):
    # Find top ECFP features with highest pearson correlation coefficient
    y = np.array(y)
    ECFP = {}
    smile_index = -1
    for smile in smiles:
        smile_index=smile_index+1
        mol = fg.convert_mol_smile_tensor.smile_to_mol(smile, params)
        ECFPs_mol =Chem.GetMorganFingerprint(mol,ECFP_degree).GetNonzeroElements()
        for ECFP_idx in list(ECFPs_mol.keys()):
            
            if(ECFP_idx in ECFP ):
                ECFP[ECFP_idx].append(np.array([ECFPs_mol[ECFP_idx],y[smile_index]]))
            else:
                ECFP[ECFP_idx] = [np.array([ECFPs_mol[ECFP_idx],y[smile_index]])]

                
    mean_y = np.mean(y)
    n = len(smiles)
    mod_std_y =    np.sqrt(np.sum(y**2) - n*mean_y**2)
    
    list_ECFPs = [-np.inf]*int(num_finger)
    list_PCs = [-np.inf]*int(num_finger)
    for ECFP_idx in list(ECFP.keys()):
        data = np.array(ECFP[ECFP_idx])
        x_ecfp = data[:,0]
        y_ecfp = data[:,1]
        mean_x = np.sum(x_ecfp)/n
        mod_std_x =   np.sqrt(np.sum(x_ecfp**2) - n*mean_x**2)
        PC_numerator =  np.sum(y_ecfp*x_ecfp) - n*mean_x*mean_y
        Pearson_coefficent = np.abs(PC_numerator/(mod_std_y*mod_std_x+0.000000001))
        
        
        insert_idx = np.searchsorted(list_PCs, Pearson_coefficent)
        if(insert_idx>0):
            list_ECFPs.insert(insert_idx, ECFP_idx)
            list_PCs.insert(insert_idx, Pearson_coefficent)
            list_ECFPs = list_ECFPs[1:len(list_ECFPs)]
            list_PCs = list_PCs[1:len(list_PCs)]
    
    return list_ECFPs
         

def gen_features_from_ECFPS(smiles, list_ECFPs, num_features,params,ECFP_degree=3):
    # select set features from ECFP

    # generates a list of all 
    X_features = []
    
    for smile in smiles:
        mol = fg.convert_mol_smile_tensor.smile_to_mol(smile, params)
        ECFPs_mol = Chem.GetMorganFingerprint(mol,ECFP_degree).GetNonzeroElements()
        ECFPs_mol_list = list(ECFPs_mol.keys())
        mol_features = np.zeros(int(num_features))
        
        for ECFP_idx in range(0,len(list_ECFPs)): 
            if(list_ECFPs[ECFP_idx] in ECFPs_mol_list):
                mol_features[ECFP_idx] = ECFPs_mol[list_ECFPs[ECFP_idx]]
        X_features.append(mol_features)
        
    X_features = np.array(X_features)
    return X_features
def find_Chem_VAE_features(Z_ChemVAE, y, params,num_finger):
    # Find top ChemVAE features with highest pearson correlation coefficient

    
    y = np.array(y)
    X = Z_ChemVAE
    
    mean_y = np.mean(y)
    n = len(Z_ChemVAE)
    mod_std_y =    np.sqrt(np.sum(y**2) - n*mean_y**2)
    
    
    
    PCs = -np.inf*np.ones(len(Z_ChemVAE[0]))
    rel_features = np.zeros(len(Z_ChemVAE[0]))
    
    for feature_idx in range(0,len(X[0])):
        x = X[:,feature_idx]
        mean_x = np.sum(x )/n
        mod_std_x =   np.sqrt(np.sum(x **2) - n*mean_x**2)
        PC_numerator =  np.sum(y*x) - n*mean_x*mean_y
        Pearson_coefficent = np.abs(PC_numerator/(mod_std_y*mod_std_x+0.000000001))
        

        PCs[feature_idx] = Pearson_coefficent
            
    
    for i in range(0,int(num_finger)):
        best_idx =np.argmax(PCs)
        rel_features[best_idx] = num_finger-i
        PCs[best_idx] = -np.inf
         
         
    return rel_features

def find_Frag_VAE_features(smiles, y, params,model,num_finger,with_F1 = False):
    '''
    Find top FragVAE features with highest pearson correlation coefficient
    '''
    y = np.array(y)
    X_Z1 = []
    X_ZHO = []
    for smile in smiles:
        #print(smile)
        atoms, edges, bonds =  fg.convert_mol_smile_tensor.smile_to_tensor(smile, params,FHO_Ring_feature=True)
        atoms = np.array([atoms])
        edges = np.array([edges])
        bonds = np.array([bonds])
        Z_1 , Z_HO,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS =  model.Z_encoder(atoms, bonds ,edges)
        
        X_Z1.append(Z_1)
        X_ZHO.append(Z_HO-ZHO_Z1)
        
    X_Z1 = np.array(X_Z1)[:,0,:]
    X_ZHO = np.array(X_ZHO)[:,0,:]
    
    X = np.concatenate((X_Z1,X_ZHO),axis = -1)
    
    mean_y = np.mean(y)
    n = len(smiles)
    mod_std_y =    np.sqrt(np.sum(y**2) - n*mean_y**2)
    
    
    PCs = -np.inf*np.ones(params['finger_print']+params['FHO_finger_print'])
    rel_features = np.inf*np.ones(params['finger_print']+params['FHO_finger_print'])
    
    for feature_idx in range(0,len(X[0])):
        x = X[:,feature_idx]
        mean_x = np.sum(x )/n
        mod_std_x =   np.sqrt(np.sum(x **2) - n*mean_x**2)
        PC_numerator =  np.sum(y*x) - n*mean_x*mean_y
        Pearson_coefficent = np.abs(PC_numerator/(mod_std_y*mod_std_x+0.000000001))
        PCs[feature_idx] = Pearson_coefficent
    
    for i in range(0,int(num_finger)):
        best_idx =np.argmax(PCs)
        rel_features[best_idx] = num_finger-i
        PCs[best_idx] = -np.inf
         
    return rel_features
            
def gen_features_from_Frag_VAE(smiles, model,rel_features,params):
    '''
    Select releavent features from FagVAE encoding of molecule
    '''
    X_features = []
    
    for smile in smiles:
        
        atoms, edges, bonds =  fg.convert_mol_smile_tensor.smile_to_tensor(smile, params,FHO_Ring_feature=True)
        atoms = np.array([atoms])
        edges = np.array([edges])
        bonds = np.array([bonds])
        Z_1 , Z_HO,ZHO_Z1,ZHO_Z2,ZHO_ZR,ZHO_ZS =  model.Z_encoder(atoms, bonds ,edges)
        Z_1=(Z_1)[0]
        Z_HO=(Z_HO-ZHO_Z1)[0]
        X_features.append(select_rel_features(Z_1 , Z_HO,rel_features,params))
        
    X_features = np.array(X_features)
        
    return X_features

def gen_features_from_ChemVAE(Z_ChemVAE, rel_features):
    '''
    Select releavent features from ChemVAE encoding of molecule
    '''

    X_features = []
    
    for Z in Z_ChemVAE:          
        features = select_rel_features(Z , Z,rel_features)
        X_features.append(copy.deepcopy(features))
    X_features = np.array(X_features)
        
    return X_features

def  ROC_curve(Z,Z_truth):
    '''
    Function to calculate the area under Receiver operating characteristic
    
    '''
    from scipy.interpolate import CubicSpline
    threshold =np.array(range(1000,-1001,-1))/1000
    lenx = len(threshold)
    
    TP=np.zeros(lenx)
    FP=np.zeros(lenx)
    
    for threshold_idx in range(lenx):

        for Z_idx in range(0,len(Z)):
            if(Z[Z_idx]>threshold[threshold_idx]):
                if(Z_truth[Z_idx]==1):
                    TP[threshold_idx] = TP[threshold_idx]+1/(np.sum(Z_truth))
                else:
                    FP[threshold_idx] = FP[threshold_idx]+1/(len(Z)-np.sum(Z_truth))
    x=[FP[0]]
    y=[]
    temp =[TP[0]]
    for i in range(len(TP)):
        if(x[-1]<FP[i]):
            x.append(FP[i])
            y.append(np.average(np.array(temp)))
            temp =[TP[i]]
        elif(x[-1]==FP[i]):
            temp.append(TP[i])
    y.append(np.average(np.array(temp)))
    x = np.array(x)
    y = np.array(y)

    area = 0
    for i in range(len(x)-1):
        area = area + (x[i+1]-x[i])*(y[i+1]+y[i])/2
    return area 

def gen_pred_TF(Z, Y,input_dim,regl2=0.01,epochs=3000, printMe = False,early_stop=0):
    # Define the input layers
    freatures = Input(name='freatures', shape=(input_dim,), dtype='float32')


    output_model = Dense(2,activation = 'softmax',kernel_regularizer= regularizers.l2(regl2) )(freatures) 

    optimizer = optimizers.Adam()

    model = models.Model(inputs=[freatures], outputs=output_model)
    model.compile(optimizer=optimizer, loss='mse')
    Y_clases = np.zeros((len(Y),2))
    for i in range(len(Y)):
        if(Y[i]==0):
            Y_clases[i,Y[i]]=1
        else:
            Y_clases[i,Y[i]]=1
    model.fit([Z], Y_clases,epochs= epochs,batch_size=int(len(Y_clases)), shuffle=True,verbose=0) 
    return model

def gen_dropout_fun(model):
    # for some model with dropout ...
    f = tf.keras.backend.function([model.layers[0].input, tf.keras.backend.learning_phase()],
                   [model.layers[-1].output])
    return f

def predict_with_uncertainty(f, x, num_class, n_iter=100):
    result = np.zeros((n_iter,) + (x[0].shape[0], num_class) )

    for i in range(n_iter):
        result[i,:, :] = f((x[0], 1))[0]

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty

def reset_weights(model):
    weights = model.get_weights()
    new_weights = []
    for weight in weights: 
        if(len(weight.shape)>1):
            limit= np.sqrt(6/(weight.shape[1]+weight.shape[0]))
            new_weights.append(np.random.uniform(-limit,limit,(weight.shape[0],weight.shape[1])))
        elif(len(weight.shape)==1):
            limit= np.sqrt(6/(weight.shape[0]))
            new_weights.append(np.random.uniform(-limit,limit,weight.shape[0]))
        else:
            limit= np.sqrt(6/(1))
            new_weights.append(np.random.uniform(-limit,limit,(1,))[0])
    model.set_weights(new_weights)
    return model        