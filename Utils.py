# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 05:54:54 2024

@author: Jahirul
"""
# import numpy as np
# 


import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tcn import compiled_tcn, tcn_full_summary
import seaborn as sns
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[abs(z_scores) < threshold]

def getregdata():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list)*0.75)
    train_dir_list=list(individual_list[:train_dir_len])
    val_dir_len=int((len(individual_list)-train_dir_len)*0.67)
    val_dir_list=list(individual_list[train_dir_len:train_dir_len+val_dir_len])
    test_dir_len=len(individual_list)-train_dir_len-val_dir_len
    test_dir_list=list(individual_list[train_dir_len+val_dir_len:])
    
    #Function to calculate Symmetric Mean Absolute Percentage Error
    def smape(a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))
    
    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    count2=0
    count3=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
        elif(test_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [test_eye_df, dff]
            test_eye_df=pd.concat(frames)
            count2+=1
        if(val_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [val_eye_df, dff]
            val_eye_df=pd.concat(frames)
            count1+=1
    
    trainY_eye=training_eye_df[['fms']]
    trainX_eye=training_eye_df.drop(columns=['fms'])
    
    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    y_scaler_for_trainY_eye=MinMaxScaler()
    trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)
    
    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    #for sampling val datafram
    val_eye_df.shape
    valY_eye=val_eye_df[['fms']]
    valX_eye=val_eye_df.drop(columns=['fms'])
    
    #set scaling
    # x_scaler_for_valX_eye=MinMaxScaler()
    # y_scaler_for_valY_eye=MinMaxScaler()
    valY_eye=y_scaler_for_trainY_eye.transform(valY_eye)
    valX_eye=x_scaler_for_trainX_eye.transform(valX_eye)
    
    #X_val_eye, Y_val_eye = np.array([]),np.array([]) 
    for i in tqdm(range(int((val_eye_df.shape[0])/28))):
        if (i==0):
            X_val_eye = np.expand_dims(valX_eye[i*28:(i+1)*28,:],0)
            Y_val_eye = np.expand_dims(valY_eye[i*28],0)
        else:
            X_val_eye=np.concatenate((X_val_eye,np.expand_dims(valX_eye[i*28:(i+1)*28,:],0)),0)
            Y_val_eye = np.concatenate((Y_val_eye,np.expand_dims(valY_eye[i*28],0)),0)
    
    #for sampling test datafram
    test_eye_df.shape
    testY_eye=test_eye_df[['fms']]
    testX_eye=test_eye_df.drop(columns=['fms'])
    #set scaling
    # x_scaler_for_testX_eye=MinMaxScaler()
    # y_scaler_for_testY_eye=MinMaxScaler()
    testY_eye=y_scaler_for_trainY_eye.transform(testY_eye)
    testX_eye=x_scaler_for_trainX_eye.transform(testX_eye)
    
    #X_test_eye, Y_test_eye = np.array([]),np.array([]) 
    for i in tqdm(range(int((test_eye_df.shape[0])/28))):
        if (i==0):
            X_test_eye = np.expand_dims(testX_eye[i*28:(i+1)*28,:],0)
            Y_test_eye = np.expand_dims(testY_eye[i*28],0)
        else:
            X_test_eye=np.concatenate((X_test_eye,np.expand_dims(testX_eye[i*28:(i+1)*28,:],0)),0)
            Y_test_eye = np.concatenate((Y_test_eye,np.expand_dims(testY_eye[i*28],0)),0)
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    print( "X_val_eye.shape = "+str(X_val_eye.shape))
    print( "Y_val_eye = "+str(Y_val_eye.shape))
    print("X_test_eye.shape = "+str(X_test_eye.shape))
    print("Y_test_eye.shape = "+str(Y_test_eye.shape))
    return X_train_eye, Y_train_eye, X_val_eye, Y_val_eye, X_test_eye, Y_test_eye,x_scaler_for_trainX_eye,y_scaler_for_trainY_eye


def getclassificationXY():
    
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
        
            
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
        
    print("X_train_eye= "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))

    return X_train_eye, Y_train_eye, x_scaler_for_trainX_eye, le

def getregXY():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()

    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
    
    trainY_eye=training_eye_df[['fms']]
    trainX_eye=training_eye_df.drop(columns=['fms'])
    
    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    y_scaler_for_trainY_eye=MinMaxScaler()
    trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)
    
    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    return X_train_eye, Y_train_eye,x_scaler_for_trainX_eye,y_scaler_for_trainY_eye

def getreg_cross_XY():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()

    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
    
    trainY_eye=training_eye_df[['fms']]
    trainX_eye=training_eye_df.drop(columns=['fms'])
    
    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    y_scaler_for_trainY_eye=MinMaxScaler()
    trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)
    
    
    # for i in tqdm(range(int(training_eye_df.shape[0]-28))):
    #     if (i==0):
    #         X_train_eye = np.expand_dims(trainX_eye[i:i+28,:],0)
    #         Y_train_eye = np.expand_dims(trainY_eye[i+28],0)
    #     else:
    #         X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i:i+28,:],0)),0)
    #         Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i+28],0)),0)
            
            
    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    # X_train_data,X_test_data,Y_train_data,Y_test_data=[],[],[],[]

    # for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, Y_train_eye)):
    #     x_train= X_train_eye[train_indices]
    #     y_train= Y_train_eye[train_indices]
        
    #     x_test= X_train_eye[test_indices]
    #     y_test=Y_train_eye[test_indices]
        
        
    #     X_train_data.append(X_train_eye[train_indices]), X_test_data.append(X_train_eye[test_indices]),
    #     Y_train_data.append(Y_train_eye[test_indices]), Y_test_data.append(Y_train_eye[test_indices]),

    # return X_data,Y_data, x_scaler_for_trainX_eye, y_scaler_for_trainY_eye


def getclassificationdata():
    
    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    # X_data,Y_data=[],[]

    # for i, (X_indices, Y_indices) in enumerate(skf.split(train_dir_list)):
    #     X_data.append(X_train_eye[X_indices]),Y_data.append(Y_train_eye[Y_indices])
    
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list)*0.75)
    train_dir_list=list(individual_list[:train_dir_len])
    val_dir_len=int((len(individual_list)-train_dir_len)*0.67)
    val_dir_list=list(individual_list[train_dir_len:train_dir_len+val_dir_len])
    test_dir_len=len(individual_list)-train_dir_len-val_dir_len
    test_dir_list=list(individual_list[train_dir_len+val_dir_len:])
    # test_dir_list=[8]
    # val_dir_list=[7,9]
    #Function to calculate Symmetric Mean Absolute Percentage Error
    def smape(a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    count2=0
    count3=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
        elif(test_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [test_eye_df, dff]
            test_eye_df=pd.concat(frames)
            count2+=1
        if(val_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [val_eye_df, dff]
            val_eye_df=pd.concat(frames)
            count3+=1
            
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    #for sampling val datafram
    val_eye_df=val_eye_df.drop(columns=['fms'])
    val_eye_df.shape
    valY_eye=val_eye_df[['CSS']]
    valX_eye=val_eye_df.drop(columns=['CSS'])

    #set scaling
    # x_scaler_for_valX_eye=MinMaxScaler()
    # y_scaler_for_valY_eye=MinMaxScaler()
    valY_eye=le.transform(valY_eye)
    valX_eye=x_scaler_for_trainX_eye.transform(valX_eye)

    #X_val_eye, Y_val_eye = np.array([]),np.array([]) 
    for i in tqdm(range(int((val_eye_df.shape[0])/28))):
        if (i==0):
            X_val_eye = np.expand_dims(valX_eye[i*28:(i+1)*28,:],0)
            Y_val_eye = np.expand_dims(valY_eye[i*28],0)
        else:
            X_val_eye=np.concatenate((X_val_eye,np.expand_dims(valX_eye[i*28:(i+1)*28,:],0)),0)
            Y_val_eye = np.concatenate((Y_val_eye,np.expand_dims(valY_eye[i*28],0)),0)

    #for sampling test datafram
    test_eye_df=test_eye_df.drop(columns=['fms'])
    test_eye_df.shape
    testY_eye=test_eye_df[['CSS']]
    testX_eye=test_eye_df.drop(columns=['CSS'])
    #set scaling
    # x_scaler_for_testX_eye=MinMaxScaler()
    # y_scaler_for_testY_eye=MinMaxScaler()
    testY_eye=le.transform(testY_eye)
    testX_eye=x_scaler_for_trainX_eye.transform(testX_eye)

    #X_test_eye, Y_test_eye = np.array([]),np.array([]) 
    for i in tqdm(range(int((test_eye_df.shape[0])/28))):
        if (i==0):
            X_test_eye = np.expand_dims(testX_eye[i*28:(i+1)*28,:],0)
            Y_test_eye = np.expand_dims(testY_eye[i*28],0)
        else:
            X_test_eye=np.concatenate((X_test_eye,np.expand_dims(testX_eye[i*28:(i+1)*28,:],0)),0)
            Y_test_eye = np.concatenate((Y_test_eye,np.expand_dims(testY_eye[i*28],0)),0)
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    print( "X_val_eye.shape = "+str(X_val_eye.shape))
    print( "Y_val_eye = "+str(Y_val_eye.shape))
    print("X_test_eye.shape = "+str(X_test_eye.shape))
    print("Y_test_eye.shape = "+str(Y_test_eye.shape))
    return X_train_eye, Y_train_eye, X_val_eye, Y_val_eye, X_test_eye, Y_test_eye, x_scaler_for_trainX_eye, le

def getclassification_processed_XY():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import numpy as np
    import logging
    from datetime import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import matplotlib.dates as mdates

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from scipy.signal import resample
    import seaborn as sns

    from scipy.signal import butter, filtfilt, resample
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import callbacks
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    from scipy.signal import butter, filtfilt, resample
    from tensorflow.keras.utils import to_categorical
    # Connect with Google Drive
    # %cd '/content/drive/MyDrive/forecast_data'

    base_path = os.getcwd()+'\\forecast_data'
    base_path +="\\"
    meta_file = base_path+ 'meta_data.csv'
    meta_data = pd.read_csv(meta_file)
    meta_data

    # Funciton to filter data from the meta data
    def get_filtered_meta_data(individual_num, simulation_name):
      filtered_data = meta_data
      if simulation_name:
          filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
      if individual_num:
          filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]

      return filtered_data

    # Sort function to sort the data according to time

    # For Umama's Data (Need to update her data to include milliseconds to work with other one):
    # format='%H-%M-%S'

    # For Sam's Data:
    # format='%H-%M-%S-%f'
    def sort_by_time(data):
      data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
      data = data.sort_values(by='Time')
      return data
    # The function to select data
    def get_data(base_path=base_path, simulation_name=None, individual_num=None,
                data_type='', sort=True):
        pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
        file_paths = pruned_meta_data.apply(
            lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                    x[data_type]),
            axis=1)
        data_list = []
        for file in file_paths:
            head_data = pd.read_csv(base_path + file)
            data_list.append(head_data)
        data = pd.concat(data_list, ignore_index=True)
        if sort:
            data = sort_by_time(data)
        return data


    train_individual_list = [1,2,3,4,7,8,9,11,12,13,14] # The list of individuals in the data file
    train_simulation_list = ['sea','roller','beach','room','walk'] # List of Simulation in data file


    # We want to select all the different data time.
    data_type = ['hr', 'eda', 'eye', 'head']
    total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                       individual_num=train_individual_list, data_type = 'eye')
    total_data.dropna(axis=1,inplace=True)
    total_data

    #we can define class based on quantile or mean
    train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]


    def classify_fms(value, quantiles):
        if value <= quantiles[0.25]:
            return 0
        elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
            return 1
        elif value> quantiles[0.75]:
             return 2


    # # Ensure 'fms_class_subjectwise' column is properly defined with categories
    # total_data['fms_class_subjectwise'] = pd.Categorical(total_data['fms_class_subjectwise'], categories=[0, 1, 2, 3])

    for individual in train_individual_list:
        sub_individual_data = total_data[total_data['individual'] == individual]
        # sub_individual_data=remove_outliers_zscore(sub_individual_data, 'fms')
        quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
        print('quantiles\n',quantiles)

        # Apply classification and assign back to the original DataFrame
        total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
            lambda x: classify_fms(x, quantiles)
        )
    total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
    total_data



    total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
    total_data.groupby('fms_class_subjectwise').count()

    total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                        'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                        'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])


    # Define constants
    n_hours = 28
    n_features = 14
    n_obj = n_features * n_hours
    lowcut = 0.03  # Lower cutoff frequency (adjust as needed)
    highcut = 0.3  # Higher cutoff frequency (adjust as needed)
    fs_original = 1  # Original sampling frequency (Hz)
    fs_resampled = 2  # Resampled sampling frequency (Hz)

    # Define EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Min-max normalization function
    def min_max_normalization(signal):
        scaler = MinMaxScaler()
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        return scaled_signal, scaler

    # Function to sort data by time
    def sort_by_time(data):
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data = data.sort_values(by='Time')
        return data

    # Function to apply a bandpass filter
    def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    # Preprocess data function
    def preprocess_data(data, sequence_length=28, columns=['Left_Eye_Openness', 'Right_Eye_Openness',
           'LeftPupilDiameter', 'RightPupilDiameter', 'LeftPupilPosInSensorX',
           'LeftPupilPosInSensorY', 'RightPupilPosInSensorX',
           'RightPupilPosInSensorY', 'NrmSRLeftEyeGazeDirX',
           'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
           'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ']):
        
        individuals = np.unique(data['individual'])
        simulations = np.unique(data['simulation'])

        X_train = []
        train_label = []

        for participant in individuals:
            for simulation in simulations:
                df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
                df.index = np.arange(len(df))
                n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
                features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}

                for col in columns:
                    if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                        features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                        features_resampled[col], scaler = min_max_normalization(features_resampled[col])

                fms_list = df['fms_class_subjectwise']
                fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])

                if df.shape[0] > sequence_length:
                    x_arr, y = segment_data(features_resampled, fms_doubled_list, sequence_length)
                    X_train.append(x_arr)
                    train_label.append(y)

        X_train_whole = np.concatenate(X_train, axis=0)
        train_label_whole = np.concatenate(train_label, axis=0)

        return X_train_whole, train_label_whole, scaler

    # Function to segment data
    def segment_data(features, target, sequence_length):
        X, y = [], []
        data_length = len(next(iter(features.values())))  # Get length of the first feature

        for i in range(data_length - sequence_length):
            x_segment = [features[col][i:i+sequence_length] for col in features]
            X.append(np.array(x_segment).T)  # Transpose to get shape (sequence_length, n_features)
            y.append(target[i + sequence_length])
        
        X_arr = np.array(X)
        y = np.array(y)
        return X_arr, y
    

    # Assuming `total_data` is your dataframe
    data = preprocess_data(total_data)
    X, Y, scaler= data[0], data[1], data[2]




    # Split data into training, validation, and test sets
    train_X, temp_X, train_y, temp_y = train_test_split(X, Y, test_size=0.3, random_state=42)
    valid_X, test_X, valid_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
    
    return train_X, train_y, valid_X, valid_y, test_X, test_y, scaler

def getclassification_processed_with_generated_XY():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import numpy as np
    import logging
    from datetime import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import matplotlib.dates as mdates

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from scipy.signal import resample
    import seaborn as sns

    from scipy.signal import butter, filtfilt, resample
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import callbacks
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    from scipy.signal import butter, filtfilt, resample
    from tensorflow.keras.utils import to_categorical
    # Connect with Google Drive
    # %cd '/content/drive/MyDrive/forecast_data'

    base_path = os.getcwd()+'\\forecast_data'
    base_path +="\\"
    meta_file = base_path+ 'meta_data.csv'
    meta_data = pd.read_csv(meta_file)
    meta_data

    # Funciton to filter data from the meta data
    def get_filtered_meta_data(individual_num, simulation_name):
      filtered_data = meta_data
      if simulation_name:
          filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
      if individual_num:
          filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]

      return filtered_data

    # Sort function to sort the data according to time

    # For Umama's Data (Need to update her data to include milliseconds to work with other one):
    # format='%H-%M-%S'

    # For Sam's Data:
    # format='%H-%M-%S-%f'
    def sort_by_time(data):
      data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
      data = data.sort_values(by='Time')
      return data
    # The function to select data
    def get_data(base_path=base_path, simulation_name=None, individual_num=None,
                data_type='', sort=True):
        pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
        file_paths = pruned_meta_data.apply(
            lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                    x[data_type]),
            axis=1)
        data_list = []
        for file in file_paths:
            head_data = pd.read_csv(base_path + file)
            data_list.append(head_data)
        data = pd.concat(data_list, ignore_index=True)
        if sort:
            data = sort_by_time(data)
        return data


    train_individual_list = [1,2,3,4,7,8,9,11,12,13,14] # The list of individuals in the data file
    train_simulation_list = ['sea','roller','beach','room','walk'] # List of Simulation in data file


    # We want to select all the different data time.
    data_type = ['hr', 'eda', 'eye', 'head']
    total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                       individual_num=train_individual_list, data_type = 'eye')
    total_data.dropna(axis=1,inplace=True)
    total_data

    #we can define class based on quantile or mean
    train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]


    def classify_fms(value, quantiles):
        if value <= quantiles[0.25]:
            return 0
        elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
            return 1
        elif value> quantiles[0.75]:
             return 2


    # # Ensure 'fms_class_subjectwise' column is properly defined with categories
    # total_data['fms_class_subjectwise'] = pd.Categorical(total_data['fms_class_subjectwise'], categories=[0, 1, 2, 3])

    for individual in train_individual_list:
        sub_individual_data = total_data[total_data['individual'] == individual]
        # sub_individual_data=remove_outliers_zscore(sub_individual_data, 'fms')
        quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
        print('quantiles\n',quantiles)

        # Apply classification and assign back to the original DataFrame
        total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
            lambda x: classify_fms(x, quantiles)
        )
    total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
    total_data



    total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
    total_data.groupby('fms_class_subjectwise').count()

    total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                        'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                        'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])


    # Define constants
    n_hours = 28
    n_features = 14
    n_obj = n_features * n_hours
    lowcut = 0.03  # Lower cutoff frequency (adjust as needed)
    highcut = 0.3  # Higher cutoff frequency (adjust as needed)
    fs_original = 1  # Original sampling frequency (Hz)
    fs_resampled = 2  # Resampled sampling frequency (Hz)

    # Define EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Min-max normalization function
    def min_max_normalization(signal):
        scaler = MinMaxScaler()
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        return scaled_signal, scaler

    # Function to sort data by time
    def sort_by_time(data):
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data = data.sort_values(by='Time')
        return data

    # Function to apply a bandpass filter
    def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    # Preprocess data function
    def preprocess_data(data, sequence_length=28, columns=['Left_Eye_Openness', 'Right_Eye_Openness',
           'LeftPupilDiameter', 'RightPupilDiameter', 'LeftPupilPosInSensorX',
           'LeftPupilPosInSensorY', 'RightPupilPosInSensorX',
           'RightPupilPosInSensorY', 'NrmSRLeftEyeGazeDirX',
           'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
           'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ']):
        
        individuals = np.unique(data['individual'])
        simulations = np.unique(data['simulation'])

        X_train = []
        train_label = []

        for participant in individuals:
            for simulation in simulations:
                df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
                df.index = np.arange(len(df))
                n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
                features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}

                for col in columns:
                    if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                        features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                        features_resampled[col], scaler = min_max_normalization(features_resampled[col])

                fms_list = df['fms_class_subjectwise']
                fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])

                if df.shape[0] > sequence_length:
                    x_arr, y = segment_data(features_resampled, fms_doubled_list, sequence_length)
                    X_train.append(x_arr)
                    train_label.append(y)

        X_train_whole = np.concatenate(X_train, axis=0)
        train_label_whole = np.concatenate(train_label, axis=0)

        return X_train_whole, train_label_whole, scaler

    # Function to segment data
    def segment_data(features, target, sequence_length):
        X, y = [], []
        data_length = len(next(iter(features.values())))  # Get length of the first feature

        for i in range(data_length - sequence_length):
            x_segment = [features[col][i:i+sequence_length] for col in features]
            X.append(np.array(x_segment).T)  # Transpose to get shape (sequence_length, n_features)
            y.append(target[i + sequence_length])
        
        X_arr = np.array(X)
        y = np.array(y)
        return X_arr, y

    # Assuming `total_data` is your dataframe
    data = preprocess_data(total_data)
    X, Y, scaler= data[0], data[1], data[2]




    # Split data into training, validation, and test sets
    train_X, temp_X, train_y, temp_y = train_test_split(X, Y, test_size=0.3, random_state=42)
    valid_X, test_X, valid_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
    
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, Dense, Reshape, LeakyReLU, Concatenate, Embedding, Conv2DTranspose
    from keras import callbacks
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    import Utils as utils  # Ensure Utils module is available

    # Parameters
    latent_dim = 100
    n_classes = 3
    n_hours = 28
    n_features = 14

    # Define the generator model
    def define_generator(latent_dim, n_classes=n_classes):
        in_label = Input(shape=(1,))
        li = Embedding(n_classes, 50)(in_label)  # Adjust embedding size if necessary
        n_nodes = 28 * 14  # Adjust to match the final output shape
        li = Dense(n_nodes)(li)
        li = Reshape((28, 14, 1))(li)  # Adjust to match the intermediate shape of gen
        
        in_lat = Input(shape=(latent_dim,))
        n_nodes = 28 * 14 * 64  # Adjust to match the intermediate dimensions
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((28, 14, 64))(gen)  # Adjust to have an intermediate depth

        merge = Concatenate()([gen, li])
        
        # Adjust Conv2DTranspose layers to achieve the desired shape
        gen = Conv2DTranspose(64, (4, 4), strides=(1, 1), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        
        gen = Conv2DTranspose(32, (4, 4), strides=(1, 1), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        
        out_layer = Conv2DTranspose(1, (4, 4), strides=(1, 1), activation='tanh', padding='same')(gen)  # Final Conv2DTranspose layer to get the output shape
        out_layer = Reshape((28, 14))(out_layer)  # Ensure the correct final shape
        
        model = Model([in_lat, in_label], out_layer)
        return model

    # Load the generator model
    generator = define_generator(latent_dim)

    # Function to generate synthetic samples
    def generate_samples(generator, latent_dim, n_samples_per_class, class_label):
        z_input = np.random.randn(n_samples_per_class, latent_dim)
        labels_input = np.full((n_samples_per_class, 1), class_label)
        images = generator.predict([z_input, labels_input])
        return images, labels_input

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(train_y, return_counts=True)
    print(unique_classes, class_counts)
    max_class_count = max(class_counts)

    # Generate enough samples to balance the dataset
    generated_X_list = []
    generated_y_list = []

    for class_label, count in zip(unique_classes, class_counts):
        n_samples_needed = max_class_count - count
        if n_samples_needed > 0:
            print(f"Class {class_label}: {count} samples, need {n_samples_needed} more")
            generated_X, generated_y = generate_samples(generator, latent_dim, n_samples_needed, class_label)
            print(generated_X.shape, generated_y.shape)
            generated_X_list.append(generated_X)
            generated_y_list.append(generated_y)

    if generated_X_list:
        generated_X = np.vstack(generated_X_list)
        generated_y = np.vstack(generated_y_list)
    else:
        generated_X = np.empty((0, n_hours, n_features))
        generated_y = np.empty((0,))

    # Augment training data with generated data
    train_X_augmented = np.concatenate((train_X, generated_X), axis=0)
    train_y_augmented = np.concatenate((train_y, generated_y.reshape(generated_y.shape[0])), axis=0)

    return train_X_augmented, train_y_augmented, valid_X, valid_y, test_X, test_y, scaler

def getclassification_processed_crossfolded_XY_for_regression():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import numpy as np
    import logging
    from datetime import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import matplotlib.dates as mdates

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from scipy.signal import resample
    import seaborn as sns

    from scipy.signal import butter, filtfilt, resample
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import callbacks
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    from scipy.signal import butter, filtfilt, resample
    from tensorflow.keras.utils import to_categorical
    # Connect with Google Drive
    # %cd '/content/drive/MyDrive/forecast_data'

    base_path = os.getcwd()+'\\forecast_data'
    base_path +="\\"
    meta_file = base_path+ 'meta_data.csv'
    meta_data = pd.read_csv(meta_file)
    meta_data
    all_column_scalers={}
    # Funciton to filter data from the meta data
    def get_filtered_meta_data(individual_num, simulation_name):
      filtered_data = meta_data
      if simulation_name:
          filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
      if individual_num:
          filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]

      return filtered_data

    
    def sort_by_time(data):
      data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
      data = data.sort_values(by='Time')
      return data
    # The function to select data
    def get_data(base_path=base_path, simulation_name=None, individual_num=None,
                data_type='', sort=True):
        pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
        file_paths = pruned_meta_data.apply(
            lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                    x[data_type]),
            axis=1)
        data_list = []
        for file in file_paths:
            head_data = pd.read_csv(base_path + file)
            data_list.append(head_data)
        data = pd.concat(data_list, ignore_index=True)
        if sort:
            data = sort_by_time(data)
        return data


    train_individual_list = [1,2,3,4,7,8,9,11,12,13,14] # The list of individuals in the data file
    train_simulation_list = ['sea','roller','beach','room','walk'] # List of Simulation in data file


    # We want to select all the different data time.
    data_type = ['hr', 'eda', 'eye', 'head']
    total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                       individual_num=train_individual_list, data_type = 'eye')
    total_data.dropna(axis=1,inplace=True)
    total_data

    #we can define class based on quantile or mean
    train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]


    def classify_fms(value, quantiles):
        if value <= quantiles[0.25]:
            return 0
        elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
            return 1
        elif value> quantiles[0.75]:
             return 2


    # # Ensure 'fms_class_subjectwise' column is properly defined with categories
    # total_data['fms_class_subjectwise'] = pd.Categorical(total_data['fms_class_subjectwise'], categories=[0, 1, 2, 3])

    for individual in train_individual_list:
        sub_individual_data = total_data[total_data['individual'] == individual]
        # sub_individual_data=remove_outliers_zscore(sub_individual_data, 'fms')
        quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
        print('quantiles\n',quantiles)

        # Apply classification and assign back to the original DataFrame
        total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
            lambda x: classify_fms(x, quantiles)
        )
    total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
    total_data



    total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
    total_data.groupby('fms_class_subjectwise').count()

    total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                        'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                        'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])


    # Define constants
    n_hours = 28
    n_features = 14
    n_obj = n_features * n_hours
    lowcut = 0.03  # Lower cutoff frequency (adjust as needed)
    highcut = 0.3  # Higher cutoff frequency (adjust as needed)
    fs_original = 1  # Original sampling frequency (Hz)
    fs_resampled = 2  # Resampled sampling frequency (Hz)

    # Define EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Min-max normalization function
    def min_max_normalization(signal):
        scaler = MinMaxScaler()
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        return scaled_signal, scaler

    # Function to sort data by time
    def sort_by_time(data):
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data = data.sort_values(by='Time')
        return data

    # Function to apply a bandpass filter
    def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    # Function to segment data
    def segment_data(features, target, sequence_length):
        X, y = [], []
        data_length = len(next(iter(features.values())))  # Get length of the first feature

        for i in range(int((data_length - sequence_length)/5)):
            i=i*5
            x_segment = [features[col][i:i+sequence_length] for col in features]
            X.append(np.array(x_segment).T)  # Transpose to get shape (sequence_length, n_features)
            y.append(target[i + sequence_length])
        
        X_arr = np.array(X)
        y = np.array(y)
        return X_arr, y

    # Preprocess data function
    def preprocess_data(data, sequence_length=28, columns=['Left_Eye_Openness', 'Right_Eye_Openness',
           'LeftPupilDiameter', 'RightPupilDiameter', 'LeftPupilPosInSensorX',
           'LeftPupilPosInSensorY', 'RightPupilPosInSensorX',
           'RightPupilPosInSensorY', 'NrmSRLeftEyeGazeDirX',
           'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
           'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ']):
        
        individuals = np.unique(data['individual'])
        simulations = np.unique(data['simulation'])

        X_train = []
        train_label = []

        for participant in individuals:
            for simulation in simulations:
                df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
                df.index = np.arange(len(df))
                n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
                features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}

                for col in columns:
                    if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                        features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                        features_resampled[col], scaler = min_max_normalization(features_resampled[col])
                        all_column_scalers[col] = scaler

                fms_list = df['fms']
                fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])

                if df.shape[0] > sequence_length:
                    x_arr, y = segment_data(features_resampled, fms_doubled_list, sequence_length)
                    X_train.append(x_arr)
                    train_label.append(y)

        X_train_whole = np.concatenate(X_train, axis=0)
        train_label_whole = np.concatenate(train_label, axis=0)

        return X_train_whole, train_label_whole, all_column_scalers

    # Assuming `total_data` is your dataframe
    data = preprocess_data(total_data)
    X_train_eye, Y_train_eye, x_scaler_for_trainX_eye = data[0], data[1], data[2]
    
    from sklearn.model_selection import StratifiedKFold, KFold
    # skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    kf = KFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    
    for i, (train_indices, test_indices) in enumerate(kf.split(X_train_eye, Y_train_eye)):
        
        # Split data into training, validation, and test sets
        # train_X, temp_X, train_y, temp_y = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        temp_X= X_train_eye[test_indices]
        temp_y=Y_train_eye[test_indices]
        
        x_val, x_test, y_val, y_test = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
        
        
        # valdata_len=int(len(x_test)*0.6)
        
        # x_val=x_test[:valdata_len]
        # y_val=y_test[:valdata_len]
        
        # x_test=x_test[valdata_len:]
        # y_test=y_test[valdata_len:]
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye

def getHR_processed_crossfolded_XY_classification():
    import pandas as pd
    import os
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, BatchNormalization, Reshape, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import numpy as np
    import logging
    from datetime import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import matplotlib.dates as mdates
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from scipy.signal import resample
    import seaborn as sns
    
    from scipy.signal import butter, filtfilt, resample
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import callbacks
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    from scipy.signal import butter, filtfilt, resample
    from tensorflow.keras.utils import to_categorical
    
    base_path = os.getcwd()+'\\forecast_data'
    base_path +="\\"
    meta_file = base_path+ 'meta_data.csv'
    meta_data = pd.read_csv(meta_file)
    meta_data
    
    def get_filtered_meta_data(individual_num, simulation_name):
      filtered_data = meta_data
      if simulation_name:
          filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
      if individual_num:
          filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]
    
      return filtered_data
    
    
    def sort_by_time(data):
      data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
      data = data.sort_values(by='Time')
      return data
    
    def get_data(base_path=base_path, simulation_name=None, individual_num=None,
                data_type='', sort=True):
        pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
        file_paths = pruned_meta_data.apply(
            lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                    x[data_type]),
            axis=1)
        data_list = []
        for file in file_paths:
            head_data = pd.read_csv(base_path + file)
            data_list.append(head_data)
        data = pd.concat(data_list, ignore_index=True)
        if sort:
            data = sort_by_time(data)
        return data
    
    
    train_individual_list = [1,2,3,4,7,8,9,11,12,13,14]
    train_simulation_list = ['sea','roller','beach','room','walk']
    
    data_type = ['hr', 'eda', 'eye', 'head']
    total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                       individual_num=train_individual_list, data_type = 'hr')
    total_data.dropna(axis=1,inplace=True)
    total_data
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    unique_label=total_data['simulation'].unique()
    le.fit(total_data['simulation'])
    total_data['simulation_encoded'] = le.transform(total_data['simulation'])
    
    train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]
    
    
    def classify_fms(value, quantiles):
        if value <= quantiles[0.25]:
            return 0
        elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
            return 1
        elif value> quantiles[0.75]:
             return 2
    
    for individual in train_individual_list:
        sub_individual_data = total_data[total_data['individual'] == individual]
        quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
        print('quantiles\n',quantiles)
    
        total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
            lambda x: classify_fms(x, quantiles)
        )
    total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
    total_data
    
    total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
    total_data.groupby('fms_class_subjectwise').count()
    
    # total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
    #                                     'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
    #                                     'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])
    
    n_hours = 28
    n_features = 1
    n_obj = n_features * n_hours
    lowcut = 0.03
    highcut = 0.3
    fs_original = 1
    fs_resampled = 2
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    def min_max_normalization(signal):
        scaler = MinMaxScaler()
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        return scaled_signal, scaler
    
    def sort_by_time(data):
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data = data.sort_values(by='Time')
        return data
    
    def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def segment_data(features, target, sequence_length, simul_enc):
        X, y, z= [], [], []
        data_length = len(next(iter(features.values())))
    
        for i in range(int((data_length - sequence_length)/1)):
            # i=i*5
            x_segment = [features[col][i:i+sequence_length] for col in features]
            X.append(np.array(x_segment).T)
            y.append(target[i + sequence_length])
            z.append(simul_enc[i + sequence_length])
        
        X_arr = np.array(X)
        y = np.array(y)
        z = np.array(z)
        return X_arr, y, z
    
    def preprocess_data(data, sequence_length=28, columns=['HR']):
        
        individuals = np.unique(data['individual'])
        simulations = np.unique(data['simulation'])
    
        X_train = []
        train_label = []
        simul_enc= []
    
        for participant in individuals:
            for simulation in simulations:
                df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
                df.index = np.arange(len(df))
                n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
                features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}
    
                for col in columns:
                    if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                        features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                        features_resampled[col], scaler = min_max_normalization(features_resampled[col])
    
                fms_list = df['fms_class_subjectwise']
                fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])
                
                simul_enc_double = np.array([x for x in df['simulation_encoded'] for _ in range(int(fs_resampled/fs_original))])
    
                if df.shape[0] > sequence_length:
                    x_arr, y, z = segment_data(features_resampled, fms_doubled_list, sequence_length, simul_enc=simul_enc_double )
                    X_train.append(x_arr)
                    train_label.append(y)
                    simul_enc.append(z)
    
        X_train_whole = np.concatenate(X_train, axis=0)
        train_label_whole = np.concatenate(train_label, axis=0)
        simul_enc_whole = np.concatenate(simul_enc, axis=0)
    
        return X_train_whole, train_label_whole, simul_enc_whole, scaler
    
    
    data = preprocess_data(total_data)
    X_train_eye, Y_train_eye, simulation_encoded, x_scaler_for_trainX_eye = data[0], data[1], data[2], data[3]
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data, Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    
    for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, simulation_encoded)):
    
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        temp_X= X_train_eye[test_indices]
        temp_y=Y_train_eye[test_indices]
        
        x_val, x_test, y_val, y_test = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
    
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye

def getHR_processed_crossfolded_XY_for_regression():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import numpy as np
    import logging
    from datetime import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import matplotlib.dates as mdates

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from scipy.signal import resample
    import seaborn as sns

    from scipy.signal import butter, filtfilt, resample
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import callbacks
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    from scipy.signal import butter, filtfilt, resample
    from tensorflow.keras.utils import to_categorical
    # Connect with Google Drive
    # %cd '/content/drive/MyDrive/forecast_data'

    base_path = os.getcwd()+'\\forecast_data'
    base_path +="\\"
    meta_file = base_path+ 'meta_data.csv'
    meta_data = pd.read_csv(meta_file)
    meta_data

    # Funciton to filter data from the meta data
    def get_filtered_meta_data(individual_num, simulation_name):
      filtered_data = meta_data
      if simulation_name:
          filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
      if individual_num:
          filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]

      return filtered_data

    
    def sort_by_time(data):
      data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
      data = data.sort_values(by='Time')
      return data
    # The function to select data
    def get_data(base_path=base_path, simulation_name=None, individual_num=None,
                data_type='', sort=True):
        pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
        file_paths = pruned_meta_data.apply(
            lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                    x[data_type]),
            axis=1)
        data_list = []
        for file in file_paths:
            head_data = pd.read_csv(base_path + file)
            data_list.append(head_data)
        data = pd.concat(data_list, ignore_index=True)
        if sort:
            data = sort_by_time(data)
        return data


    train_individual_list = [1,2,3,4,7,8,9,11,12,13,14] # The list of individuals in the data file
    train_simulation_list = ['sea','roller','beach','room','walk'] # List of Simulation in data file


    # We want to select all the different data time.
    data_type = ['hr', 'eda', 'eye', 'head']
    total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                       individual_num=train_individual_list, data_type = 'hr')
    total_data.dropna(axis=1,inplace=True)
    total_data

    #we can define class based on quantile or mean
    train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]


    def classify_fms(value, quantiles):
        if value <= quantiles[0.25]:
            return 0
        elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
            return 1
        elif value> quantiles[0.75]:
             return 2


    # # Ensure 'fms_class_subjectwise' column is properly defined with categories
    # total_data['fms_class_subjectwise'] = pd.Categorical(total_data['fms_class_subjectwise'], categories=[0, 1, 2, 3])

    for individual in train_individual_list:
        sub_individual_data = total_data[total_data['individual'] == individual]
        # sub_individual_data=remove_outliers_zscore(sub_individual_data, 'fms')
        quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
        print('quantiles\n',quantiles)
        
        # Apply classification and assign back to the original DataFrame
        total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
            lambda x: classify_fms(x, quantiles)
        )
    total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
    total_data



    total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
    total_data.groupby('fms_class_subjectwise').count()

    # total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
    #                                     'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
    #                                     'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])


    # Define constants
    n_hours = 28
    n_features = 1
    n_obj = n_features * n_hours
    lowcut = 0.03  # Lower cutoff frequency (adjust as needed)
    highcut = 0.3  # Higher cutoff frequency (adjust as needed)
    fs_original = 1  # Original sampling frequency (Hz)
    fs_resampled = 2  # Resampled sampling frequency (Hz)

    # Define EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Min-max normalization function
    def min_max_normalization(signal):
        scaler = MinMaxScaler()
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        return scaled_signal, scaler

    # Function to sort data by time
    def sort_by_time(data):
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data = data.sort_values(by='Time')
        return data

    # Function to apply a bandpass filter
    def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    # Function to segment data
    def segment_data(features, target, sequence_length):
        X, y = [], []
        data_length = len(next(iter(features.values())))  # Get length of the first feature

        for i in range(int((data_length - sequence_length)/5)):
            i=i*5
            x_segment = [features[col][i:i+sequence_length] for col in features]
            X.append(np.array(x_segment).T)  # Transpose to get shape (sequence_length, n_features)
            y.append(target[i + sequence_length])
        
        X_arr = np.array(X)
        y = np.array(y)
        return X_arr, y

    # Preprocess data function
    def preprocess_data(data, sequence_length=28, columns=['HR']):
        
        individuals = np.unique(data['individual'])
        simulations = np.unique(data['simulation'])

        X_train = []
        train_label = []

        for participant in individuals:
            for simulation in simulations:
                df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
                df.index = np.arange(len(df))
                n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
                features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}

                for col in columns:
                    if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                        features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                        features_resampled[col], scaler = min_max_normalization(features_resampled[col])

                fms_list = df['fms']
                fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])

                if df.shape[0] > sequence_length:
                    x_arr, y = segment_data(features_resampled, fms_doubled_list, sequence_length)
                    X_train.append(x_arr)
                    train_label.append(y)

        X_train_whole = np.concatenate(X_train, axis=0)
        train_label_whole = np.concatenate(train_label, axis=0)

        return X_train_whole, train_label_whole, scaler

    # Assuming `total_data` is your dataframe
    data = preprocess_data(total_data)
    X_train_eye, Y_train_eye, x_scaler_for_trainX_eye = data[0], data[1], data[2]
    
    from sklearn.model_selection import StratifiedKFold, KFold
    # skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    kf = KFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    
    for i, (train_indices, test_indices) in enumerate(kf.split(X_train_eye, Y_train_eye)):
        
        # Split data into training, validation, and test sets
        # train_X, temp_X, train_y, temp_y = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        temp_X= X_train_eye[test_indices]
        temp_y=Y_train_eye[test_indices]
        
        x_val, x_test, y_val, y_test = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye

def getclassification_processed_crossfolded_XY():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import numpy as np
    import logging
    from datetime import datetime
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import datetime
    import matplotlib.dates as mdates

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from scipy.signal import resample
    import seaborn as sns

    from scipy.signal import butter, filtfilt, resample
    from sklearn.preprocessing import MinMaxScaler
    from tqdm import tqdm

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras import callbacks
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    from scipy.signal import butter, filtfilt, resample
    from tensorflow.keras.utils import to_categorical
    # Connect with Google Drive
    # %cd '/content/drive/MyDrive/forecast_data'

    base_path = os.getcwd()+'\\forecast_data'
    base_path +="\\"
    meta_file = base_path+ 'meta_data.csv'
    meta_data = pd.read_csv(meta_file)
    meta_data

    # Funciton to filter data from the meta data
    def get_filtered_meta_data(individual_num, simulation_name):
      filtered_data = meta_data
      if simulation_name:
          filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
      if individual_num:
          filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]

      return filtered_data

    
    def sort_by_time(data):
      data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
      data = data.sort_values(by='Time')
      return data
    # The function to select data
    def get_data(base_path=base_path, simulation_name=None, individual_num=None,
                data_type='', sort=True):
        pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
        file_paths = pruned_meta_data.apply(
            lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                    x[data_type]),
            axis=1)
        data_list = []
        for file in file_paths:
            head_data = pd.read_csv(base_path + file)
            data_list.append(head_data)
        data = pd.concat(data_list, ignore_index=True)
        if sort:
            data = sort_by_time(data)
        return data


    train_individual_list = [1,2,3,4,7,8,9,11,12,13,14] # The list of individuals in the data file
    train_simulation_list = ['sea','roller','beach','room','walk'] # List of Simulation in data file


    # We want to select all the different data time.
    data_type = ['hr', 'eda', 'eye', 'head']
    total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                       individual_num=train_individual_list, data_type = 'eye')
    total_data.dropna(axis=1,inplace=True)
    total_data
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    unique_label=total_data['simulation'].unique()
    le.fit(total_data['simulation'])
    total_data['simulation_encoded'] = le.transform(total_data['simulation'])

    #we can define class based on quantile or mean
    train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
    def remove_outliers_zscore(df, column, threshold=3):
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]


    def classify_fms(value, quantiles):
        if value <= quantiles[0.25]:
            return 0
        elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
            return 1
        elif value> quantiles[0.75]:
             return 2


    # # Ensure 'fms_class_subjectwise' column is properly defined with categories
    # total_data['fms_class_subjectwise'] = pd.Categorical(total_data['fms_class_subjectwise'], categories=[0, 1, 2, 3])

    for individual in train_individual_list:
        sub_individual_data = total_data[total_data['individual'] == individual]
        # sub_individual_data=remove_outliers_zscore(sub_individual_data, 'fms')
        quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
        print('quantiles\n',quantiles)

        # Apply classification and assign back to the original DataFrame
        total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
            lambda x: classify_fms(x, quantiles)
        )
    total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
    total_data

    total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
    total_data.groupby('fms_class_subjectwise').count()

    total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                        'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                        'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])


    # Define constants
    n_hours = 28
    n_features = 14
    n_obj = n_features * n_hours
    lowcut = 0.03  # Lower cutoff frequency (adjust as needed)
    highcut = 0.3  # Higher cutoff frequency (adjust as needed)
    fs_original = 1  # Original sampling frequency (Hz)
    fs_resampled = 2  # Resampled sampling frequency (Hz)

    # Define EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Min-max normalization function
    def min_max_normalization(signal):
        scaler = MinMaxScaler()
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        return scaled_signal, scaler

    # Function to sort data by time
    def sort_by_time(data):
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
        data = data.sort_values(by='Time')
        return data

    # Function to apply a bandpass filter
    def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    # Function to segment data
    def segment_data(features, target, sequence_length, simul_enc):
        X, y, z= [], [], []
        data_length = len(next(iter(features.values())))  # Get length of the first feature

        for i in range(int((data_length - sequence_length)/5)):
            i=i*5
            x_segment = [features[col][i:i+sequence_length] for col in features]
            X.append(np.array(x_segment).T)  # Transpose to get shape (sequence_length, n_features)
            y.append(target[i + sequence_length])
            z.append(simul_enc[i + sequence_length])
        
        X_arr = np.array(X)
        y = np.array(y)
        z = np.array(z)
        return X_arr, y, z
    

    # Preprocess data function
    def preprocess_data(data, sequence_length=28, columns=['Left_Eye_Openness', 'Right_Eye_Openness',
           'LeftPupilDiameter', 'RightPupilDiameter', 'LeftPupilPosInSensorX',
           'LeftPupilPosInSensorY', 'RightPupilPosInSensorX',
           'RightPupilPosInSensorY', 'NrmSRLeftEyeGazeDirX',
           'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
           'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ']):
        
        individuals = np.unique(data['individual'])
        simulations = np.unique(data['simulation'])

        X_train = []
        train_label = []
        simul_enc= []

        for participant in individuals:
            for simulation in simulations:
                df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
                df.index = np.arange(len(df))
                n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
                features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}

                for col in columns:
                    if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                        features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                        features_resampled[col], scaler = min_max_normalization(features_resampled[col])

                fms_list = df['fms_class_subjectwise']
                fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])
                
                simul_enc_double = np.array([x for x in df['simulation_encoded'] for _ in range(int(fs_resampled/fs_original))])

                if df.shape[0] > sequence_length:
                    x_arr, y, z = segment_data(features_resampled, fms_doubled_list, sequence_length, simul_enc=simul_enc_double )
                    X_train.append(x_arr)
                    train_label.append(y)
                    simul_enc.append(z)

        X_train_whole = np.concatenate(X_train, axis=0)
        train_label_whole = np.concatenate(train_label, axis=0)
        simul_enc_whole = np.concatenate(simul_enc, axis=0)

        return X_train_whole, train_label_whole, simul_enc_whole, scaler

    # Assuming `total_data` is your dataframe
    data = preprocess_data(total_data)
    X_train_eye, Y_train_eye, simulation_encoded, x_scaler_for_trainX_eye = data[0], data[1], data[2], data[3]
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data, Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    
    for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, simulation_encoded)):
        
        # Split data into training, validation, and test sets
        # train_X, temp_X, train_y, temp_y = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        temp_X= X_train_eye[test_indices]
        temp_y=Y_train_eye[test_indices]
        
        x_val, x_test, y_val, y_test = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye


    


def getclassification_crossfolded_XY():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','individual','simulation'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            
            #######This is for balancing classes
            '''
            if (df_eye_csv['CSS'][0]=='High'):
                frames = [training_eye_df, dff]
                training_eye_df=pd.concat(frames)
            else:
                if(df_eye_csv['CSS'][0]=='Low'):
                    count1+=1
                if(count1>=82):
                    frames = [training_eye_df, dff]
                    training_eye_df=pd.concat(frames)
            '''
        

    training_eye_df=remove_outliers_zscore(training_eye_df,'fms',3)
      
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye= "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, Y_train_eye)):
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        x_test= X_train_eye[test_indices]
        y_test=Y_train_eye[test_indices]
        
        valdata_len=int(len(x_test)*0.6)
        
        x_val=x_test[:valdata_len]
        y_val=y_test[:valdata_len]
        
        x_test=x_test[valdata_len:]
        y_test=y_test[valdata_len:]
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye, le

def getclassification_crossfolded_XY_2class():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived_2class)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived_2class)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            
            #######This is for balancing classes
            '''
            if (df_eye_csv['CSS'][0]=='High'):
                frames = [training_eye_df, dff]
                training_eye_df=pd.concat(frames)
            else:
                if(df_eye_csv['CSS'][0]=='Low'):
                    count1+=1
                if(count1>=82):
                    frames = [training_eye_df, dff]
                    training_eye_df=pd.concat(frames)
            '''
        
            
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye= "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, Y_train_eye)):
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        x_test= X_train_eye[test_indices]
        y_test=Y_train_eye[test_indices]
        
        valdata_len=int(len(x_test)*0.6)
        
        x_val=x_test[:valdata_len]
        y_val=y_test[:valdata_len]
        
        x_test=x_test[valdata_len:]
        y_test=y_test[valdata_len:]
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye, le

def get_hp_data(x_scaler_for_trainX_eye):
    cwd= os.getcwd()
    hp_eye_df = pd.read_csv(cwd+'\\EyeTrackingData2024-06-11 16-59.csv')
    try:
        hp_eye_df = hp_eye_df.drop(columns=['Timestamp', 'FrameNumber', 'LeftEye_Gaze_Confidence',
        'LeftEye_PupilPosition_Confidence', 'LeftEye_Openness_Confidence',
        'LeftEye_PupilDilation_Confidence', 'RightEye_Gaze_Confidence',
        'RightEye_PupilPosition_Confidence', 'RightEye_Openness_Confidence',
        'RightEye_PupilDilation_Confidence', 'CombinedGaze_X', 'CombinedGaze_Y',
        'CombinedGaze_Z', 'CombinedGaze_Confidence'])
        hp_eye_df=hp_eye_df.rename(columns={'LeftEye_Openness_Openness':'Left_Eye_Openness',
               'RightEye_Openness_Openness':'Right_Eye_Openness', 'LeftEye_PupilDilation_PupilDilation':'LeftPupilDiameter',
               'RightEye_PupilDilation_PupilDilation':'RightPupilDiameter', 'LeftEye_PupilPosition_X':'LeftPupilPosInSensorX',
               'LeftEye_PupilPosition_Y':'LeftPupilPosInSensorY', 'RightEye_PupilPosition_X':'RightPupilPosInSensorX',
               'RightEye_PupilPosition_Y':'RightPupilPosInSensorY', 'LeftEye_Gaze_X':'NrmSRLeftEyeGazeDirX', 'LeftEye_Gaze_Y':'NrmSRLeftEyeGazeDirY',
               'LeftEye_Gaze_Z':'NrmSRLeftEyeGazeDirZ', 'RightEye_Gaze_X':'NrmSRRightEyeGazeDirX', 'RightEye_Gaze_Y':'NrmSRRightEyeGazeDirY',
               'RightEye_Gaze_Z':'NrmSRRightEyeGazeDirZ'})
    except:
        pass

    hp_eye_df=x_scaler_for_trainX_eye.transform(hp_eye_df)

    #X_test_eye, Y_test_eye = np.array([]),np.array([]) 
    for i in tqdm(range(int((hp_eye_df.shape[0])/28))):
        if (i==0):
            X_test_eye_hp = np.expand_dims(hp_eye_df[i*28:(i+1)*28,:],0)
        else:
            X_test_eye_hp=np.concatenate((X_test_eye_hp,np.expand_dims(hp_eye_df[i*28:(i+1)*28,:],0)),0)
            
    print("X_test_eye_hp = "+str(X_test_eye_hp.shape))
    return X_test_eye_hp

def save_data(X_train_eye,Y_train_eye,X_val_eye,Y_val_eye,X_test_eye,Y_test_eye):
    np.save('X_train_eye.npy', X_train_eye)
    np.save('Y_train_eye.npy', Y_train_eye)
    np.save('X_val_eye.npy', X_val_eye)
    np.save('Y_val_eye.npy', Y_val_eye)
    np.save('X_test_eye.npy',X_test_eye )
    np.save('Y_test_eye.npy', Y_test_eye)
    
def performance_metrics(Y_test_eye, y_pred_eye):
    accuracy = accuracy_score(Y_test_eye, y_pred_eye)
    precision = precision_score(Y_test_eye, y_pred_eye, average='weighted')
    recall = recall_score(Y_test_eye, y_pred_eye, average='weighted')
    f1 = f1_score(Y_test_eye, y_pred_eye, average='weighted')
    # print(f"Accuracy: {accuracy:0.2f}")
    # print(f"Precision: {precision:0.2f}")
    # print(f"Recall: {recall:0.2f}")
    # print(f"F1 Score: {f1:0.2f}")
    return accuracy, precision, recall, f1

def display_cm(Y_test_eye, y_pred_eye):
    cm = confusion_matrix(Y_test_eye, y_pred_eye)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return cm

def Error_metrics(Y_test_eye,y_pred_eye):
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
    from scipy.stats import pearsonr
    results = {}
    results['mean_squared_error'] = round(mean_squared_error(Y_test_eye, y_pred_eye),3)
    results['root_mean_squared_error'] = round(math.sqrt(mean_squared_error(Y_test_eye, y_pred_eye)),3)
    results['mean_absolute_error'] = round(mean_absolute_error(Y_test_eye, y_pred_eye),3)
    results['r2_score'] = round(r2_score(Y_test_eye, y_pred_eye),3)
    # print(results)
    return results



if __name__=='__main__':
    getregdata()
