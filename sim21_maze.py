# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 02:54:08 2024

@author: Jahirul
"""
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
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import random

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[abs(z_scores) < threshold]

def get_maze_sim21_cross_regdata():
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
    min_datapoint=[]
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            
            df_eye_csv = pd.read_csv(eye_csv)
            #df_eye_csv=remove_outliers_zscore(df_eye_csv,'fms',0)
            #min_datapoint.append(len(df_eye_csv))
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
    print('simulation21 data')
    for (i,f) in enumerate(training_eye_df['fms'].unique()):
        print(f" {f} has datapoints: {len(training_eye_df[training_eye_df['fms']==f])}")
    
    maze_data_df = pd.read_csv(cwd+'\\final_data.csv')
    selected_columns=['fms','Left_Openness','Right_Openness', 'Left_Diameter',
           'Right_Diameter', 'Left_PupilSensor_X',
           'Left_PupilSensor_Y', 'Right_PupilSensor_X',
           'Right_PupilSensor_Y', 'Left_GazeDir_X', 'Left_GazeDir_Y',
           'Left_GazeDir_Z', 'Right_GazeDir_X', 'Right_GazeDir_Y',
           'Right_GazeDir_Z']
    maze_data_df = maze_data_df[selected_columns]

    new_column_names=['fms', 'Left_Eye_Openness', 'Right_Eye_Openness', 'LeftPupilDiameter',
           'RightPupilDiameter', 'LeftPupilPosInSensorX', 'LeftPupilPosInSensorY',
           'RightPupilPosInSensorX', 'RightPupilPosInSensorY',
           'NrmSRLeftEyeGazeDirX', 'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ',
           'NrmSRRightEyeGazeDirX', 'NrmSRRightEyeGazeDirY',
           'NrmSRRightEyeGazeDirZ']

    maze_data_df.columns = new_column_names
    print('maze data')
    for (i,f) in enumerate(maze_data_df['fms'].unique()):
        print(f" {f} has datapoints: {len(maze_data_df[maze_data_df['fms']==f])}")
        
    merged_df = pd.concat([training_eye_df, maze_data_df], axis=0)
    print('merged data')
    for (i,f) in enumerate(merged_df['fms'].unique()):
        print(f" {f} has datapoints: {len(merged_df[merged_df['fms']==f])}")
    
    training_eye_df = merged_df
    
    trainY_eye=training_eye_df[['fms']]
    trainX_eye=training_eye_df.drop(columns=['fms'])
    
    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainY_eye=trainY_eye.to_numpy()
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
    
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold, KFold
    
    kf = KFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    
    for i, (train_indices, test_indices) in enumerate(kf.split(X_train_eye, Y_train_eye)):
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

def get_maze_sim21_cross_class_data():
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
    min_datapoint=[]
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            
            df_eye_csv = pd.read_csv(eye_csv)
            #df_eye_csv=remove_outliers_zscore(df_eye_csv,'fms',0)
            #min_datapoint.append(len(df_eye_csv))
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
    print('simulation21 data')
    for (i,f) in enumerate(training_eye_df['fms'].unique()):
        print(f" {f} has datapoints: {len(training_eye_df[training_eye_df['fms']==f])}")
    
    maze_data_df = pd.read_csv(cwd+'\\final_data.csv')
    selected_columns=['fms','Left_Openness','Right_Openness', 'Left_Diameter',
           'Right_Diameter', 'Left_PupilSensor_X',
           'Left_PupilSensor_Y', 'Right_PupilSensor_X',
           'Right_PupilSensor_Y', 'Left_GazeDir_X', 'Left_GazeDir_Y',
           'Left_GazeDir_Z', 'Right_GazeDir_X', 'Right_GazeDir_Y',
           'Right_GazeDir_Z']
    maze_data_df = maze_data_df[selected_columns]

    new_column_names=['fms', 'Left_Eye_Openness', 'Right_Eye_Openness', 'LeftPupilDiameter',
           'RightPupilDiameter', 'LeftPupilPosInSensorX', 'LeftPupilPosInSensorY',
           'RightPupilPosInSensorX', 'RightPupilPosInSensorY',
           'NrmSRLeftEyeGazeDirX', 'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ',
           'NrmSRRightEyeGazeDirX', 'NrmSRRightEyeGazeDirY',
           'NrmSRRightEyeGazeDirZ']

    maze_data_df.columns = new_column_names
    print('maze data')
    for (i,f) in enumerate(maze_data_df['fms'].unique()):
        print(f" {f} has datapoints: {len(maze_data_df[maze_data_df['fms']==f])}")
    
    merged_df = pd.concat([training_eye_df, maze_data_df], axis=0)
    print('merged data')
    for (i,f) in enumerate(merged_df['fms'].unique()):
        print(f" {f} has datapoints: {len(merged_df[merged_df['fms']==f])}")
        
    # Plotting the distribution of the random integers list
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df['fms'], bins=11, edgecolor='black', alpha=0.7)
    plt.title('merged data fms distribution', fontsize=16)
    plt.xlabel('FMS', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()
    
    training_eye_df = merged_df
    quantiles = merged_df['fms'].quantile([0.25,0.5,0.75])
    low, mid, high =0, 1, 2
    # Apply conditions based on the specified ranges
    training_eye_df.loc[(training_eye_df['fms'] < quantiles[0.25]), 'fms_class'] = low
    training_eye_df.loc[(training_eye_df['fms'] >= quantiles[0.25]) & (training_eye_df['fms'] < quantiles[0.75]), 'fms_class'] = mid
    training_eye_df.loc[training_eye_df['fms'] >= quantiles[0.75], 'fms_class'] = high
    
    training_eye_df=training_eye_df.drop(columns=['fms'])
    
    print('merged data after making class')
    for (i,f) in enumerate(training_eye_df['fms_class'].unique()):
        print(f" {f} has datapoints: {len(training_eye_df[training_eye_df['fms_class']==f])}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(training_eye_df['fms_class'], bins=11, edgecolor='black', alpha=0.7)
    plt.title('merged data fms class distribution', fontsize=16)
    plt.xlabel('FMS class', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

    trainY_eye=training_eye_df[['fms_class']]
    trainX_eye=training_eye_df.drop(columns=['fms_class'])
    
    #set scaling
    x_scaler_for_trainX_eye_standard=StandardScaler()
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainY_eye=trainY_eye.to_numpy()
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)
    trainX_eye=x_scaler_for_trainX_eye_standard.fit_transform(trainX_eye)
    
    
    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    #seperating test set
    indx1, indx2, indx3 = np.where(Y_train_eye==low)[0], np.where(Y_train_eye==mid)[0],np.where(Y_train_eye==high)[0]
    random.seed(12)
    test_indx = random.sample(indx1.tolist(), int(len(indx1)/10))
    test_indx.extend(random.sample(indx2.tolist(), int(len(indx2)/10)))
    test_indx.extend(random.sample(indx3.tolist(), int(len(indx3)/10)))
    
    test_X = X_train_eye[test_indx]
    # test_X = test_X.tolist()
    X_train_eye = np.delete(X_train_eye, test_indx, axis=0)
    # X_train_eye.remove(X_train_eye[test_indx])
    # test_X = np.concatenate((test_X, X_train_eye[test_indx_2], ),0)
    # test_X = np.concatenate((test_X, X_train_eye[test_indx_3], ),0)
    test_Y = Y_train_eye[test_indx]
    # test_Y = test_Y.tolist()
    Y_train_eye = np.delete(Y_train_eye, test_indx, axis=0)
    # Y_train_eye.remove(Y_train_eye[test_indx])
    # test_Y = np.concatenate((test_Y, Y_train_eye[test_indx_2], ),0)
    # test_Y = np.concatenate((test_Y, Y_train_eye[test_indx_3], ),0)
        
    #random sampling started
    print('merged data after making class')
    for (i,f) in enumerate(training_eye_df['fms_class'].unique()):
        print(f" {f} has datapoints: {len(np.where(Y_train_eye==f)[0])}")
    indx1,indx2,indx3=np.where(Y_train_eye==low)[0], np.where(Y_train_eye==mid)[0], np.where(Y_train_eye==high)[0]
    batch_needed_for_1, batch_needed_for_2, batch_needed_for_3 = math.floor(Y_train_eye.shape[0]/3)-len(indx1), math.floor(Y_train_eye.shape[0]/3)-len(indx2), math.floor(Y_train_eye.shape[0]/3)-len(indx3)
    
    if batch_needed_for_1>=0:
        random_values_1 = random.sample(indx1.tolist(), batch_needed_for_1)
        indx1=random_values_1+indx1.tolist()
    else:
        batch_needed_for_1=len(indx1)-abs(batch_needed_for_1)
        random_values_1 = random.sample(indx1.tolist(), batch_needed_for_1)
    if batch_needed_for_2>=0:
        random_values_2 = random.sample(indx2.tolist(), batch_needed_for_2)
        indx2=random_values_2+indx2.tolist()
    else:
        batch_needed_for_2=len(indx2)-abs(batch_needed_for_2)
        random_values_2 = random.sample(indx2.tolist(), batch_needed_for_2)
        indx2=random_values_2
    if batch_needed_for_3>=0:
        random_values_3 = random.sample(indx3.tolist(), batch_needed_for_3)
        indx3=random_values_3+indx3.tolist()
    else:
        batch_needed_for_3=len(indx3)-abs(batch_needed_for_3)
        random_values_3 = random.sample(indx3.tolist(), batch_needed_for_3)
        indx3=random_values_3

    print(len(indx1), len(indx2), len(indx3))
    
    for i in tqdm(range(len(indx1))):
        if (i==0):
            X_train_eye_after_balancing = np.expand_dims(X_train_eye[indx1[i]],0)
            X_train_eye_after_balancing = np.concatenate((X_train_eye_after_balancing,np.expand_dims(X_train_eye[indx2[i]],0)),0)
            X_train_eye_after_balancing = np.concatenate((X_train_eye_after_balancing,np.expand_dims(X_train_eye[indx3[i]],0)),0)
            Y_train_eye_after_balancing = np.expand_dims(Y_train_eye[indx1[i]],0)
            Y_train_eye_after_balancing = np.concatenate((Y_train_eye_after_balancing,np.expand_dims(Y_train_eye[indx2[i]],0)),0)
            Y_train_eye_after_balancing = np.concatenate((Y_train_eye_after_balancing,np.expand_dims(Y_train_eye[indx3[i]],0)),0)
        else:
            X_train_eye_after_balancing = np.concatenate((X_train_eye_after_balancing,np.expand_dims(X_train_eye[indx1[i]],0)),0)
            X_train_eye_after_balancing = np.concatenate((X_train_eye_after_balancing,np.expand_dims(X_train_eye[indx2[i]],0)),0)
            X_train_eye_after_balancing = np.concatenate((X_train_eye_after_balancing,np.expand_dims(X_train_eye[indx3[i]],0)),0)
            Y_train_eye_after_balancing = np.concatenate((Y_train_eye_after_balancing,np.expand_dims(Y_train_eye[indx1[i]],0)),0)
            Y_train_eye_after_balancing = np.concatenate((Y_train_eye_after_balancing,np.expand_dims(Y_train_eye[indx2[i]],0)),0)
            Y_train_eye_after_balancing = np.concatenate((Y_train_eye_after_balancing,np.expand_dims(Y_train_eye[indx3[i]],0)),0)
    
    unique_values, counts = np.unique(Y_train_eye_after_balancing, return_counts=True)

    # Print the unique values and their counts
    print("Unique values and their counts:")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")
    
    print("X_train_eye_after_balancing.shape = "+str(X_train_eye_after_balancing.shape))
    print("Y_train_eye_after_balancing = "+str(Y_train_eye_after_balancing.shape))
    
    # Plotting the distribution of the random integers list
    plt.figure(figsize=(10, 6))
    plt.hist(Y_train_eye_after_balancing, bins=11, edgecolor='black', alpha=0.7)
    plt.title('Balancing after Random upsampling and downsampling', fontsize=16)
    plt.xlabel('FMS', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()
    
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold, KFold
    X_train_eye, Y_train_eye = X_train_eye_after_balancing, Y_train_eye_after_balancing
    kf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    
    for i, (train_indices, test_indices) in enumerate(kf.split(X_train_eye, Y_train_eye)):
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        x_val= X_train_eye[test_indices]
        y_val=Y_train_eye[test_indices]
        
        x_test, y_test = test_X, test_Y
        '''
        temp_X= X_train_eye[test_indices]
        temp_y=Y_train_eye[test_indices]
        
        x_val, x_test, y_val, y_test = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)
        '''
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye,x_scaler_for_trainX_eye_standard



