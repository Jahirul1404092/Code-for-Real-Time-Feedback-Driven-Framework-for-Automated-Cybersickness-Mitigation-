# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 04:13:28 2024

@author: Jahirul
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, BatchNormalization, Reshape, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import tensorflow as tf

cwd = os.getcwd()
save_dir = os.path.join(cwd, "eye_pred_folder","sim21_maze_data")
model_save_dir = 'CNN_LSTM_Classification'

import sim21_maze as merged_data
import Utils as utils

X_train_list,Y_train_list,X_val_list,Y_val_list,X_test_list,Y_test_list,x_scaler,x_scaler_std=merged_data.get_maze_sim21_cross_class_data()

accuracy, precision, recall, f1 = [],[],[],[]
overall_y_true = []
overall_y_pred = []

for i, (X_train_eye,Y_train_eye,X_val_eye,Y_val_eye,X_test_eye,Y_test_eye) in enumerate(zip(X_train_list,Y_train_list,X_val_list,Y_val_list,X_test_list,Y_test_list)):
    print(X_train_eye.shape,Y_train_eye.shape,X_val_eye.shape,Y_val_eye.shape,X_test_eye.shape,Y_test_eye.shape)

    # Ensure Y_train_eye is of integer type
    Y_train_eye = Y_train_eye.astype(int)

    X_train_eye = X_train_eye.reshape(-1, 28 * 14).reshape(-1, 28, 14)
    X_val_eye = X_val_eye.reshape(-1, 28 * 14).reshape(-1, 28, 14)
    X_test_eye = X_test_eye.reshape(-1, 28 * 14).reshape(-1, 28, 14)
    
    # Define the improved CNN-LSTM model
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(28, 14)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    # Additional Conv1D Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())  # Replaces Flatten to reduce overfitting
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Reshape((1, 256)))  # Reshape for LSTM input
    
    # Enhanced LSTM layers
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    
    # Compile the model
    learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Compute class weights based on class distribution
    class_counts = np.bincount(Y_train_eye.flatten())
    total_samples = len(Y_train_eye)
    class_weights = {i: total_samples / count for i, count in enumerate(class_counts)}
    
    # Callbacks for early stopping and saving the best model weights
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_save_dir = os.path.join(save_dir, model_save_dir)
    model_fold_save_dir = os.path.join(save_dir, model_save_dir, f'fold_{i}')
    
    if not os.path.exists(model_fold_save_dir):
        os.makedirs(model_fold_save_dir)
        
    checkpoint = ModelCheckpoint(filepath=os.path.join(model_fold_save_dir, 'best_model_weights.h5'), monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
    
    # Train the model
    history = model.fit(X_train_eye, Y_train_eye, validation_data=(X_val_eye, Y_val_eye), epochs=200, batch_size=32, class_weight=class_weights, callbacks=[early_stopping, checkpoint, reduce_lr])
    model.save(str(model_fold_save_dir)+'/model.h5')
    # Save training accuracy and loss curves
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_fold_save_dir, 'accuracy_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_fold_save_dir, 'loss_curve.png'))
    plt.close()
    
    # Make predictions on the test set
    y_pred_eye = np.argmax(model.predict(X_test_eye), axis=1)
    
    # Accumulate true and predicted labels for overall confusion matrix
    overall_y_true.extend(Y_test_eye)
    overall_y_pred.extend(y_pred_eye)
    
    # Save y_test_eye and y_pred_eye as .npy files
    np.save(os.path.join(model_fold_save_dir, 'Y_test_eye.npy'), Y_test_eye)
    np.save(os.path.join(model_fold_save_dir, 'y_pred_eye.npy'), y_pred_eye)
    
    # Save confusion matrix
    cm = confusion_matrix(Y_test_eye, y_pred_eye)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(model_fold_save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate performance metrics for this fold
    a, p, r, f = utils.performance_metrics(Y_test_eye, y_pred_eye)
    
    # Print and store the performance metrics
    print(f'Accuracy: {a:.2f}')
    print(f'Precision: {p:.2f}')
    print(f'Recall: {r:.2f}')
    print(f'F1 Score: {f:.2f}')
    accuracy.append(a), precision.append(p), recall.append(r), f1.append(f)
    
    # Clear session after each fold to free memory
    tf.keras.backend.clear_session()

# Compute overall confusion matrix across all folds
overall_cm = confusion_matrix(overall_y_true, overall_y_pred)

# Save overall confusion matrix as heatmap
plt.figure(figsize=(8,6))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(model_save_dir, 'overall_confusion_matrix.png'))
plt.close()

# Save x_scale as a .pkl file
x_scaler=[x_scaler,x_scaler_std]
with open(os.path.join(model_save_dir, 'x_scaler(minmax_standard).pkl'), 'wb') as f:
    pickle.dump(x_scaler, f)

# Create a DataFrame to store the fold results and mean values
metrics_df = pd.DataFrame({
    'Fold': [f'Fold_{i+1}' for i in range(len(accuracy))],
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

# Create a DataFrame for the mean values
mean_metrics = pd.DataFrame({
    'Fold': ['Mean'],
    'Accuracy': [np.mean(accuracy)],
    'Precision': [np.mean(precision)],
    'Recall': [np.mean(recall)],
    'F1 Score': [np.mean(f1)]
})

# Use pd.concat() to add the mean values to the metrics_df
metrics_df = pd.concat([metrics_df, mean_metrics], ignore_index=True)

# Round the values in the DataFrame to 3 decimal places
metrics_df = metrics_df.round(3)

# Save the DataFrame as a CSV file with values rounded to 3 decimal places
metrics_df.to_csv(os.path.join(model_save_dir, 'performance_metrics.csv'), index=False)

# Print final performance metrics
for i, (a, p, r, f) in enumerate(zip(accuracy, precision, recall, f1)):
    print(f'fold_{i+1}: accuracy= {a:.2f}, precision= {p:.2f}, recall= {r:.2f}, f1_score= {f:.2f}')
    
print(f'Mean: accuracy= {np.mean(accuracy):.2f}, precision= {np.mean(precision):.2f}, recall: {np.mean(recall):.2f}, F1_score: {np.mean(f1):.2f}')
