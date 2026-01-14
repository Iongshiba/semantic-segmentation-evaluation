import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Segmentor:
    def __init__(self, classifier_type='svm', config=None):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        if classifier_type == 'svm':
            cfg = config if config else {}
            self.model = SVC(
                kernel=cfg.get('kernel', 'rbf'),
                C=cfg.get('C', 1.0),
                gamma=cfg.get('gamma', 'scale'),
                probability=cfg.get('probability', True),
                verbose=True,
            )
        elif classifier_type == 'random_forest':
            cfg = config if config else {}
            self.model = RandomForestClassifier(
                n_estimators=cfg.get('n_estimators', 100),
                max_depth=cfg.get('max_depth', None),
                min_samples_split=cfg.get('min_samples_split', 2),
                n_jobs=cfg.get('n_jobs', -1),
                verbose=True,
            )
        elif classifier_type == 'knn':
            cfg = config if config else {}
            self.model = KNeighborsClassifier(
                n_neighbors=cfg.get('n_neighbors', 5),
                weights=cfg.get('weights', 'uniform'),
                algorithm=cfg.get('algorithm', 'auto'),
                n_jobs=cfg.get('n_jobs', -1),
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def prepare_data(self, feature_df):
        X = feature_df.drop(columns=['annotation']).values
        y = feature_df['annotation'].values
        
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model
    
    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def segment_image(self, feature_df, image_shape):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before segmentation")
        
        X = feature_df.values
        X_scaled = self.scaler.transform(X)
        predictions = self.predict(X_scaled)
        
        return predictions.reshape(image_shape)
    
    def fit_predict(self, feature_df):
        X, y = self.prepare_data(feature_df)
        
        model = self.train(X, y)

        return model