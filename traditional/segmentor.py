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
            )
        elif classifier_type == 'random_forest':
            cfg = config if config else {}
            self.model = RandomForestClassifier(
                n_estimators=cfg.get('n_estimators', 100),
                max_depth=cfg.get('max_depth', None),
                min_samples_split=cfg.get('min_samples_split', 2),
                n_jobs=cfg.get('n_jobs', -1),
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
    
    def prepare_data(self, feature_df, test_size=0.2):
        X = feature_df.drop(columns=['annotation']).values
        y = feature_df['annotation'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model
    
    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def segment_image(self, feature_df, image_shape):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before segmentation")
        
        X = feature_df.values
        X_scaled = self.scaler.transform(X)
        predictions = self.predict(X_scaled)
        
        return predictions.reshape(image_shape)
    
    def fit_predict(self, feature_df, test_size=0.2):
        X_train, X_test, y_train, y_test = self.prepare_data(feature_df, test_size)
        
        self.train(X_train, y_train)
        
        y_pred = self.predict(X_test)
        metrics = self.evaluate(X_test, y_test)
        
        return {
            'predictions': y_pred,
            'true_labels': y_test,
            'metrics': metrics,
            'X_test': X_test,
            'X_train': X_train,
            'y_train': y_train
        }
