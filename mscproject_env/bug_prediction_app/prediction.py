import numpy as np
from dataLoader import ModelLoader, WeightLoader

class BugPredictor:
    model_loader = ModelLoader()
    weight_loader = WeightLoader()
    
    def __init__(self):       
        self.models = self.model_loader.models
        self.threshold = self.weight_loader.threshold
        self.weights = self.weight_loader.weights

    def predict(self, data):
        probabilities = []
        for name, model in self.models.items():
            probabilities.append(model.predict_proba(data)[:, 1])

        combined_probs = np.sum([w * p for w, p in zip(self.weights, probabilities)], axis=0)
        predictions = (combined_probs > self.threshold).astype(int)
        return predictions



