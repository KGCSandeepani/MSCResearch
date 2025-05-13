import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataLoader import ModelLoader, WeightLoader, LimeExplainerLoader, FeatureDictionary

class ExplanationGenerator:
    EXPLANATION_FOLDER = "static/explanations"
    model_loader = ModelLoader()
    weight_loader = WeightLoader()
    explainer_loader = LimeExplainerLoader()
    feature_dict = FeatureDictionary().features

    def __init__(self):
        if not os.path.exists(self.EXPLANATION_FOLDER):
            os.makedirs(self.EXPLANATION_FOLDER)

    def clear_explanation_images(self):
        # Deletes all old explanation images
        for file in glob.glob(os.path.join(self.EXPLANATION_FOLDER, "*.png")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

    def _data_with_colname(self, data, feature_names):
        return pd.DataFrame(data, columns=feature_names)

    def _predict_fn(self, x):
        probabilitieslime = [
            model.predict_proba(x)[:, 1] for model in self.model_loader.models.values()
        ]
        combined_probs = np.sum(
            [w * p for w, p in zip(self.weight_loader.weights, probabilitieslime)], axis=0
        )
        return np.column_stack((1 - combined_probs, combined_probs))

    def get_lime_explain(self, data, index, feature_names):
        df = self._data_with_colname(data, feature_names)

        exp = self.explainer_loader.explainer.explain_instance(
            data_row=df.iloc[index], predict_fn=self._predict_fn
        )

        # Plot explanation
        fig = exp.as_pyplot_figure()
        ax = plt.gca()

        # Prediction probabilities
        probas = exp.predict_proba
        class_names = self.explainer_loader.explainer.class_names
        ax.text(-0.15, 1.2, "Prediction Probabilities", fontsize=12, ha='left', transform=ax.transAxes)
        ax.text(-0.15, 1.15, f"{class_names[0]}: {probas[0]:.2f}", fontsize=12, color='#df944d', ha='left', transform=ax.transAxes)
        ax.text(-0.15, 1.1, f"{class_names[1]}: {probas[1]:.2f}", fontsize=12, color='#66cc66', ha='left', transform=ax.transAxes)

        # Save plot
        explanation_path = os.path.join(self.EXPLANATION_FOLDER, f"explanation_{index}.png")
        fig.savefig(explanation_path, bbox_inches='tight')
        plt.close(fig)

        # Feature importance explanation
        feature_importances = exp.as_map()[1]
        etext = [
            (
                df.columns[feature_id],
                self.feature_dict.get(df.columns[feature_id], df.columns[feature_id]),
                importance,
                self._get_explanation_meaning(importance)
            )
            for feature_id, importance in feature_importances
        ]

        return explanation_path, etext

    def _get_explanation_meaning(self, importance):
        if importance > 0.1:
            return "strongly increases the probability of predicting 'Buggy'."
        elif importance > 0:
            return "slightly increases the probability of predicting 'Buggy'."
        elif importance < -0.1:
            return "strongly decreases the probability of predicting 'Buggy'."
        else:
            return "slightly decreases the probability of predicting 'Buggy'."



