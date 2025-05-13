from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from preprocessing import DataPreprocessor
from prediction import BugPredictor
from explanations import ExplanationGenerator
from summarization import SummaryPlotter

class AppController:
    def __init__(self, flask_app):
        self.app = flask_app
        self.explainer = ExplanationGenerator()
        self.predictor = BugPredictor()
        self.preprocessor = DataPreprocessor()
        self.summary_plotter = SummaryPlotter()

        self.uploaded_data = None
        self.processed_data = None
        self.predictions = None
        self.method_names = []
        self.feature_names = []

        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule("/", "home", self.home, methods=["GET", "POST"])
        self.app.add_url_rule("/explanation/<int:index>", "explanation", self.explanation)

    def home(self):
        if request.method == "POST":
            # Clear old LIME explanation images
            self.explainer.clear_explanation_images()

            file = request.files["file"]
            if not file:
                return render_template("index.html", error="No file uploaded")

            try:
                self.uploaded_data = pd.read_csv(file)
                if 'Method name' not in self.uploaded_data.columns:
                    return render_template("error.html", message="CSV must contain 'Method name' column")
                
                file_name = file.filename
                self.method_names = self.uploaded_data['Method name'].tolist()

                self.processed_data, self.feature_names = self.preprocessor.preprocess(self.uploaded_data)
                self.predictions = self.predictor.predict(self.processed_data)

                # Prepare results
                results = [{
                    "Index": i,
                    "Method": name,
                    "Prediction": "Buggy" if pred == 1 else "Not Buggy"
                } for i, (name, pred) in enumerate(zip(self.method_names, self.predictions))]

                # Pie chart data
                total_methods = len(self.predictions)
                buggy_count = sum(self.predictions)
                not_buggy_count = total_methods - buggy_count
                pie_chart_path = self.summary_plotter.generate_pie_chart(buggy_count, not_buggy_count)

                return render_template("index.html", results=results, file_name=file_name,
                                       total_methods=total_methods, buggy_count=buggy_count,
                                       not_buggy_count=not_buggy_count, pie_chart_path=pie_chart_path)
            except ValueError as e:
                return render_template('error.html', message=str(e))
    
            except Exception as e:
                return render_template('error.html', message=f"Unexpected error: {e}")
                # return render_template("index.html", error=str(e))

        return render_template("index.html")

    def explanation(self, index):
        try:
            method_name = self.method_names[index]
            explanation_path, etext = self.explainer.get_lime_explain(
                self.processed_data, index, self.feature_names
            )

            relative_path = os.path.join("explanations", os.path.basename(explanation_path)).replace("\\", "/")
            return render_template("explanation.html", method_name=method_name,
                                   explanation_path=relative_path, etext=etext)
        except Exception as e:
            return f"Error generating explanation: {str(e)}", 500


# Initialize and run Flask app
app = Flask(__name__)
controller = AppController(app)

if __name__ == "__main__":
    app.run(debug=True)




