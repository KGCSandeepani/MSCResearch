import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import dill
import glob
import os
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

models = {
    
    "RandomForest" :  RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5, 
                                       min_samples_leaf=2, class_weight='balanced', random_state=42),
    "AdaBoostClassifier" : AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42),
    "BaggingClassifier" : BaggingClassifier(n_estimators=100, max_samples=0.8, random_state=42),
    "KNeighborsClassifier" : KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan'),
    "MLPClassifier" : MLPClassifier(activation='relu', hidden_layer_sizes=(200,100), max_iter=2000, 
                               learning_rate='adaptive', random_state=42),
    "HistGradientBoostingClassifier" : HistGradientBoostingClassifier(random_state=42),   
    "DecisionTreeClassifier" : DecisionTreeClassifier(random_state=42),
    "SVC" : SVC(random_state=42, probability=True, C=10, kernel='poly', gamma='scale'),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.05, depth=4, verbose=0, random_state=42)
}

# Load models
loaded_models = {}
for name in models.keys():
    with open(f"../Models/{name}_model.pkl", "rb") as f:
        loaded_models[name] = pickle.load(f)

# Load weights and threshold
with open("../weights_threshold.pkl", "rb") as f:
    saved_data = pickle.load(f)
    weights = saved_data["weights"]
    best_threshold_pr = saved_data["best_threshold_pr"]

with open("../lime_explainer.pkl", "rb") as f:
    explainer = dill.load(f)

scaler = StandardScaler()

feature_dic = {
    "C1"	:"Lines of Code",
    "C2"	:"Number of Comment lines",
    "C3"	:"Number of all lines",
    "C4"	:"Number of Blank lines",
    "C5"	:"Number of Declare lines",
    "C6"	:"Number of Executable lines",
    "C7"	:"Number of Parameters",
    "C8"	:"Number of Statements",
    "C9"	:"Number of Declare Statements",
    "C10"	:"Number of Executable Statements",
    "C11"	:"Halstead-Vocabulary",
    "C12"	:"Halstead-Length",
    "C13"	:"Halstead-Difficulty",
    "C14"	:"Halstead-Volume",
    "C15"	:"Halstead-Effort",
    "C16"	:"Halstead-Bugs",
    "C17"	:"Cyclomatic complexity",
    "C18"	:"Number of Paths",
    "C19"	:"Max Nesting",
    "C20"	:"Fan-in",
    "C21"	:"Fan-out",
	"H1"	:"Added LOC",
	"H2"	:"Deleted LOC",
	"H3"	:"All Changed LOC",
	"H4"	:"Number of Changes",
	"H5"	:"Number of Authors",
	"H6"	:"Number of Modified Statements",
	"H7"	:"Number of Modified Expressions",
	"H8"	:"Number of Modified Comments",
	"H9"	:"Number of Modified Return type",
	"H10"	:"Number of Modified Parameters",
	"H11"	:"Number of Modified Prefix",
	"H12"	:"Added LOC / LOC",
	"H13"	:"Deleted LOC / LOC",
	"H14"	:"Added LOC / Deleted LOC",
	"H15"	:"Changed LOC / Number of Changes",
	"H16"	:"Number of Modified Statements / Number of Statements",
	"H17"	:"Number of Modified Expressions / Number of Statements",
	"H18"	:"Number of Modified Comments/ LOC",
	"H19"	:"Number of Modified Parameters/ Number of Parameters",
}


def preprocessdata(data):
    trdata_X = data.drop(columns=['Method name','bug-prone'])
    feature_names = trdata_X.columns
    trdata_X_scaled = scaler.fit_transform(trdata_X)
    return trdata_X_scaled, feature_names

def predictdata(data):
    tr_probabilities = []
    # Train all models and get probabilities
    for name, model in loaded_models.items():
        y_pred_proba = model.predict_proba(data)[:, 1]  # Probability for class 1
        tr_probabilities.append(y_pred_proba)

    tr_combined_probs = np.sum([weight * probs for weight, probs in zip(weights, tr_probabilities)], axis=0)

    # Convert probabilities to binary predictions using threshold
    tr_predictions = (tr_combined_probs > best_threshold_pr).astype(int)
    return tr_predictions


def data_with_colname(data):    
    df = pd.DataFrame(data, columns=feature_names)
    return df

def predict_fn(x):
    probabilitieslime = [model.predict_proba(x)[:, 1] for model in loaded_models.values()]
    combined_probs_lime = np.sum([w * p for w, p in zip(weights, probabilitieslime)], axis=0)
    return np.column_stack((1 - combined_probs_lime, combined_probs_lime))



# Ensure a folder for saving explanation images
EXPLANATION_FOLDER = "static/explanations"
if not os.path.exists(EXPLANATION_FOLDER):
    os.makedirs(EXPLANATION_FOLDER)



def get_explanation_meaning(importance):
    """ Generate a human-readable explanation for feature importance. """
    if importance > 0.1:
        impact = "strongly increases"
    elif importance > 0:
        impact = "slightly increases"
    elif importance < -0.1:
        impact = "strongly decreases"
    else:
        impact = "slightly decreases"

    return f"This {impact} the probability of predicting 'Buggy'."


def get_lime_explain_plot(df, index):
    """Generates and saves LIME explanation as an image."""
    df = data_with_colname(df)
    
    exp = explainer.explain_instance(
        data_row=df.iloc[index],  
        predict_fn=predict_fn  
    )

    # Generate the explanation plot
    fig = exp.as_pyplot_figure()
    
    # Save the figure as a PNG image
    explanation_path = os.path.join(EXPLANATION_FOLDER, f"explanation_{index}.png")
    fig.savefig(explanation_path, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Process LIME explanations
    # explanation_list = exp.as_list()  # Get feature importance list
    # etext = [(feature, importance, get_explanation_meaning(importance)) for feature, importance in explanation_list]

    feature_list = exp.as_map()[1]
    etext = [(df.columns[feature_id], feature_dic[df.columns[feature_id]], importance, get_explanation_meaning(importance)) 
             for feature_id, importance in feature_list]

    return explanation_path, etext



def clear_explanation_images():
    """Deletes all existing PNG images in the explanation folder."""
    for file in glob.glob(os.path.join(EXPLANATION_FOLDER, "*.png")):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


def summery_plot(buggy_count, not_buggy_count):
    # Generate pie chart
    labels = ["Buggy", "Not Buggy"]
    sizes = [buggy_count, not_buggy_count]
    colors = ["#df944d", "#66cc66"]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.

    pie_chart_path = os.path.join("static", "buggy_pie_chart.png")
    fig.savefig(pie_chart_path)
    plt.close(fig)
    return pie_chart_path




@app.route("/", methods=["GET", "POST"])
def home():
    global uploaded_data, processed_data, predictions, method_names, feature_names

    if request.method == "POST":
        # Clear old LIME explanation images
        clear_explanation_images()

        file = request.files["file"]
        if not file:
            return render_template("index.html", error="No file uploaded")

        try:
            uploaded_data = pd.read_csv(file)  
            if 'Method name' not in uploaded_data.columns:
                print("CSV must contain 'Method name' column")
                return render_template("index.html", error="CSV must contain 'Method name' column")
            file_name = file.filename
            method_names = uploaded_data['Method name'].tolist()
            processed_data, feature_names = preprocessdata(uploaded_data)  
            predictions = predictdata(processed_data)  

            results = [{"Index": i, "Method": name, "Prediction": "Buggy" if pred == 1 else "Not Buggy"} 
                       for i, (name, pred) in enumerate(zip(method_names, predictions))]

            # Count buggy and non-buggy methods
            total_methods = len(predictions)
            buggy_count = sum(predictions)
            not_buggy_count = total_methods - buggy_count
            pie_chart_path = summery_plot(buggy_count, not_buggy_count)

            # return render_template("index.html", results=results)
            return render_template("index.html", results=results, file_name=file_name, total_methods=total_methods, 
                                   buggy_count=buggy_count, not_buggy_count=not_buggy_count, 
                                   pie_chart_path=pie_chart_path)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


@app.route("/explanation/<int:index>")
def explanation(index):

    try:
        print(f"Generating explanation for index: {index}")  # Debugging log
        method_name = method_names[index]  # Get method name
        explanation_path, etext = get_lime_explain_plot(processed_data, index)

        # return render_template("explanation.html", method_name=method_name, explanation_path=explanation_path)

        relative_path = os.path.join("explanations", os.path.basename(explanation_path)).replace("\\", "/")
        return render_template("explanation.html", method_name=method_name, explanation_path=relative_path, etext=etext)


    except Exception as e:
        return f"Error generating explanation: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)








# v3
# @app.route("/", methods=["GET", "POST"])
# def home():
#     global uploaded_data, processed_data, predictions

#     if request.method == "POST":
#         file = request.files["file"]
#         if not file:
#             return render_template("index.html", error="No file uploaded")

#         try:
#             uploaded_data = pd.read_csv(file)  
#             if 'Method name' not in uploaded_data.columns:
#                 return render_template("index.html", error="CSV must contain 'Method name' column")

#             method_names = uploaded_data['Method name'].tolist()
#             processed_data = preprocessdata(uploaded_data)  
#             predictions = predictdata(processed_data)  

#             results = [{"Index": i, "Method": name, "Prediction": "Buggy" if pred == 1 else "Not Buggy"} 
#                        for i, (name, pred) in enumerate(zip(method_names, predictions))]

#             return render_template("index.html", results=results)

#         except Exception as e:
#             return render_template("index.html", error=str(e))

#     return render_template("index.html")

# @app.route("/explain/<int:index>", methods=["GET"])
# def explain(index):
#     """Returns the path to the LIME explanation image."""
#     try:
#         explanation_path = get_lime_explain(processed_data, index)
#         return jsonify({"image_url": explanation_path})
#     except Exception as e:
#         return jsonify({"error": str(e)})
    

# if __name__ == "__main__":
#     app.run(debug=True)


# v2
# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         file = request.files["file"]
#         if not file:
#             return render_template("index.html", error="No file uploaded")

#         try:
#             data = pd.read_csv(file)  # Read CSV file
            
#             if 'Method name' not in data.columns:
#                 return render_template("index.html", error="CSV file must contain a 'Method name' column")

#             method_names = data['Method name'].tolist()  # Extract method names
#             df = preprocessdata(data)  # Preprocess data
#             predictions = predictdata(df)  # Get predictions

#             # Format results with method names
#             results = [{"Method": name, "Prediction": "Buggy" if pred == 1 else "Not Buggy"} 
#                        for name, pred in zip(method_names, predictions)]

#             return render_template("index.html", results=results)

#         except Exception as e:
#             return render_template("index.html", error=str(e))

#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)



# v1
# @app.route("/")
# def home():
#     return render_template("index.html")  # HTML form to upload files

# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files["file"]
#     if not file:
#         return jsonify({"error": "No file uploaded"})

#     try:
#         data = pd.read_csv(file)  # Read CSV file
#         if 'Method name' not in data.columns:
#             return jsonify({"error": "CSV file must contain a 'Method name' column"})

#         method_names = data['Method name'].tolist()  # Extract method names

#         df = preprocessdata(data)
#         predictions = predictdata(df)
#         # Format results with method names
#         results = [{"Method": name, "Prediction": "Buggy" if pred == 1 else "Not Buggy"} 
#                    for name, pred in zip(method_names, predictions)]
        
#         return jsonify({"results": results})
    
#     except Exception as e:
#         return jsonify({"error": str(e)})