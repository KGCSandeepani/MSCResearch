import pickle
import dill

class ModelLoader:
    def __init__(self, model_dir="../Models"):
        self.model_names = [
                                "RandomForest", "AdaBoostClassifier", "BaggingClassifier", "KNeighborsClassifier",
                                "MLPClassifier", "HistGradientBoostingClassifier", "DecisionTreeClassifier",
                                "SVC", "XGBoost", "CatBoost"
                            ]
        self.model_dir = model_dir
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        for name in self.model_names:
            with open(f"{self.model_dir}/{name}_model.pkl", "rb") as f:
                models[name] = pickle.load(f)
        return models


class WeightLoader:
    def __init__(self, path="../weights_threshold.pkl"):
        self.path = path
        self.weights = None
        self.threshold = None
        self._load()

    def _load(self):
        with open(self.path, "rb") as f:
            saved_data = pickle.load(f)
            self.weights = saved_data["weights"]
            self.threshold = saved_data["best_threshold_pr"]


class LimeExplainerLoader:
    def __init__(self, path="../lime_explainer.pkl"):
        self.path = path
        self.explainer = self._load()

    def _load(self):
        with open(self.path, "rb") as f:
            return dill.load(f)


class FeatureDictionary:
    def __init__(self):
        self.features = {
            "C1": "Lines of Code",
            "C2": "Number of Comment lines",
            "C3": "Number of all lines",
            "C4": "Number of Blank lines",
            "C5": "Number of Declare lines",
            "C6": "Number of Executable lines",
            "C7": "Number of Parameters",
            "C8": "Number of Statements",
            "C9": "Number of Declare Statements",
            "C10": "Number of Executable Statements",
            "C11": "Halstead-Vocabulary",
            "C12": "Halstead-Length",
            "C13": "Halstead-Difficulty",
            "C14": "Halstead-Volume",
            "C15": "Halstead-Effort",
            "C16": "Halstead-Bugs",
            "C17": "Cyclomatic complexity",
            "C18": "Number of Paths",
            "C19": "Max Nesting",
            "C20": "Fan-in",
            "C21": "Fan-out",
            "H1": "Added LOC",
            "H2": "Deleted LOC",
            "H3": "All Changed LOC",
            "H4": "Number of Changes",
            "H5": "Number of Authors",
            "H6": "Number of Modified Statements",
            "H7": "Number of Modified Expressions",
            "H8": "Number of Modified Comments",
            "H9": "Number of Modified Return type",
            "H10": "Number of Modified Parameters",
            "H11": "Number of Modified Prefix",
            "H12": "Added LOC / LOC",
            "H13": "Deleted LOC / LOC",
            "H14": "Added LOC / Deleted LOC",
            "H15": "Changed LOC / Number of Changes",
            "H16": "Number of Modified Statements / Number of Statements",
            "H17": "Number of Modified Expressions / Number of Statements",
            "H18": "Number of Modified Comments/ LOC",
            "H19": "Number of Modified Parameters/ Number of Parameters",
        }





