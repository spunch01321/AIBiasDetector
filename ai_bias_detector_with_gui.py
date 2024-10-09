import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class AIBiasDetector:
    def __init__(self):
        self.data = None
        self.target = None
        self.sensitive_features = []
        self.model = None
        self.preprocessor = None
        self.feature_names = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        return self.data.head()
    
    def set_target(self, target_column):
        if target_column in self.data.columns:
            self.target = self.data[target_column]
            self.data = self.data.drop(columns=[target_column])
            print(f"Target variable set to: {target_column}")
        else:
            raise ValueError(f"{target_column} not found in the dataset.")
    
    def set_sensitive_features(self, features):
        self.sensitive_features = [f for f in features if f in self.data.columns]
        if len(self.sensitive_features) != len(features):
            print(f"Warning: Some features were not found in the dataset.")
        print(f"Sensitive features set to: {self.sensitive_features}")
    
    def preprocess_data(self):
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.data.select_dtypes(include=['object']).columns

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ])

        self.data_preprocessed = self.preprocessor.fit_transform(self.data)
        
        # Get feature names for output features
        numeric_feature_names = numeric_features.tolist()
        try:
            categorical_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        except AttributeError:
            categorical_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names(categorical_features).tolist()
        
        self.feature_names = numeric_feature_names + categorical_feature_names
        print("Data preprocessing completed.")
    
    def train_model(self, n_folds=5):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        for train_index, test_index in kf.split(self.data_preprocessed):
            X_train, X_test = self.data_preprocessed[train_index], self.data_preprocessed[test_index]
            y_train, y_test = self.target.iloc[train_index], self.target.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            fold_scores.append(self.model.score(X_test, y_test))
        
        print(f"Cross-validation scores: {fold_scores}")
        print(f"Average CV score: {np.mean(fold_scores):.4f}")
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
    
    def detect_bias(self):
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained or feature names not set. Run preprocess_data and train_model first.")
        
        bias_scores = {}
        feature_importances = self.model.feature_importances_
        
        for feature in self.sensitive_features:
            related_columns = [i for i, col in enumerate(self.feature_names) if col.startswith(feature)]
            importance = np.sum(feature_importances[related_columns])
            bias_scores[feature] = importance
        
        # Normalize the scores
        total_importance = sum(bias_scores.values())
        for feature in bias_scores:
            bias_scores[feature] /= total_importance
        
        return bias_scores
    
    def visualize_bias(self, bias_scores, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(x=list(bias_scores.keys()), y=list(bias_scores.values()), ax=ax)
        ax.set_title("Relative Importance of Sensitive Attributes")
        ax.set_xlabel("Sensitive Features")
        ax.set_ylabel("Normalized Importance Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return ax

class AIBiasDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Bias Detector")
        self.master.geometry("800x600")

        self.detector = AIBiasDetector()
        self.data = None

        self.create_widgets()

    def create_widgets(self):
        self.file_button = tk.Button(self.master, text="Select Data File", command=self.load_file)
        self.file_button.pack(pady=10)

        self.target_label = tk.Label(self.master, text="Select Target Column:")
        self.target_label.pack()
        self.target_var = tk.StringVar(self.master)
        self.target_menu = tk.OptionMenu(self.master, self.target_var, "")
        self.target_menu.pack()

        self.sensitive_label = tk.Label(self.master, text="Select Sensitive Features:")
        self.sensitive_label.pack()
        self.sensitive_listbox = tk.Listbox(self.master, selectmode=tk.MULTIPLE)
        self.sensitive_listbox.pack()

        self.n_folds_label = tk.Label(self.master, text="Number of CV Folds:")
        self.n_folds_label.pack()
        self.n_folds_entry = tk.Entry(self.master)
        self.n_folds_entry.insert(0, "5")
        self.n_folds_entry.pack()

        self.run_button = tk.Button(self.master, text="Run Bias Detection", command=self.run_detection)
        self.run_button.pack(pady=10)

        self.results_text = tk.Text(self.master, height=10, width=50)
        self.results_text.pack()

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = self.detector.load_data(file_path)
                self.update_column_options()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def update_column_options(self):
        menu = self.target_menu["menu"]
        menu.delete(0, "end")
        for column in self.detector.data.columns:
            menu.add_command(label=column, command=lambda col=column: self.target_var.set(col))

        self.sensitive_listbox.delete(0, tk.END)
        for column in self.detector.data.columns:
            self.sensitive_listbox.insert(tk.END, column)

    def run_detection(self):
        if self.detector.data is None:
            messagebox.showerror("Error", "Please load data first.")
            return

        target_column = self.target_var.get()
        if not target_column:
            messagebox.showerror("Error", "Please select a target column.")
            return

        sensitive_features = [self.sensitive_listbox.get(idx) for idx in self.sensitive_listbox.curselection()]
        if not sensitive_features:
            messagebox.showerror("Error", "Please select at least one sensitive feature.")
            return

        try:
            n_folds = int(self.n_folds_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of folds. Please enter a valid integer.")
            return

        try:
            self.detector.set_target(target_column)
            self.detector.set_sensitive_features(sensitive_features)
            self.detector.preprocess_data()
            self.detector.train_model(n_folds=n_folds)
            bias_scores = self.detector.detect_bias()

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Bias Scores (Normalized):\n")
            for feature, score in bias_scores.items():
                self.results_text.insert(tk.END, f"{feature}: {score:.4f}\n")

            self.ax.clear()
            self.detector.visualize_bias(bias_scores, ax=self.ax)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIBiasDetectorGUI(root)
    root.mainloop()
