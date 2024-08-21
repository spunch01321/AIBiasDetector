class AIBiasDetector:
    def __init__(self):
        self.data = None
        self.target = None
        self.sensitive_features = []
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
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
        for feature in features:
            if feature in self.data.columns:
                self.sensitive_features.append(feature)
            else:
                print(f"Warning: {feature} not found in the dataset.")
        print(f"Sensitive features set to: {self.sensitive_features}")
    
    def preprocess_data(self):
        self.data = pd.get_dummies(self.data, columns=self.sensitive_features)
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numerical_features] = self.scaler.fit_transform(self.data[numerical_features])
        print("Data preprocessing completed.")
    
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
    
    def detect_bias(self):
        bias_scores = {}
        for feature in self.sensitive_features:
            feature_cols = [col for col in self.data.columns if col.startswith(feature)]
            feature_importance = self.model.feature_importances_[self.data.columns.isin(feature_cols)].sum()
            bias_scores[feature] = feature_importance
        return bias_scores
    
    def visualize_bias(self, bias_scores):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(bias_scores.keys()), y=list(bias_scores.values()))
        plt.title("Feature Importance of Sensitive Attributes")
        plt.xlabel("Sensitive Features")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    detector = AIBiasDetector()
    detector.load_data("sample_loan_data.csv")
    detector.set_target("loan_approved")
    detector.set_sensitive_features(["gender", "race", "age"])
    detector.preprocess_data()
    detector.train_model()
    bias_scores = detector.detect_bias()
    detector.visualize_bias(bias_scores)