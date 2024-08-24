AI Bias Detector
Overview
The AI Bias Detector is a tool designed to identify and quantify potential biases in machine learning models, with a focus on fairness in decision-making systems such as loan approval processes. This project aims to help data scientists and AI practitioners assess and mitigate unfair bias in their models.
Features

Data loading and preprocessing
Configurable target variable and sensitive features
Random Forest Classifier for model training
Bias detection based on feature importance
Graphical User Interface (GUI) for easy interaction
Visualization of bias scores

Installation

Clone this repository:
git clone https://github.com/spunch01321/ai-bias-detector.git
cd ai-bias-detector

Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:
pip install -r requirements.txt


Usage
Command-line Interface
To use the AI Bias Detector from the command line:

Ensure you're in the project directory.
Run the following command:
python ai_bias_detector.py path_to_your_data.csv

Follow the prompts to select the target variable and sensitive features.

Graphical User Interface
To use the GUI version of the AI Bias Detector:

Run the following command:
python ai_bias_detector_gui.py

Use the interface to load your data, select the target variable and sensitive features, and run the bias detection.

Input Data Format
The AI Bias Detector expects input data in CSV format. The CSV file should include:

A target variable (e.g., loan approval status)
Features used for prediction
Sensitive attributes (e.g., gender, race, age)

Example:
age,gender,race,income,credit_score,loan_approved
35,Male,Caucasian,50000,700,1
28,Female,African American,45000,680,0...


Interpreting Results
The tool provides bias scores for each sensitive feature based on its importance in the model's decision-making process. Higher scores indicate a stronger influence on the model's predictions, which may suggest potential bias.
Contributing
Contributions to the AI Bias Detector are welcome! Please feel free to submit pull requests, create issues or spread the word.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Scikit-learn for machine learning utilities
Tkinter for the graphical user interface
Matplotlib and Seaborn for data visualization

Contact
For any queries or feedback, please open an issue on this GitHub repository.
