import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ai_bias_detector import AIBiasDetector

class AIBiasDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Bias Detector")
        self.master.geometry("800x600")

        self.detector = AIBiasDetector()
        self.data = None

        self.create_widgets()

    def create_widgets(self):
        # File selection
        self.file_button = tk.Button(self.master, text="Select Data File", command=self.load_file)
        self.file_button.pack(pady=10)

        # Target column selection
        self.target_label = tk.Label(self.master, text="Select Target Column:")
        self.target_label.pack()
        self.target_var = tk.StringVar(self.master)
        self.target_menu = tk.OptionMenu(self.master, self.target_var, "")
        self.target_menu.pack()

        # Sensitive features selection
        self.sensitive_label = tk.Label(self.master, text="Select Sensitive Features:")
        self.sensitive_label.pack()
        self.sensitive_listbox = tk.Listbox(self.master, selectmode=tk.MULTIPLE)
        self.sensitive_listbox.pack()

        # Run button
        self.run_button = tk.Button(self.master, text="Run Bias Detection", command=self.run_detection)
        self.run_button.pack(pady=10)

        # Results area
        self.results_text = tk.Text(self.master, height=10, width=50)
        self.results_text.pack()

        # Plot area
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
        # Update target column options
        menu = self.target_menu["menu"]
        menu.delete(0, "end")
        for column in self.data.columns:
            menu.add_command(label=column, command=lambda col=column: self.target_var.set(col))

        # Update sensitive features listbox
        self.sensitive_listbox.delete(0, tk.END)
        for column in self.data.columns:
            self.sensitive_listbox.insert(tk.END, column)

    def run_detection(self):
        if self.data is None:
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
            self.detector.set_target(target_column)
            self.detector.set_sensitive_features(sensitive_features)
            self.detector.preprocess_data()
            self.detector.train_model()
            bias_scores = self.detector.detect_bias()

            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Bias Scores:\n")
            for feature, score in bias_scores.items():
                self.results_text.insert(tk.END, f"{feature}: {score:.4f}\n")

            # Visualize bias
            self.ax.clear()
            self.detector.visualize_bias(bias_scores, ax=self.ax)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIBiasDetectorGUI(root)
    root.mainloop()