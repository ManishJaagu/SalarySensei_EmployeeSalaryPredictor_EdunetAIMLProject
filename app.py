import pandas as pd
import numpy as np
import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class SalaryPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("\ud83d\udcbc SalarySensei")
        self.root.geometry("1100x800")
        self.root.resizable(True, True)

        self.data = None
        self.processed_data = None
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.trained_models = {}
        self.scores = {}
        self.encoders = {}
        self.scaler = None
        self.model_vars = {}
        self.remove_outliers = tk.BooleanVar()

        self.create_scrollable_app()

    def create_scrollable_app(self):
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scrollbar = tb.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame = tb.Frame(self.canvas)
        self.scroll_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.scroll_window, width=e.width))
        self.scroll_frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.scroll_frame.bind("<Enter>", lambda e: self._bind_to_mousewheel())
        self.scroll_frame.bind("<Leave>", lambda e: self._unbind_from_mousewheel())

        self.setup_ui()

    def _bind_to_mousewheel(self):
        if self.root.tk.call("tk", "windowingsystem") == "aqua":
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_mac)
        else:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_from_mousewheel(self):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_mac(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta)), "units")

    def setup_ui(self):
        frame = self.scroll_frame
        tb.Label(frame, text="SalarySensei - Employee Salary Predictor", font=("Segoe UI", 20, "bold")).pack(pady=15)

        top_frame = tb.Frame(frame)
        top_frame.pack(pady=10)

        tb.Button(top_frame, text="\ud83d\udcc2 Load Dataset", bootstyle="primary", command=self.load_data).pack(side="left", padx=10)
        tb.Button(top_frame, text="\u2b07\ufe0f Download Processed Data", bootstyle="success", command=self.download_processed_data).pack(side="left", padx=10)

        model_frame = tb.LabelFrame(frame, text="Select ML Models", bootstyle="info", padding=10)
        model_frame.pack(pady=10, padx=20, fill="x")

        self.available_models = [
            "Linear Regression", "Decision Tree", "Random Forest",
            "Gradient Boosting", "KNN", "SVR", "Neural Network"
        ]

        for i, model_name in enumerate(self.available_models):
            var = tk.BooleanVar()
            chk = tb.Checkbutton(model_frame, text=model_name, variable=var, bootstyle="round-toggle")
            chk.grid(row=i // 3, column=i % 3, padx=10, pady=5, sticky='w')
            self.model_vars[model_name] = var

        tb.Label(frame, text="Train-Test Split (e.g., 0.8)", font=("Segoe UI", 10)).pack()
        self.split_entry = tb.Entry(frame)
        self.split_entry.insert(0, "0.8")
        self.split_entry.pack(pady=5)

        tb.Checkbutton(frame, text="Remove Outliers", variable=self.remove_outliers, bootstyle="danger").pack(pady=5)

        tb.Button(frame, text="\ud83c\udcb5 Train Model(s)", bootstyle="primary outline", command=self.train_models).pack(pady=5)
        tb.Button(frame, text="\ud83d\udcca View Accuracy Graph", bootstyle="info", command=self.plot_accuracies).pack(pady=5)
        tb.Button(frame, text="\ud83d\udcca Show Outlier Boxplots", bootstyle="warning", command=self.show_boxplots).pack(pady=5)

        self.predict_frame = tb.LabelFrame(frame, text="Enter Inputs for Prediction", bootstyle="warning", padding=10)
        self.predict_frame.pack(pady=10, padx=20, fill="x")
        self.input_entries = {}

        tb.Button(frame, text="\ud83d\udd2e Predict Salary", bootstyle="success", command=self.predict_salary).pack(pady=5)
        tb.Button(frame, text="\ud83d\udcc0 Export Predictions", bootstyle="secondary outline", command=self.export_predictions).pack(pady=5)

        result_frame = tb.LabelFrame(frame, text="Prediction Results", bootstyle="light", padding=10)
        result_frame.pack(pady=10, fill="both", expand=True, padx=20)

        self.result_text = tk.Text(result_frame, height=12, font=("Consolas", 10))
        self.result_text.pack(side="left", fill="both", expand=True)
        y_scroll = tb.Scrollbar(result_frame, command=self.result_text.yview)
        y_scroll.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=y_scroll.set)

    def load_data(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.data = pd.read_csv(file)
            messagebox.showinfo("Success", "Dataset Loaded Successfully!")
            self.prepare_features()

    def prepare_features(self):
        df = self.data.copy()

        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].mean())

        if self.remove_outliers.get():
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        self.processed_data = df

        self.target_window = tk.Toplevel(self.root)
        self.target_window.title("Select Target Column")
        tb.Label(self.target_window, text="Select the target column:").pack()
        self.target_var = tk.StringVar()
        dropdown = tb.Combobox(self.target_window, textvariable=self.target_var, values=list(df.columns))
        dropdown.pack(pady=5)
        tb.Button(self.target_window, text="Confirm", bootstyle="success", command=self.confirm_target).pack(pady=5)

    def confirm_target(self):
        target = self.target_var.get()
        if target not in self.processed_data.columns:
            messagebox.showerror("Error", "Invalid Target Selected")
            return
        self.target_window.destroy()

        self.X = self.processed_data.drop(columns=[target])
        self.y = self.processed_data[target]

        self.scaler = StandardScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)

        for widget in self.predict_frame.winfo_children():
            widget.destroy()

        self.input_entries.clear()
        for col in self.X.columns:
            tb.Label(self.predict_frame, text=col).pack(pady=2)
            if col in self.encoders:
                values = list(self.encoders[col].classes_)
                combo = tb.Combobox(self.predict_frame, values=values)
                combo.pack(pady=2)
                self.input_entries[col] = (combo, self.encoders[col])
            else:
                entry = tb.Entry(self.predict_frame)
                entry.pack(pady=2)
                self.input_entries[col] = entry

    def train_models(self):
        try:
            ratio = float(self.split_entry.get())
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=1 - ratio, random_state=42)

            self.scores.clear()
            self.trained_models.clear()

            for name, var in self.model_vars.items():
                if var.get():
                    if name == "KNN":
                        model = GridSearchCV(KNeighborsRegressor(), {'n_neighbors': range(2, 11)}, cv=5)
                    elif name == "SVR":
                        model = GridSearchCV(SVR(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'epsilon': [0.1, 0.2]}, cv=5)
                    elif name == "Neural Network":
                        model = GridSearchCV(MLPRegressor(max_iter=1500, random_state=42), {
                            'hidden_layer_sizes': [(64,), (64, 32), (128,)],
                            'activation': ['relu', 'tanh'],
                            'solver': ['adam', 'lbfgs']
                        }, cv=3)
                    elif name == "Linear Regression":
                        model = LinearRegression()
                    elif name == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=42)
                    elif name == "Random Forest":
                        model = RandomForestRegressor(random_state=42)
                    elif name == "Gradient Boosting":
                        model = GradientBoostingRegressor(random_state=42)

                    model.fit(self.X_train, self.y_train)
                    final_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
                    score = cross_val_score(final_model, self.X, self.y, cv=5, scoring='neg_mean_absolute_error')
                    accuracy = 100 * (1 + np.mean(score) / self.y.mean())

                    self.trained_models[name] = final_model
                    self.scores[name] = accuracy

            messagebox.showinfo("Training Complete", "Models trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_accuracies(self):
        if not self.scores:
            messagebox.showerror("Error", "No trained models to plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.bar(self.scores.keys(), self.scores.values(), color='royalblue')
        plt.ylabel("Accuracy (%)")
        plt.title("Model Accuracy Comparison")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def show_boxplots(self):
        if self.processed_data is not None:
            self.processed_data.drop(columns=[self.target_var.get()], errors='ignore').boxplot(figsize=(12, 6), rot=45)
            plt.title("Boxplots for Outlier Detection")
            plt.tight_layout()
            plt.show()

    def predict_salary(self):
        try:
            input_data = []
            for col in self.X.columns:
                widget = self.input_entries[col]
                if isinstance(widget, tuple):
                    val = widget[0].get()
                    val = widget[1].transform([val])[0]
                else:
                    val = float(widget.get())
                input_data.append(val)

            input_scaled = self.scaler.transform([input_data])

            self.result_text.delete('1.0', tk.END)
            self.latest_predictions = []

            for model_name, model in self.trained_models.items():
                pred = model.predict(input_scaled)[0]
                accuracy = self.scores.get(model_name, 0)
                self.result_text.insert(tk.END, f"{model_name} Prediction: {pred:.2f} | Accuracy: {accuracy:.2f}%\n")
                self.latest_predictions.append({"Model": model_name, "Prediction": pred, "Accuracy": accuracy})
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def export_predictions(self):
        if not hasattr(self, 'latest_predictions') or not self.latest_predictions:
            messagebox.showerror("Error", "Please run a prediction first.")
            return
        df = pd.DataFrame(self.latest_predictions)
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Exported", f"Predictions saved to:\n{file_path}")

    def download_processed_data(self):
        if self.processed_data is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if file_path:
                self.processed_data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Processed data saved.")
        else:
            messagebox.showerror("Error", "No processed data available.")


if __name__ == '__main__':
    app = tb.Window(themename="cosmo")
    app.state('zoomed')
    SalaryPredictorApp(app)
    app.mainloop()
