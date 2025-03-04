import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from PIL import Image, ImageTk
import os
import sys

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from PIL import Image, ImageTk
import os
import sys

class PhishingEmailDetectorApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Phishing Email Detector")
        self.configure_gui()
        self.create_widgets()

    def configure_gui(self):
        self.master.configure(bg="#6495ED")  # Set background color to a light blue (Cornflower Blue)

    def create_widgets(self):
        # Create and pack the label for entering email
        self.email_label = tk.Label(self.master, text="WELCOME TO THE PHISHING EMAIL DETECTOR\nPlease Enter The Email You Would Like To Check In The Space Below:", bg="#6495ED", fg="#0000FF", font=("Arial", 14, "bold"))
        self.email_label.pack(pady=(20, 5))

        # Create and pack the text entry field for email
        self.email_entry = scrolledtext.ScrolledText(self.master, width=60, height=10)
        self.email_entry.pack(pady=(0, 10))

        # Create and pack the button for analyzing email
        self.analyze_button = tk.Button(self.master, text="Analyze", command=self.analyze_email, bg="#0000FF", fg="#FFFFFF", font=("Arial", 12, "bold"))
        self.analyze_button.pack(pady=(5, 10))

        # Create and pack the button for saving email
        self.save_button = tk.Button(self.master, text="Save Email", command=self.save_email, bg="#008000", fg="#FFFFFF", font=("Arial", 12, "bold"))
        self.save_button.pack(pady=(5, 10))

        # Create and pack the text field for displaying analysis result
        self.result_text = scrolledtext.ScrolledText(self.master, width=60, height=5, wrap=tk.WORD)
        self.result_text.pack(pady=(0, 20))

        # Configure text tags for result text field
        self.result_text.tag_config("phishing", foreground="#FF0000", font=("Arial", 10, "bold"))
        self.result_text.tag_config("not_phishing", foreground="#008000", font=("Arial", 10, "italic"))
        self.result_text.tag_config("warning", foreground="#FFA500", font=("Arial", 10, "underline"))

        # Initialize the image label
        self.img_label = tk.Label(self.master, bg="#6495ED")
        self.img_label.pack(pady=10)  # Pack the image label with padding

    def analyze_email(self):
        raw_email = self.email_entry.get("1.0", tk.END).strip()
        if raw_email:
            prediction = self.predict_email(raw_email)
            self.result_text.delete(1.0, tk.END)
            if prediction == 1:
                self.result_text.insert(tk.END, "WARNING!!! THIS IS A PHISHING EMAIL!!! IT IS NOT SAFE!!!!", "phishing")
                self.show_image("X.png")
            else:
                self.result_text.insert(tk.END, "This is not a Phishing email. Thank you for using the Phishing Email Detector.", "not_phishing")
                self.show_image("checkmark.png")
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please enter an email to analyze.", "warning")

    def predict_email(self, raw_email):
        processed_email = self.preprocess_text(raw_email)
        X = self.vectorizer.transform([processed_email])
        prediction = self.model.predict(X)[0]
        return prediction

    def preprocess_text(self, text):
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        words.extend(urls)
        return ' '.join(words)

    def show_image(self, image_filename):
        try:
            # Get the directory containing the executable
            exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

            # Construct path to the images directory relative to the executable
            image_dir = os.path.join(exe_dir, "images")
            image_path = os.path.join(image_dir, image_filename)

            img = Image.open(image_path)
            img = img.resize((100, 100), Image.LANCZOS)  # Resize the image with LANCZOS filter for antialiasing
            photo_img = ImageTk.PhotoImage(img)

            # Update the existing image label with the new image
            self.img_label.configure(image=photo_img)
            self.img_label.image = photo_img  # Keep a reference to the PhotoImage object
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")

    def save_email(self):
        email_content = self.email_entry.get("1.0", tk.END).strip()
        if email_content:
            # Specify the directory to save user-submitted emails
            save_dir = "user_emails"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Generate a unique filename (e.g., using timestamp)
            filename = f"user_email_{len(os.listdir(save_dir)) + 1}.txt"
            filepath = os.path.join(save_dir, filename)

            # Write the email content to a text file
            with open(filepath, "w") as file:
                file.write(email_content)

            # Provide feedback to the user
            messagebox.showinfo("Email Saved", "The email has been saved successfully.")
        else:
            messagebox.showwarning("Empty Email", "Please enter an email before saving.")

def get_file_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Email Dataset File", filetypes=[("Text files", "*.txt")])
    root.destroy()  # Destroy the file dialog window after selection
    return file_path

def load_dataset(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("File not found. Please select a valid file.")
        return []

    emails = []
    labels = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            email_text = '\t'.join(parts[:-1])
            emails.append(email_text)
            labels.append(1 if parts[-1].strip() == "Phishing" else 0)

    dataset = list(zip(emails, labels))
    return dataset

def train_classifier(dataset):
    emails, labels = zip(*dataset)
    app = PhishingEmailDetectorApp(master=tk.Tk())  # Create a Tkinter root window explicitly
    app.vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = app.vectorizer.fit_transform([app.preprocess_text(email) for email in emails])
    X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, labels, test_size=0.2, random_state=42)

    app.model = RandomForestClassifier(n_estimators=100)
    app.model.fit(X_train, y_train)

    app.mainloop()  # Start the main event loop of the application

if __name__ == "__main__":
    dataset_path = get_file_path()
    print("Dataset file selected:", dataset_path)

    if dataset_path:
        print("Loading dataset and training the classifier...")
        dataset = load_dataset(dataset_path)
        if dataset:
            train_classifier(dataset)
        else:
            print("No valid dataset found.")
    else:
        print("No dataset selected.")
class PhishingEmailDetectorApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Phishing Email Detector")
        self.configure_gui()
        self.create_widgets()

    def configure_gui(self):
        self.master.configure(bg="#6495ED")  # Set background color to a light blue (Cornflower Blue)

    def create_widgets(self):
        # Create and pack the label for entering email
        self.email_label = tk.Label(self.master, text="WELCOME TO THE PHISHING EMAIL DETECTOR\nPlease Enter The Email You Would Like To Check In The Space Below:", bg="#6495ED", fg="#0000FF", font=("Arial", 14, "bold"))
        self.email_label.pack(pady=(20, 5))

        # Create and pack the text entry field for email
        self.email_entry = scrolledtext.ScrolledText(self.master, width=60, height=10)
        self.email_entry.pack(pady=(0, 10))

        # Create and pack the button for analyzing email
        self.analyze_button = tk.Button(self.master, text="Analyze", command=self.analyze_email, bg="#0000FF", fg="#FFFFFF", font=("Arial", 12, "bold"))
        self.analyze_button.pack(pady=(5, 10))

        # Create and pack the button for saving email
        self.save_button = tk.Button(self.master, text="Save Email", command=self.save_email, bg="#008000", fg="#FFFFFF", font=("Arial", 12, "bold"))
        self.save_button.pack(pady=(5, 10))

        # Create and pack the text field for displaying analysis result
        self.result_text = scrolledtext.ScrolledText(self.master, width=60, height=5, wrap=tk.WORD)
        self.result_text.pack(pady=(0, 20))

        # Configure text tags for result text field
        self.result_text.tag_config("phishing", foreground="#FF0000", font=("Arial", 10, "bold"))
        self.result_text.tag_config("not_phishing", foreground="#008000", font=("Arial", 10, "italic"))
        self.result_text.tag_config("warning", foreground="#FFA500", font=("Arial", 10, "underline"))

        # Initialize the image label
        self.img_label = tk.Label(self.master, bg="#6495ED")
        self.img_label.pack(pady=10)  # Pack the image label with padding

    def analyze_email(self):
        raw_email = self.email_entry.get("1.0", tk.END).strip()
        if raw_email:
            prediction = self.predict_email(raw_email)
            self.result_text.delete(1.0, tk.END)
            if prediction == 1:
                self.result_text.insert(tk.END, "WARNING!!! THIS IS A PHISHING EMAIL!!! IT IS NOT SAFE!!!!", "phishing")
                self.show_image("X.png")
            else:
                self.result_text.insert(tk.END, "This is not a Phishing email. Thank you for using the Phishing Email Detector.", "not_phishing")
                self.show_image("checkmark.png")
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please enter an email to analyze.", "warning")

    def predict_email(self, raw_email):
        processed_email = self.preprocess_text(raw_email)
        X = self.vectorizer.transform([processed_email])
        prediction = self.model.predict(X)[0]
        return prediction

    def preprocess_text(self, text):
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
        porter = PorterStemmer()
        words = [porter.stem(word) for word in words]
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        words.extend(urls)
        return ' '.join(words)

    def show_image(self, image_filename):
        try:
            # Get the directory containing the executable
            exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)

            # Construct path to the images directory relative to the executable
            image_dir = os.path.join(exe_dir, "images")
            image_path = os.path.join(image_dir, image_filename)

            img = Image.open(image_path)
            img = img.resize((100, 100), Image.LANCZOS)  # Resize the image with LANCZOS filter for antialiasing
            photo_img = ImageTk.PhotoImage(img)

            # Update the existing image label with the new image
            self.img_label.configure(image=photo_img)
            self.img_label.image = photo_img  # Keep a reference to the PhotoImage object
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")

    def save_email(self):
        email_content = self.email_entry.get("1.0", tk.END).strip()
        if email_content:
            # Specify the directory to save user-submitted emails
            save_dir = "user_emails"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Generate a unique filename (e.g., using timestamp)
            filename = f"user_email_{len(os.listdir(save_dir)) + 1}.txt"
            filepath = os.path.join(save_dir, filename)

            # Write the email content to a text file
            with open(filepath, "w") as file:
                file.write(email_content)

            # Provide feedback to the user
            messagebox.showinfo("Email Saved", "The email has been saved successfully.")
        else:
            messagebox.showwarning("Empty Email", "Please enter an email before saving.")

def get_file_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Email Dataset File", filetypes=[("Text files", "*.txt")])
    root.destroy()  # Destroy the file dialog window after selection
    return file_path

def load_dataset(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("File not found. Please select a valid file.")
        return []

    emails = []
    labels = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            email_text = '\t'.join(parts[:-1])
            emails.append(email_text)
            labels.append(1 if parts[-1].strip() == "Phishing" else 0)

    dataset = list(zip(emails, labels))
    return dataset

def train_classifier(dataset):
    emails, labels = zip(*dataset)
    app = PhishingEmailDetectorApp(master=tk.Tk())  # Create a Tkinter root window explicitly
    app.vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vectorized = app.vectorizer.fit_transform([app.preprocess_text(email) for email in emails])
    X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, labels, test_size=0.2, random_state=42)

    app.model = RandomForestClassifier(n_estimators=100)
    app.model.fit(X_train, y_train)

    app.mainloop()  # Start the main event loop of the application

if __name__ == "__main__":
    dataset_path = get_file_path()
    print("Dataset file selected:", dataset_path)

    if dataset_path:
        print("Loading dataset and training the classifier...")
        dataset = load_dataset(dataset_path)
        if dataset:
            train_classifier(dataset)
        else:
            print("No valid dataset found.")
    else:
        print("No dataset selected.")