import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from virtualenv.activation import python


class DiabetesClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Classifier")

        self.file_path = None
        self.data = None
        self.training_set = None
        self.testing_set = None

        self.label_file = tk.Label(root, text="No file selected")
        self.label_file.pack()

        self.btn_browse = tk.Button(root, text="Browse", command=self.browse_file)
        self.btn_browse.pack(pady=10)

        self.label_percentage = tk.Label(root, text="Enter the percentage of data for training (0-100):")
        self.label_percentage.pack()

        self.entry_percentage = tk.Entry(root)
        self.entry_percentage.pack(pady=5)

        self.btn_process = tk.Button(root, text="Process", command=self.process_data)
        self.btn_process.pack(pady=10)


    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.label_file.config(text=self.file_path)

    def process_data(self):
        try:
            percentage = float(self.entry_percentage.get())
            if percentage <= 0 or percentage > 100:
                raise ValueError("Percentage must be between 0 and 100.")

            self.load_data()
            self.split_data(percentage)
            self.apply_classifiers()
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            self.data.drop(columns=['HbA1c_level'], inplace=True)

            age_bins = [0, 12, 16, 35, 50, 100]
            age_labels = ['Child', 'Teenager', 'Youth', 'Middle Aged', 'Senior']

            self.data['age'] = pd.cut(self.data['age'], bins=age_bins, labels=age_labels)

            low_threshold = 100
            high_threshold = 140

            self.data['blood_glucose_level'] = pd.cut(self.data['blood_glucose_level'],
                                                      bins=[-np.inf, low_threshold, high_threshold, np.inf],
                                                      labels=['Low', 'Medium', 'High'])

            # hba1c_bins = [0, 5.6, 6.5, 6.9, float('inf')]  # Define the bin edges
            # hba1c_labels = ['Normal', 'Prediabetes', 'Diabetes Stage 1', 'Diabetes Stage 2']
            # self.data['HbA1c_level'] = pd.cut(self.data['HbA1c_level'], bins=hba1c_bins, labels=hba1c_labels)

            self.data['bmi'] = pd.cut(self.data['bmi'],
                                      bins=[-np.inf, 18.5, 25, 30, np.inf],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obesity'])
        except Exception:
            messagebox.showerror("Error", "Failed to load data from the file.")
            raise

    def split_data(self, percentage):
        num_records = len(self.data)
        num_training = int(num_records * (percentage / 100))

        shuffled_data = self.data.sample(frac=1, random_state=42)
        self.training_set = shuffled_data[:num_training]
        self.testing_set = shuffled_data[num_training:]

    def apply_classifiers(self):
        training_features = self.training_set.iloc[:, :-1].values
        training_labels = self.training_set.iloc[:, -1].values

        testing_features = self.testing_set.iloc[:, :-1].values
        testing_labels = self.testing_set.iloc[:, -1].values

        bayesian_predictions, bayesian_accuracy = self.apply_bayesian_classifier(training_features, training_labels,
                                                                                 testing_features, testing_labels)

        decision_tree_predictions, decision_tree_accuracy = self.apply_decision_tree_classifier(training_features,
                                                                                               training_labels,
                                                                                               testing_features,
                                                                                               testing_labels)
        output_window2 = tk.Toplevel()
        output_window2.title("Bayesian Classifier")
        output_window2.geometry("500x400")
        output_text = tk.Text(output_window2, height=90, width=180, bg="lightblue", fg="black", font=("Arial", 10))
        output_text.pack(pady=10)

        decision_tree_output = (f"Accuracy of Bayesian classifier: {bayesian_accuracy}\n\n "
                  f"Predicted labels by Bayesian classifier: \n")
        for feature, label in bayesian_predictions:
            decision_tree_output += f"\n{feature}   predicted label: {label}\n"
        output_text.insert(tk.END, decision_tree_output)

        output_window2 = tk.Toplevel()
        output_window2.title("Decision Tree Classifier")
        output_window2.geometry("500x400")
        output_text = tk.Text(output_window2, height=90, width=180, bg="lightblue", fg="black", font=("Arial", 10))
        output_text.pack(pady=10)

        decision_tree_output = (f"Accuracy of Decision Tree classifier: {decision_tree_accuracy}\n\n "
                  f"Predicted labels by Decision Tree classifier: \n")
        for feature, label in decision_tree_predictions:
            decision_tree_output += f"\n{feature}   predicted label: {label}\n"
        output_text.insert(tk.END, decision_tree_output)

        # f"Accuracy of Decision Tree classifier: {decision_tree_accuracy}\n"
        # f"Predicted labels by Decision Tree classifier: {decision_tree_predictions}")

    def apply_bayesian_classifier(self, training_features, training_labels, testing_features, testing_labels):
        # Calculate class probabilities
        unique_classes, class_counts = np.unique(training_labels, return_counts=True)
        class_probabilities = class_counts / len(training_labels)

        # Calculate feature probabilities for each class
        feature_probabilities = []
        all_unique_feature_values = []
        for feature_index in range(training_features.shape[1]):
            feature_values = np.unique(training_features[:, feature_index])
            all_unique_feature_values.append(feature_values)
            if len(feature_values) == 0:
                # No unique feature values, assign uniform probabilities
                feature_probs = [1 / len(unique_classes)] * len(unique_classes)
            else:
                feature_probs = []
                for class_label in unique_classes:
                    class_indices = np.where(training_labels == class_label)[0]
                    class_feature_values = training_features[class_indices, feature_index]
                    feature_value_counts = np.array(
                        [np.count_nonzero(class_feature_values == value) for value in feature_values])
                    feature_value_probs = feature_value_counts / len(class_indices)
                    feature_probs.append(feature_value_probs)
            feature_probabilities.append(feature_probs)

        # Apply the classifier to testing data
        predicted_labels = []
        results = []  # List to store (testing_feature, predicted_label) tuples
        for test_feature in testing_features:
            class_posteriors = []
            for class_label in unique_classes:
                class_posterior = class_probabilities[class_label]
                for feature_index, feature_value in enumerate(test_feature):
                    feature_values_for_index = all_unique_feature_values[feature_index]
                    value_index = np.where(feature_values_for_index == feature_value)[0][0]
                    feature_prob = feature_probabilities[feature_index][class_label][value_index]
                    class_posterior *= feature_prob
                class_posteriors.append(class_posterior)
            predicted_label = unique_classes[np.argmax(class_posteriors)]
            predicted_labels.append(predicted_label)
            results.append((test_feature.tolist(), predicted_label))

        # accuracy = np.mean(predicted_labels == testing_labels)
        accuracy = self.calculate_accuracy(testing_labels, predicted_labels)
        print("Predicted labels:", predicted_labels)
        print("Accuracy:", accuracy)
        return results, accuracy

    def apply_decision_tree_classifier(self, training_features, training_labels, testing_features, testing_labels):
        class Node:
            def __init__(self, feature_index=None, split_value=None, label=None):
                self.feature_index = feature_index  # Index of the feature to split on (categorical data)
                self.split_value = split_value  # Value for splitting categorical data
                self.label = label  # Class label if the node is a leaf
                self.left = None  # Left child node
                self.right = None  # Right child node

        def entropy(labels):
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            probabilities = label_counts / len(labels)
            entropy_value = -np.sum(probabilities * np.log2(probabilities))
            return entropy_value

        def find_best_split(features, labels):
            best_gain = 0
            best_feature_index = None
            best_split_value = None

            for feature_index in range(features.shape[1]):
                unique_values = np.unique(features[:, feature_index])
                for value in unique_values:
                    left_indices = np.where(features[:, feature_index] == value)[0]
                    right_indices = np.where(features[:, feature_index] != value)[0]

                    left_labels = labels[left_indices]
                    right_labels = labels[right_indices]

                    left_entropy = entropy(left_labels)
                    right_entropy = entropy(right_labels)

                    total_entropy = (len(left_labels) / len(labels)) * left_entropy + (
                            len(right_labels) / len(labels)) * right_entropy
                    information_gain = entropy(labels) - total_entropy

                    if information_gain > best_gain:
                        best_gain = information_gain
                        best_feature_index = feature_index
                        best_split_value = value

                return best_feature_index, best_split_value

        def build_tree(features, labels):
            if len(np.unique(labels)) == 1:
                return Node(label=labels[0])

            feature_index, best_split_value = find_best_split(features, labels)
            if feature_index is None or best_split_value is None:
                unique_labels, label_counts = np.unique(labels, return_counts=True)
                most_frequent_label_index = np.argmax(label_counts)
                return Node(label=unique_labels[most_frequent_label_index])

            left_indices = np.where(features[:, feature_index] == best_split_value)[0]
            right_indices = np.where(features[:, feature_index] != best_split_value)[0]

            left_features = features[left_indices]
            left_labels = labels[left_indices]

            right_features = features[right_indices]
            right_labels = labels[right_indices]

            node = Node(feature_index=feature_index, split_value=best_split_value)
            node.left = build_tree(left_features, left_labels)
            node.right = build_tree(right_features, right_labels)

            return node

        def predict(node, instance):
            if node.label is not None:
                return node.label

            if instance[node.feature_index] == node.split_value:
                return predict(node.left, instance)
            else:
                return predict(node.right, instance)

        # Build the decision tree
        root = build_tree(training_features, training_labels)

        # Make predictions on testing data
        predictions = [predict(root, instance) for instance in testing_features]

        # Calculate accuracy
        # accuracy = accuracy_score(testing_labels, predictions)
        accuracy = self.calculate_accuracy(testing_labels, predictions)
        # Prepare results as list of tuples (testing_feature, predicted_label)
        results = [(feature.tolist(), prediction) for feature, prediction in zip(testing_features, predictions)]

        return results, accuracy

    def calculate_accuracy(self, test_features, predictions):
        """
        Calculates the accuracy of a model given test features and predictions.

        Args:
            test_features: A list of lists representing the test features.
            predictions: A list of the predicted labels for the test features.

        Returns:
            The accuracy of the model as a float.
        """

        correct_predictions = 0
        for feature, prediction in zip(test_features, predictions):
            # Assuming the last element of the feature list is the label
            if prediction == feature:
                correct_predictions += 1

        accuracy = correct_predictions / len(test_features)
        return accuracy


if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesClassifierGUI(root)
    root.mainloop()
