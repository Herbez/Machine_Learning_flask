from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_test', methods=['POST'])
def train_test():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    selected_model = request.form.get('model')
    
    if selected_model == 'decision_tree':
        # Load the Breast Cancer Wisconsin dataset
        breast_cancer = load_breast_cancer()
        
        # Access the features and target variable
        X = breast_cancer.data
        y = breast_cancer.target

        # Step 4: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 5: Create a Decision Tree model
        model = DecisionTreeClassifier(random_state=42)
        
        # Step 6: Train the model on the training set
        model.fit(X_train, y_train)
        
        # Step 7: Make predictions on the testing set
        y_pred = model.predict(X_test)
        
        # Step 8: Evaluate the model
        accuracy_train = accuracy_score(y_train, model.predict(X_train))
        accuracy_test = accuracy_score(y_test, y_pred)
        accuracy_model = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Save the confusion matrix visualization as an image
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=breast_cancer.target_names, yticklabels=breast_cancer.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('static/confusion_matrix.png')  # Save the plot to the static folder
        plt.close()
    
        return render_template('result.html',
            accuracy_train=f"Training Accuracy: {accuracy_train * 100:.2f}%",
            accuracy_model=f"Accuracy Score: {accuracy_model}",
            accuracy_test=f"Testing Accuracy: {accuracy_test * 100:.2f}%",
            confusion_matrix=conf_matrix,
            classification_report=classification_rep)
    
    elif selected_model == 'random_forest':
        breast_cancer = load_breast_cancer()
        # Access the features and target variable
        X = breast_cancer.data
        y = breast_cancer.target

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        random_forest_predictions = model.predict(X_test_scaled)

        accuracy_train = model.score(X_train_scaled, y_train)
        accuracy_test = model.score(X_test_scaled, y_test)
        accuracy_model2 = model.score(X_test_scaled, y_test)
        conf_matrix = confusion_matrix(y_test, random_forest_predictions)
        classification_rep = classification_report(y_test, random_forest_predictions)

        # Save the confusion matrix visualization as an image
        cm = confusion_matrix(y_test, random_forest_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=breast_cancer.target_names, yticklabels=breast_cancer.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('static/random_confusion_matrix.png')  # Save the plot to the static folder
        plt.close()  # Close the plot to prevent it from being displayed in the Flask app

        return render_template('result2.html',
            train_accuracy=f"Training Accuracy: {accuracy_train * 100:.2f}%",
            test_accuracy=f"Testing Accuracy: {accuracy_test * 100:.2f}%",
            accuracy_model2 = f"Accuracy Score: {accuracy_model2}",
            confusion_matrix=f"Confusion Matrix:{conf_matrix}",
            classification_report=f"Classification Report:{classification_rep}")
    
    else:
        return render_template('index.html', error='Invalid model selection')

if __name__ == '__main__':
    app.run(debug=True)
