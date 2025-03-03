 # Model training & evaluation funcs

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# This function trains models to classify left vs. right hand movement imagery
# (and evaluates how well they perform)
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
   
    # Standardize features
    # This scales all features to have mean=0 and std=1
    # (Important for many ML algorithms to work properly)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'SVM': SVC(kernel='rbf', probability=True), # Support Vector Machine
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=23),
        'Multi-Layer Perceptron': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42) 
        
    
    }
    
    results = {} # Will hold accuracy scores
    conf_matrices = {} # Will hold confusion matrices
    

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model on the training data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test_scaled)

         # Calculate accuracy (percentage of correct predictions)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Store results for visualization
        results[name] = accuracy
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    return results, conf_matrices