import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.svm import SVC
import xgboost as xgb
import io
import base64

# Function to download model
def get_model_download_link(model, filename):
    """Generate a link to download the trained model"""
    import pickle
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

# Function to evaluate and display model results
def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    st.subheader(f"{model_name} Results")
    
    # Create two columns for the metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.write("Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)
    
    with col2:
        if y_pred_proba is not None:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            st.pyplot(plt)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR curve (AP = {ap_score:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            st.pyplot(plt)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred)
    st.text(report)
    
    return model, y_pred_proba, accuracy, roc_auc if y_pred_proba is not None else None

def show():
    st.title("Predictive Maintenance Analysis and Modeling")
    
    # Initialize session state for storing data and models
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'encoder' not in st.session_state:
        st.session_state.encoder = None
        
    # Display tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Loading & Preprocessing", "Data Exploration", "Model Training", "Prediction"])
    
    with tab1:
        st.header("Data Loading and Preprocessing")
        
        # Option to load data from UCI repository or upload file
        data_source = st.radio("Select data source:", ["UCI ML Repository", "Upload CSV file"])
        
        if data_source == "UCI ML Repository":
            st.info("Loading data from UCI Machine Learning Repository...")
            try:
                from ucimlrepo import fetch_ucirepo
                
                # Fetch dataset
                ai4i_dataset = fetch_ucirepo(id=601)
                
                # Extract features and targets
                X = ai4i_dataset.data.features
                y = ai4i_dataset.data.targets
                
                # Combine them
                data = pd.concat([X, y], axis=1)
                
                st.success("Data loaded successfully from UCI repository!")
                st.session_state.data = data
                
            except Exception as e:
                st.error(f"Error loading data from UCI repository: {e}")
                st.info("If the UCI repository access fails, please upload a CSV file instead.")
        
        else:  # Upload CSV file option
            uploaded_file = st.file_uploader("Upload predictive maintenance CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success("CSV file uploaded successfully!")
                    st.session_state.data = data
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        
        # If data is loaded, show preprocessing options
        if st.session_state.data is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head())
            
            st.subheader("Data Preprocessing")
            
            # Optional preprocessing steps
            st.write("Select preprocessing steps:")
            
            drop_cols = st.multiselect(
                "Select columns to drop:",
                options=st.session_state.data.columns.tolist(),
                default=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"] if all(col in st.session_state.data.columns for col in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]) else []
            )
            
            scale_data = st.checkbox("Scale numerical features", value=True)
            
            if st.button("Preprocess Data"):
                try:
                    # Make a copy of the data to avoid modifying the original
                    processed_data = st.session_state.data.copy()
                    
                    # Drop selected columns
                    if drop_cols:
                        processed_data = processed_data.drop(columns=drop_cols, errors='ignore')
                    
                    # Handle categorical variables (assuming 'Type' is the categorical feature)
                    if 'Type' in processed_data.columns:
                        encoder = LabelEncoder()
                        processed_data['Type'] = encoder.fit_transform(processed_data['Type'])
                        st.session_state.encoder = encoder
                    
                    # Check for missing values
                    missing_values = processed_data.isnull().sum()
                    if missing_values.sum() > 0:
                        st.warning("Missing values found in the data:")
                        st.write(missing_values[missing_values > 0])
                    else:
                        st.success("No missing values found in the data.")
                    
                    # Determine target column (assuming 'Machine failure' or 'Target' is the target)
                    target_col = None
                    for col in ['Machine failure', 'Target']:
                        if col in processed_data.columns:
                            target_col = col
                            break
                    
                    if target_col is None:
                        st.error("Target column not found. Please make sure the dataset contains 'Machine failure' or 'Target' column.")
                        return
                    
                    # Split features and target
                    X = processed_data.drop(columns=[target_col])
                    y = processed_data[target_col]
                    
                    # Scale numerical features if selected
                    if scale_data:
                        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
                        scaler = StandardScaler()
                        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                        st.session_state.scaler = scaler
                    
                    # Split data into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Save to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.processed_data = processed_data
                    
                    st.success("Data preprocessing completed successfully!")
                    
                    # Display preprocessed data summary
                    st.subheader("Preprocessed Data Summary")
                    st.write(f"Training set shape: {X_train.shape}")
                    st.write(f"Testing set shape: {X_test.shape}")
                    st.write(f"Target distribution in training set: {pd.Series(y_train).value_counts().to_dict()}")
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    
    with tab2:
        st.header("Data Exploration")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Show basic statistics
            st.subheader("Basic Statistics")
            st.write(data.describe())
            
            # Target distribution
            target_col = None
            for col in ['Machine failure', 'Target']:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col:
                st.subheader(f"Target Distribution ({target_col})")
                fig, ax = plt.subplots(figsize=(10, 6))
                target_counts = data[target_col].value_counts()
                sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax)
                plt.xlabel(target_col)
                plt.ylabel("Count")
                plt.title(f"Distribution of {target_col}")
                # Add count labels on top of bars
                for i, v in enumerate(target_counts.values):
                    ax.text(i, v + 5, str(v), ha='center')
                st.pyplot(fig)
                
                # Calculate imbalance ratio
                imbalance_ratio = target_counts.values[0] / target_counts.values[1] if len(target_counts) > 1 else "N/A"
                st.write(f"Class imbalance ratio: {imbalance_ratio:.2f}")
            
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            plt.figure(figsize=(12, 10))
            corr = numeric_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
            plt.title("Correlation Matrix")
            st.pyplot(plt)
            
            # Feature distributions
            st.subheader("Feature Distributions")
            
            # Select features for distribution plots
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_features:
                # Create distribution plots
                fig, axes = plt.subplots(len(selected_features), 1, figsize=(10, 4 * len(selected_features)))
                if len(selected_features) == 1:
                    axes = [axes]
                
                for i, feature in enumerate(selected_features):
                    if target_col:
                        sns.histplot(data=data, x=feature, hue=target_col, multiple="stack", ax=axes[i])
                    else:
                        sns.histplot(data=data, x=feature, ax=axes[i])
                    axes[i].set_title(f"Distribution of {feature}")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature relationships
            st.subheader("Feature Relationships")
            
            if len(selected_features) >= 2:
                # Select features for scatter plot
                x_feature = st.selectbox("Select X-axis feature:", selected_features)
                y_feature = st.selectbox("Select Y-axis feature:", 
                               [f for f in selected_features if f != x_feature], 
                               index=min(1, len(selected_features)-1))
                
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                if target_col:
                    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=target_col, ax=ax)
                else:
                    sns.scatterplot(data=data, x=x_feature, y=y_feature, ax=ax)
                plt.title(f"{y_feature} vs {x_feature}")
                st.pyplot(fig)
                
                # Calculate correlation
                corr_value = data[[x_feature, y_feature]].corr().iloc[0, 1]
                st.write(f"Correlation between {x_feature} and {y_feature}: {corr_value:.4f}")
            
            # Box plots for feature comparison by target
            if target_col:
                st.subheader("Features by Target Class")
                
                feature_for_box = st.selectbox("Select feature for boxplot:", selected_features)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=data, x=target_col, y=feature_for_box, ax=ax)
                plt.title(f"{feature_for_box} by {target_col}")
                st.pyplot(fig)
        else:
            st.info("Please load and preprocess data first.")
    
    with tab3:
        st.header("Model Training")
        
        if (st.session_state.X_train is not None and 
            st.session_state.X_test is not None and 
            st.session_state.y_train is not None and 
            st.session_state.y_test is not None):
            
            st.subheader("Select Models to Train")
            
            # Model selection
            use_logreg = st.checkbox("Logistic Regression", value=True)
            use_rf = st.checkbox("Random Forest", value=True)
            use_xgb = st.checkbox("XGBoost", value=True)
            use_svm = st.checkbox("Support Vector Machine", value=False)
            
            # Training button
            if st.button("Train Selected Models"):
                results = []
                best_model = None
                best_score = -1
                best_model_name = ""
                
                with st.spinner("Training models... This may take a moment."):
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test
                    
                    # Train and evaluate selected models
                    if use_logreg:
                        with st.expander("Logistic Regression Results", expanded=True):
                            model = LogisticRegression(max_iter=1000, random_state=42)
                            model.fit(X_train, y_train)
                            model, y_pred_proba, accuracy, roc_auc = evaluate_model(model, X_test, y_test, "Logistic Regression")
                            results.append(("Logistic Regression", model, accuracy, roc_auc))
                    
                    if use_rf:
                        with st.expander("Random Forest Results", expanded=True):
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            model, y_pred_proba, accuracy, roc_auc = evaluate_model(model, X_test, y_test, "Random Forest")
                            results.append(("Random Forest", model, accuracy, roc_auc))
                    
                    if use_xgb:
                        with st.expander("XGBoost Results", expanded=True):
                            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
                            model.fit(X_train, y_train)
                            model, y_pred_proba, accuracy, roc_auc = evaluate_model(model, X_test, y_test, "XGBoost")
                            results.append(("XGBoost", model, accuracy, roc_auc))
                    
                    if use_svm:
                        with st.expander("Support Vector Machine Results", expanded=True):
                            model = SVC(kernel='rbf', probability=True, random_state=42)
                            model.fit(X_train, y_train)
                            model, y_pred_proba, accuracy, roc_auc = evaluate_model(model, X_test, y_test, "Support Vector Machine")
                            results.append(("Support Vector Machine", model, accuracy, roc_auc))
                
                # Compare models and find the best one
                if results:
                    st.subheader("Model Comparison")
                    
                    comparison_data = []
                    for name, model, accuracy, roc_auc in results:
                        comparison_data.append({
                            "Model": name,
                            "Accuracy": accuracy,
                            "ROC-AUC": roc_auc
                        })
                        if roc_auc > best_score:
                            best_score = roc_auc
                            best_model = model
                            best_model_name = name
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Save best model to session state
                    st.session_state.best_model = best_model
                    st.session_state.best_model_name = best_model_name
                    
                    # Visualize comparison
                    plt.figure(figsize=(10, 6))
                    bar_width = 0.35
                    index = np.arange(len(comparison_data))
                    
                    plt.bar(index, comparison_df["Accuracy"], bar_width, label="Accuracy")
                    plt.bar(index + bar_width, comparison_df["ROC-AUC"], bar_width, label="ROC-AUC")
                    
                    plt.xlabel("Model")
                    plt.ylabel("Score")
                    plt.title("Model Performance Comparison")
                    plt.xticks(index + bar_width / 2, comparison_df["Model"])
                    plt.legend()
                    plt.ylim(0, 1)
                    
                    st.pyplot(plt)
                    
                    st.success(f"Best model: {best_model_name} with ROC-AUC score of {best_score:.4f}")
                    
                    # Feature importance for tree-based models
                    if best_model_name in ["Random Forest", "XGBoost"]:
                        st.subheader("Feature Importance")
                        
                        feature_names = st.session_state.X_train.columns
                        importances = best_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        plt.figure(figsize=(10, 6))
                        plt.title(f"Feature Importance ({best_model_name})")
                        plt.bar(range(len(importances)), importances[indices])
                        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                        plt.tight_layout()
                        st.pyplot(plt)
                    
                    # Download trained model
                    if best_model:
                        st.markdown(get_model_download_link(best_model, f"{best_model_name.replace(' ', '_').lower()}_model.pkl"), unsafe_allow_html=True)
        else:
            st.info("Please load and preprocess data first.")
    
    with tab4:
        st.header("Make Predictions")
        
        if st.session_state.best_model is not None:
            st.subheader("Predict Equipment Failure")
            
            # Get feature names from training data
            feature_names = st.session_state.X_train.columns
            
            # Create input form for prediction
            st.write("Enter values for prediction:")
            
            with st.form("prediction_form"):
                input_values = {}
                
                # For each feature, create an input field
                for feature in feature_names:
                    # Check if feature is Type (categorical)
                    if feature == 'Type':
                        if st.session_state.encoder:
                            # Display as original categories
                            type_value = st.selectbox(f"{feature}", ["L", "M", "H"])
                            # Will be transformed later
                            input_values[feature] = type_value
                        else:
                            # If no encoder saved, just use numerical input
                            input_values[feature] = st.number_input(f"{feature}", value=0)
                    else:
                        # For numerical features
                        # Try to set reasonable defaults based on feature name
                        if "temperature" in feature.lower():
                            default_value = 300.0  # Reasonable temperature in K
                        elif "speed" in feature.lower():
                            default_value = 1500.0  # Reasonable rotation speed
                        elif "torque" in feature.lower():
                            default_value = 40.0  # Reasonable torque
                        elif "wear" in feature.lower():
                            default_value = 100.0  # Reasonable tool wear time
                        else:
                            default_value = 0.0
                        
                        input_values[feature] = st.number_input(f"{feature}", value=default_value)
                
                submit = st.form_submit_button("Make Prediction")
            
            if submit:
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_values])
                    
                    # Handle categorical features
                    if 'Type' in input_df.columns and st.session_state.encoder:
                        input_df['Type'] = st.session_state.encoder.transform([input_df['Type'].iloc[0]])
                    
                    # Scale numerical features if a scaler was used
                    if st.session_state.scaler:
                        numerical_cols = input_df.select_dtypes(include=['float64', 'int64']).columns
                        input_df[numerical_cols] = st.session_state.scaler.transform(input_df[numerical_cols])
                    
                    # Make prediction
                    prediction = st.session_state.best_model.predict(input_df)
                    prediction_proba = st.session_state.best_model.predict_proba(input_df)[0]
                    
                    # Display prediction results
                    st.subheader("Prediction Result")
                    
                    if prediction[0] == 1:
                        st.error(f"⚠️ Equipment Failure Predicted! (Probability: {prediction_proba[1]:.4f})")
                    else:
                        st.success(f"✅ No Failure Predicted (Probability of no failure: {prediction_proba[0]:.4f})")
                    
                    # Display probability values
                    st.write("Prediction Probabilities:")
                    prob_df = pd.DataFrame({
                        'Class': ['No Failure', 'Failure'],
                        'Probability': prediction_proba
                    })
                    
                    # Bar chart for probabilities
                    plt.figure(figsize=(8, 4))
                    bars = plt.bar(prob_df['Class'], prob_df['Probability'], color=['green', 'red'])
                    
                    # Add values on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom')
                    
                    plt.ylim(0, 1)
                    plt.ylabel('Probability')
                    plt.title('Prediction Probabilities')
                    
                    st.pyplot(plt)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        else:
            st.info("Please train a model first.")