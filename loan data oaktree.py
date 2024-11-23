import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ImprovedLoanDefaultModel:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, index_col=0)
        self.numeric_features = [
            'person_age', 'person_income', 'person_emp_exp', 
            'loan_int_rate', 'loan_percent_income', 
            'cb_person_cred_hist_length', 'credit_score',
            'loan_to_income_ratio', 'dependents_count',
            'regional_unemployment_rate', 'borrower_risk_score'
        ]
        self.categorical_features = [
            'person_gender', 'person_education', 'person_home_ownership',
            'loan_intent', 'previous_loan_defaults_on_file', 'loan_type'
        ]
        self.output_path = r"C:\Users\valim\Downloads\test\loan ML model.xlsx"

    def safe_divide(self, a, b, fill_value=0):
        """Safely divide two columns, handling division by zero"""
        return np.where(b != 0, a / b, fill_value)
        
    def feature_engineering(self):
        """Enhanced feature engineering with safety checks"""
        df_engineered = self.df.copy()
        
        # Safe calculations for new features
        df_engineered['debt_burden'] = (
            df_engineered['loan_percent_income'] * 
            df_engineered['loan_int_rate']
        ).clip(0, 1e6)
        
        # Safe division for income per dependent
        df_engineered['income_per_dependent'] = self.safe_divide(
            df_engineered['person_income'],
            (df_engineered['dependents_count'] + 1)
        ).clip(0, 1e6)
        
        # Safe calculation for credit income ratio
        df_engineered['credit_income_ratio'] = self.safe_divide(
            df_engineered['credit_score'],
            df_engineered['loan_to_income_ratio'],
            fill_value=df_engineered['credit_score'].median()
        ).clip(0, 1e6)
        
        # Create risk buckets
        df_engineered['risk_bucket'] = pd.qcut(
            df_engineered['borrower_risk_score'],
            q=5,
            labels=['very_low_risk', 'low_risk', 'medium_risk', 'high_risk', 'very_high_risk']
        )
        
        # Update feature lists
        new_numeric_features = ['debt_burden', 'income_per_dependent', 'credit_income_ratio']
        self.numeric_features.extend(new_numeric_features)
        self.categorical_features.append('risk_bucket')
        
        # Remove any remaining infinite values
        for col in new_numeric_features:
            df_engineered[col] = df_engineered[col].replace([np.inf, -np.inf], np.nan)
            df_engineered[col] = df_engineered[col].fillna(df_engineered[col].median())
        
        return df_engineered
    
    def create_pipeline(self):
        """Create preprocessing pipeline with tuned XGBoost"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        # Tuned XGBoost parameters for better precision
        clf = xgb.XGBClassifier(
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=4,
            min_child_weight=2,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.0,
            eval_metric='aucpr',
            random_state=42
        )
            
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
    
    def save_to_excel(self, classification_rep, conf_matrix, roc_auc, threshold, conf_matrix_path):
        """Save results to Excel file"""
        # Create a directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Convert classification report to DataFrame
        class_report_df = pd.DataFrame(classification_rep).transpose()
        
        # Convert confusion matrix to DataFrame
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        # Create Excel writer
        with pd.ExcelWriter(self.output_path, engine='xlsxwriter') as writer:
            # Write classification report
            class_report_df.to_excel(writer, sheet_name='Classification Report')
            
            # Write confusion matrix
            conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
            
            # Create summary sheet
            summary_data = {
                'Metric': ['ROC AUC Score', 'Optimal Threshold'],
                'Value': [roc_auc, threshold]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Add confusion matrix visualization
            worksheet = writer.sheets['Confusion Matrix']
            worksheet.insert_image('E2', conf_matrix_path)
            
            # Adjust column widths
            for sheet in writer.sheets.values():
                for idx, col in enumerate(class_report_df.columns):
                    sheet.set_column(idx, idx, 15)

    def train_and_evaluate(self):
        """Train model and evaluate performance"""
        # Prepare data
        df_engineered = self.feature_engineering()
        X = df_engineered.drop(['loan_status'], axis=1)
        y = df_engineered['loan_status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline and preprocess data
        pipeline = self.create_pipeline()
        
        # Fit preprocessor
        X_train_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        
        # Train classifier with sample weights to handle class imbalance
        class_weights = dict(zip(
            np.unique(y_train),
            1 / np.bincount(y_train) * len(y_train) / 2
        ))
        sample_weights = np.array([class_weights[label] for label in y_train])
        
        # Train classifier
        pipeline.named_steps['classifier'].fit(
            X_train_transformed, 
            y_train,
            sample_weight=sample_weights
        )
        
        # Get predictions
        y_pred_proba = pipeline.named_steps['classifier'].predict_proba(X_test_transformed)[:, 1]
        
        # Find optimal threshold prioritizing precision
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls)
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Adjust threshold to favor precision
        optimal_threshold = max(optimal_threshold, 0.65)
        
        # Make predictions with optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_test, y_pred, normalize='true'),
            annot=True,
            fmt='.2%',
            cmap='Blues'
        )
        plt.title('Normalized Confusion Matrix - Optimized XGBoost')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix plot
        conf_matrix_path = os.path.join(os.path.dirname(self.output_path), 'confusion_matrix.png')
        plt.savefig(conf_matrix_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save results to Excel
        self.save_to_excel(class_report, conf_mat, roc_auc, optimal_threshold, conf_matrix_path)
        
        # Clean up temporary files
        os.remove(conf_matrix_path)
        
        # Print results
        print("\nResults for Optimized XGBoost model:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nROC AUC Score:", roc_auc)
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"\nResults saved to: {self.output_path}")
        
        return pipeline, optimal_threshold

# Example usage
if __name__ == "__main__":
    model = ImprovedLoanDefaultModel(r"C:\Users\valim\Downloads\Part 2. loan_data_final.csv")
    pipeline, threshold = model.train_and_evaluate()