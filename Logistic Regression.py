import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# Generate more balanced sample loan data
np.random.seed(42)
n_samples = 1000

# Create synthetic features
income = np.random.normal(50000, 20000, n_samples)
income = np.abs(income)
debt_ratio = np.random.uniform(0, 0.6, n_samples)

# Create target variable with stronger signal for default
# Adjust parameters to create more default cases
log_odds = -2 + (income/15000)*0.3 - debt_ratio*10  # Stronger coefficients
prob_default = 1 / (1 + np.exp(-log_odds))
default = np.random.binomial(1, prob_default)

print(f"Class distribution:")
print(f"Non-default (0): {(default == 0).sum()} samples ({((default == 0).sum()/n_samples)*100:.1f}%)")
print(f"Default (1): {(default == 1).sum()} samples ({((default == 1).sum()/n_samples)*100:.1f}%)")

# Create DataFrame
df = pd.DataFrame({
    'income': income,
    'debt_ratio': debt_ratio,
    'default': default
})

# If we still have too few default cases, let's create a more balanced dataset
if (default == 1).sum() < 50:
    print("\nAdjusting data to create more balanced classes...")
    # Create more default cases by focusing on high-risk individuals
    n_defaults_needed = 150
    current_defaults = (default == 1).sum()
    
    # Generate additional high-risk samples
    n_additional = n_defaults_needed - current_defaults
    high_risk_income = np.random.normal(30000, 10000, n_additional)
    high_risk_debt = np.random.uniform(0.4, 0.6, n_additional)
    
    # These should definitely default
    high_risk_default = np.ones(n_additional)
    
    # Combine with original data
    income = np.concatenate([income, high_risk_income])
    debt_ratio = np.concatenate([debt_ratio, high_risk_debt])
    default = np.concatenate([default, high_risk_default])
    
    df = pd.DataFrame({
        'income': income,
        'debt_ratio': debt_ratio,
        'default': default
    })
    
    print(f"New class distribution:")
    print(f"Non-default (0): {(default == 0).sum()} samples ({((default == 0).sum()/len(default))*100:.1f}%)")
    print(f"Default (1): {(default == 1).sum()} samples ({((default == 1).sum()/len(default))*100:.1f}%)")

# Prepare features and target
X = df[['income', 'debt_ratio']]
y = df['default']

# Split the data with stratification to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model with class weights
# This helps handle imbalanced data
model = LogisticRegression(
    random_state=42, 
    class_weight='balanced',  # Automatically adjust weights based on class frequency
    max_iter=1000  # Increase iterations for convergence
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Model evaluation with zero_division parameter to handle warnings
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 No Default  Default")
print(f"Actual No Default     {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"Actual Default        {cm[1,0]:6d}      {cm[1,1]:6d}")

# Classification Report with zero_division parameter to suppress warnings
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Default', 'Default'],
                          zero_division=0))  # This handles the warning

# AUC-ROC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {roc_auc:.4f}")

# Model coefficients
print("\n" + "="*50)
print("MODEL COEFFICIENTS")
print("="*50)
coefficients = pd.DataFrame({
    'Feature': ['income', 'debt_ratio'],
    'Coefficient': model.coef_[0]
})
print(coefficients)
print(f"Intercept: {model.intercept_[0]:.4f}")

# Additional metrics for imbalanced classification
from sklearn.metrics import precision_recall_curve, average_precision_score

# Precision-Recall curve (better for imbalanced data)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)
print(f"\nAverage Precision Score: {avg_precision:.4f}")

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Feature distributions by default status
axes[0, 0].hist([df[df['default']==0]['income'], 
                 df[df['default']==1]['income']], 
                label=['No Default', 'Default'], alpha=0.7, bins=30)
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Income Distribution by Default Status')
axes[0, 0].legend()

axes[0, 1].hist([df[df['default']==0]['debt_ratio'], 
                 df[df['default']==1]['debt_ratio']], 
                label=['No Default', 'Default'], alpha=0.7, bins=30)
axes[0, 1].set_xlabel('Debt Ratio')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Debt Ratio Distribution by Default Status')
axes[0, 1].legend()

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[0, 2].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curve')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curve (better for imbalanced data)
axes[1, 0].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Decision Boundary
x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1, 1].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu_r')
scatter = axes[1, 1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
                           c=y_test, cmap='RdYlBu_r', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Income (scaled)')
axes[1, 1].set_ylabel('Debt Ratio (scaled)')
axes[1, 1].set_title('Decision Boundary')
plt.colorbar(scatter, ax=axes[1, 1])

# Plot 5: Probability Distribution
axes[1, 2].hist(y_pred_proba[y_test==0], bins=30, alpha=0.7, label='Actual No Default', density=True)
axes[1, 2].hist(y_pred_proba[y_test==1], bins=30, alpha=0.7, label='Actual Default', density=True)
axes[1, 2].set_xlabel('Predicted Probability of Default')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('Predicted Probability Distribution')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# Function to make predictions for new applicants
def predict_loan_default(income, debt_ratio, model, scaler):
    """
    Predict loan default probability for a new applicant
    """
    # Create feature array
    applicant = np.array([[income, debt_ratio]])
    
    # Scale features
    applicant_scaled = scaler.transform(applicant)
    
    # Predict probability
    prob_default = model.predict_proba(applicant_scaled)[0, 1]
    prediction = model.predict(applicant_scaled)[0]
    
    return prob_default, prediction

# Example predictions for new applicants
print("\n" + "="*50)
print("PREDICTIONS FOR NEW APPLICANTS")
print("="*50)

test_applicants = [
    (80000, 0.15),  # High income, low debt
    (30000, 0.45),  # Low income, high debt
    (50000, 0.30),  # Medium income, medium debt
    (25000, 0.50),  # Very high risk
    (100000, 0.10), # Very low risk
]

for income, debt_ratio in test_applicants:
    prob, pred = predict_loan_default(income, debt_ratio, model, scaler)
    default_status = "Default" if pred == 1 else "No Default"
    risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    print(f"Income: ${income:,.0f}, Debt Ratio: {debt_ratio:.2f} -> "
          f"Prediction: {default_status} (Risk: {risk_level}, Probability: {prob:.3f})")
