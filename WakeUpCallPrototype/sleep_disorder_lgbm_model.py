import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
import shap
import json
from flask import jsonify
from io import BytesIO
from PIL import Image

# Load the dataset
file_path = 'Sleep_health_and_lifestyle_dataset.csv'
df = pd.read_csv(file_path)

# Preprocess the data
df['Sleep Disorder'] = df['Sleep Disorder'].astype(str)

df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None').astype(str)

# Label encode the target variable
le_target = LabelEncoder()
df['Sleep Disorder Label'] = le_target.fit_transform(df['Sleep Disorder'])

# Print the target label mapping
print("Target Label Mapping:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))
# Target Label Mapping: {'Insomnia': np.int64(0), 'Sleep Apnea': np.int64(1),'nan': np.int64(2)}

# Extract blood pressure values
df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

# Drop unnecessary columns
df = df.drop(['Person ID', 'Blood Pressure', 'Sleep Disorder'], axis=1)

# Encode and map categorical columns
label_encoders = {}
label_mappings = {}

# Encode categorical columns
categorical_cols = ['Gender', 'Occupation', 'BMI Category']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Print the categorical column mappings
print("\nCategorical Column Mappings:")
for col, mapping in label_mappings.items():
    print(f"{col}: {mapping}")
# Gender: {'Female': np.int64(0), 'Male': np.int64(1)}
# Occupation: {'Accountant': np.int64(0), 'Doctor': np.int64(1),
#  'Engineer': np.int64(2), 'Lawyer': np.int64(3), 'Manager': np.int64(4),
#  'Nurse': np.int64(5), 'Sales Representative': np.int64(6), 'Salesperson': np.int64(7), 'Scientist': np.int64(8), 
#  'Software Engineer': np.int64(9), 'Teacher': np.int64(10)}
# BMI Category: {'Normal': np.int64(0), 'Normal Weight': np.int64(1), 'Obese': np.int64(2), 'Overweight': np.int64(3)}

# Remove outliers
num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Stress Level', 'Heart Rate', 'Daily Steps', 'BP_Systolic', 'BP_Diastolic']
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Prepare data
X = df.drop('Sleep Disorder Label', axis=1)
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'sleep_disorder_feature_names.pkl')
y = df['Sleep Disorder Label']

# Print the target values
print("\nTarget Values (y):")
print(y)

# Print the class distribution
print("\nClass Distribution:")
print(y.value_counts(normalize=True))

# Train the model
print("\nTraining the model...")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and label encoder
joblib.dump(scaler, 'sleep_disorder_scaler.pkl')
joblib.dump(le_target, 'sleep_disorder_label_encoder.pkl')

# Train the model
train_data = lgb.Dataset(X_train_scaled, label=y_train)
test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

# Define the parameters
params = {
    'objective': 'multiclass',
    'num_class': len(le_target.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42
}

# Train the model
model = lgb.train(params,
                 train_data,
                 num_boost_round=1000,
                 valid_sets=[test_data],
                 callbacks=[
                     lgb.early_stopping(stopping_rounds=50),
                     lgb.log_evaluation(period=10)
                 ])

# Save model and label mappings
joblib.dump(model, 'sleep_disorder_multiclass_lgbm_model.pkl')
joblib.dump(label_mappings, 'categorical_label_mappings.pkl')
print("\nModel and categorical label mappings saved.")

# Load saved feature names
feature_names = joblib.load('sleep_disorder_feature_names.pkl')
print("Feature order used for training:")
print(feature_names)
# Feature order used for training:
# ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
#   'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'BP_Systolic', 'BP_Diastolic']



# Evaluate the model
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le_target.classes_)
cm = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(report)


# SHAP Analysis
print("\n🔍 Generating SHAP explanation...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Save SHAP explainer
joblib.dump(explainer, 'sleep_disorder_shap_explainer.pkl')

def create_shap_explanation_panel(shap_values, sample_idx, predicted_class_idx,
                                 feature_names, class_names, sample_values,
                                 expected_value, max_features=5):
    """
    Creates a SHAP explanation panel similar to the reference image.
    """
    # Get SHAP values for this sample and predicted class
    sample_shap = shap_values[sample_idx, :, predicted_class_idx]

    # Combine with feature names and values
    feature_info = list(zip(feature_names, sample_shap, sample_values))

    # Sort by absolute SHAP value (most impactful first)
    feature_info.sort(key=lambda x: abs(x[1]), reverse=True)

    # Take top features
    top_features = feature_info[:max_features]

    # Calculate percentages for visualization
    total_impact = sum(abs(x[1]) for x in feature_info)
    percentages = [(abs(x[1])/total_impact)*100 for x in top_features]

    # Create the plot with a 3x2 grid
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2)

    # Create subplots
    ax1 = fig.add_subplot(gs[0:2, 0])  # Diagram area
    ax2 = fig.add_subplot(gs[0, 1])    # Code area
    ax3 = fig.add_subplot(gs[1, 1])    # Top factors area
    ax4 = fig.add_subplot(gs[2, :])    # Feature details area

    # Remove axes from code area
    ax2.axis('off')
    ax2.text(0.5, 0.5, "Code", ha='center', va='center', fontsize=12)

    # Create force plot as a separate figure and embed it
    force_fig = plt.figure()
    shap.force_plot(expected_value,
                   sample_shap,
                   sample_values,
                   feature_names=feature_names,
                   out_names=class_names[predicted_class_idx],
                   matplotlib=True,
                   show=False,
                   plot_cmap=['#77dd77', '#f99191'],
                   text_rotation=15)

    # Save the force plot to a temporary buffer
    from io import BytesIO
    buf = BytesIO()
    force_fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(force_fig)

    # Display the force plot in our panel
    from PIL import Image
    img = Image.open(buf)
    ax1.imshow(img)
    ax1.axis('off')

    # Top contributing factors (percentage bars)
    ax3.set_title('Top Contributing Factors', fontsize=12)
    y_pos = np.arange(len(top_features))
    ax3.barh(y_pos, percentages, color='#1f77b4')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"{p:.0f}%" for p in percentages])
    ax3.invert_yaxis()
    ax3.set_xlim(0, 100)

    # Feature details - formatted like in the image
    ax4.axis('off')
    feature_details = []
    for i, (name, shap_val, value) in enumerate(top_features):
        # Format specific features to match the image
        if name == 'BMI Category':
            bmi_mapping = {v: k for k, v in label_mappings['BMI Category'].items()}
            formatted_name = f"BMI ({bmi_mapping[int(value)]})"
        elif name == 'Sleep Duration':
            formatted_name = f"Sleep Duration (<{int(value)}hrs)"
        elif name in ['BP_Systolic', 'BP_Diastolic']:
            # Find both BP values in the sample
            systolic_idx = feature_names.index('BP_Systolic')
            diastolic_idx = feature_names.index('BP_Diastolic')
            formatted_name = f"BP ({int(sample_values[systolic_idx])}/{int(sample_values[diastolic_idx])})"
        elif name == 'Stress Level':
            formatted_name = f"Stress Level ({int(value)}/10)"
        else:
            formatted_name = name

        feature_details.append(f"{formatted_name} [{int(abs(shap_val)*100)}]")

    ax4.text(0, 0.8, "\n".join(feature_details), fontsize=12, va='top')

    plt.suptitle(f"SHAP Explanation for {class_names[predicted_class_idx]} Prediction",
                fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

# SHAP Visualizations
import matplotlib.pyplot as plt # Re-import pyplot to ensure plt is the module
plt.style.use('ggplot')
shap.initjs()

# # Example prediction with SHAP explanation panel
# print("\nMaking example prediction with SHAP explanation panel...")
# new_features = [1, 27, 9, 6.1, 6, 42, 6, 2, 77, 4200, 126, 83]    # Example values
# new_data = pd.DataFrame([new_features], columns=feature_names)
# new_data_scaled = scaler.transform(new_data)

# # Get prediction
# predicted_probs = model.predict(new_data_scaled)
# predicted_class = np.argmax(predicted_probs, axis=1)
# predicted_label = le_target.inverse_transform(predicted_class)

# # Generate SHAP values for the new prediction
# sample_shap_values = explainer.shap_values(new_data_scaled)

# # Create explanation panel
# print(f"\nSHAP Explanation Panel for New Prediction (Predicted: {predicted_label[0]})")
# panel = create_shap_explanation_panel(sample_shap_values, 0, predicted_class[0],
#                                     feature_names, le_target.classes_, new_data_scaled[0],
#                                     explainer.expected_value[predicted_class[0]])
# panel.savefig('shap_explanation_panel_new_prediction.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Print prediction results
# print("\n🧠 Prediction Result:")
# print("Predicted probabilities:", predicted_probs)
# print("Most likely sleep disorder:", predicted_label[0])
# print("Confidence:", np.max(predicted_probs)*100, "%")

# # Print all possible disorders with their probabilities
# for i, (prob, class_name) in enumerate(zip(predicted_probs[0], le_target.classes_)):
#     print(f"{class_name}: {prob*100:.1f}%")



# def generate_shap_values(input_data, model):
#     """
#     Generates SHAP values for the given input data using the provided model.
    
#     Parameters:
#     - input_data: pd.DataFrame with feature names for which SHAP values are to be computed.
#     - model: The trained model used for SHAP analysis.
    
#     Returns:
#     - explainer: The SHAP explainer object.
#     - shap_values: The computed SHAP values (list or array depending on model).
#     """
#     # Check that input_data is a DataFrame with feature names to avoid warning
#     if not isinstance(input_data, pd.DataFrame):
#         raise ValueError("input_data must be a pandas DataFrame with feature names.")

#     print("\nPerforming SHAP analysis...")
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(input_data)

#     # Save SHAP explainer
#     joblib.dump(explainer, 'sleep_disorder_shap_explainer.pkl')
    
#     return explainer, shap_values


# def generate_shap_summary_plot(shap_values, sample_idx, predicted_class_idx,
#                                 feature_names, class_names, sample_values,
#                                 expected_value, max_features=5):
#     """
#     Creates a SHAP explanation panel similar to the reference image.
    
#     Parameters:
#     - shap_values: The SHAP values (list or np.array) returned by the explainer.
#     - sample_idx: Index of the sample to explain.
#     - predicted_class_idx: Index of the predicted class.
#     - feature_names: Names of the features.
#     - class_names: Names of the classes.
#     - sample_values: Values of the features for the sample.
#     - expected_value: The expected value for the model output (float or array).
#     - max_features: Maximum number of features to display.
    
#     Returns:
#     - fig: The generated figure containing the SHAP explanation panel.
#     """
#     # Handle shap_values if it is a list (multi-class classification)
#     # Each element corresponds to the shap values for one class.
#     if isinstance(shap_values, list):
#         # shap_values[class][sample, feature]
#         sample_shap = shap_values[predicted_class_idx][sample_idx, :]
#     else:
#         # shap_values array shaped (samples, features, classes)
#         sample_shap = shap_values[sample_idx, :, predicted_class_idx]

#     # Combine with feature names and values
#     feature_info = list(zip(feature_names, sample_shap, sample_values))

#     # Sort by absolute SHAP value (most impactful first)
#     feature_info.sort(key=lambda x: abs(x[1]), reverse=True)

#     # Take top features
#     top_features = feature_info[:max_features]

#     # Calculate percentages for visualization
#     total_impact = sum(abs(x[1]) for x in feature_info)
#     percentages = [(abs(x[1])/total_impact)*100 for x in top_features]

#     # Create the plot with a 3x2 grid
#     fig = plt.figure(figsize=(12, 8))
#     gs = fig.add_gridspec(3, 2)

#     # Create subplots
#     ax1 = fig.add_subplot(gs[0:2, 0])  # Diagram area
#     ax2 = fig.add_subplot(gs[0, 1])    # Code area
#     ax3 = fig.add_subplot(gs[1, 1])    # Top factors area
#     ax4 = fig.add_subplot(gs[2, :])    # Feature details area

#     # Remove axes from code area
#     ax2.axis('off')
#     ax2.text(0.5, 0.5, "Code", ha='center', va='center', fontsize=12)

#     # Create force plot as a separate figure and embed it
#     force_fig = plt.figure()
#     shap.force_plot(expected_value,
#                      sample_shap,
#                      sample_values,
#                      feature_names=feature_names,
#                      out_names=class_names[predicted_class_idx],
#                      matplotlib=True,
#                      show=False,
#                      plot_cmap=['#77dd77', '#f99191'],
#                      text_rotation=15)

#     # Save the force plot to a temporary buffer
#     buf = BytesIO()
#     force_fig.savefig(buf, format='png', bbox_inches='tight')
#     buf.seek(0)
#     plt.close(force_fig)

#     # Display the force plot in our panel
#     img = Image.open(buf)
#     ax1.imshow(img)
#     ax1.axis('off')

#     # Top contributing factors (percentage bars)
#     ax3.set_title('Top Contributing Factors', fontsize=12)
#     y_pos = np.arange(len(top_features))
#     ax3.barh(y_pos, percentages, color='#1f77b4')
#     ax3.set_yticks(y_pos)
#     ax3.set_yticklabels([f"{p:.0f}%" for p in percentages])
#     ax3.invert_yaxis()
#     ax3.set_xlim(0, 100)

#     # Feature details - formatted like in the image
#     ax4.axis('off')
#     feature_details = []

#     # For label_mappings, it must be defined in the scope where this is called
#     # Otherwise, just fallback to raw names/values
#     for i, (name, shap_val, value) in enumerate(top_features):
#         if 'label_mappings' in globals():
#             if name == 'BMI Category' and 'BMI Category' in label_mappings:
#                 bmi_mapping = {v: k for k, v in label_mappings['BMI Category'].items()}
#                 formatted_name = f"BMI ({bmi_mapping.get(int(value), 'Unknown')})"
#             elif name == 'Sleep Duration':
#                 formatted_name = f"Sleep Duration (<{int(value)}hrs)"
#             elif name in ['BP_Systolic', 'BP_Diastolic']:
#                 if 'BP_Systolic' in feature_names and 'BP_Diastolic' in feature_names:
#                     systolic_idx = feature_names.index('BP_Systolic')
#                     diastolic_idx = feature_names.index('BP_Diastolic')
#                     formatted_name = f"BP ({int(sample_values[systolic_idx])}/{int(sample_values[diastolic_idx])})"
#                 else:
#                     formatted_name = name
#             elif name == 'Stress Level':
#                 formatted_name = f"Stress Level ({int(value)}/10)"
#             else:
#                 formatted_name = name
#         else:
#             formatted_name = name

#         feature_details.append(f"{formatted_name} [{int(abs(shap_val)*100)}]")

#     ax4.text(0, 0.8, "\n".join(feature_details), fontsize=12, va='top')

#     plt.suptitle(f"SHAP Explanation for {class_names[predicted_class_idx]} Prediction",
#                 fontsize=14, y=1.05)
#     plt.tight_layout()
#     return fig



print("Feature names used for training:", feature_names)
print(model.params)  # Should show 'num_class': 3


# Predict and explain function
# def predict_and_explain(user_input_dict):
#     """
#     Predicts the sleep disorder from user input and shows SHAP explanation.

#     Parameters:
#         user_input_dict (dict): A dictionary with keys matching the feature names and raw user inputs as values.
#     """
#     # Load encoders, scalers, and mappings
#     le_target = joblib.load('sleep_disorder_label_encoder.pkl')
#     scaler = joblib.load('sleep_disorder_scaler.pkl')
#     model = joblib.load('sleep_disorder_multiclass_lgbm_model.pkl')
#     feature_names = joblib.load('sleep_disorder_feature_names.pkl')
#     label_mappings = joblib.load('categorical_label_mappings.pkl')

#     # Create a DataFrame from the input
#     user_df = pd.DataFrame([user_input_dict])

#     # Encode categorical inputs using saved mappings
#     for col in ['Gender', 'Occupation', 'BMI Category']:
#         if col in label_mappings:
#             mapping = label_mappings[col]
#             user_df[col] = user_df[col].map(mapping)
#             if user_df[col].isnull().any():
#                 raise ValueError(f"Invalid value for {col}. Available: {list(mapping.keys())}")

#     # Reorder columns to match training
#     user_df = user_df[feature_names]

#     # Scale input
#     user_scaled = scaler.transform(user_df)

#     # Predict
#     pred_proba = model.predict(user_scaled)
#     pred_label = np.argmax(pred_proba)
#     pred_class = le_target.inverse_transform([pred_label])[0]
#     confidence = pred_proba[0][pred_label] * 100

#     print("\n🧠 Prediction Result:")
#     print(f"Most likely sleep disorder: {pred_class}")
#     print(f"Confidence: {confidence:.2f}%")
#     print("\n📊 Class probabilities:")
#     for i, class_name in enumerate(le_target.classes_):
#         print(f"{class_name}: {pred_proba[0][i]*100:.1f}%")

#     # Explain prediction using SHAP
#     print("\n🔍 Generating SHAP explanation...")
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(user_scaled)
    
#     # Visualize SHAP force plot
#     # Use SHAP values to explain the prediction
#     shap_values_single = shap_values[pred_label][0]

#     # Create a SHAP explanation object for one row
#     explanation = shap.Explanation(
#         values=shap_values_single,
#         base_values=explainer.expected_value[pred_label],
#         data=user_df.iloc[0],
#         feature_names=feature_names
#     )

#     # Plot the SHAP force plot
#     shap.plots.bar(explanation)

# # Example user input
# user_input_example = {
#     'Gender': 'Male',
#     'Age': 28,
#     'Occupation': 'Engineer',
#     'Sleep Duration': 5.5,
#     'Quality of Sleep': 4,
#     'Physical Activity Level': 3,
#     'Stress Level': 7,
#     'BMI Category': 'Overweight',
#     'Heart Rate': 82,
#     'Daily Steps': 4000,
#     'BP_Systolic': 130,
#     'BP_Diastolic': 85
# }

# predict_and_explain(user_input_example)