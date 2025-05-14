from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import shap
import pandas as pd
from io import BytesIO
import base64
from PIL import Image
import os
import matplotlib.pyplot as plt
from flask import request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
# from sleep_disorder_lgbm_model import le_target
from datetime import timedelta
plt.switch_backend('Agg')

app = Flask(__name__)
app.secret_key = 'WakeUpCall_Prototype'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Load model and preprocessing artifacts at startup
try:
    model = joblib.load('sleep_disorder_lgbm_model.pkl')
    scaler = joblib.load('sleep_disorder_scaler.pkl')
    feature_names = joblib.load('sleep_disorder_feature_names.pkl')
    categorical_mappings = joblib.load('categorical_label_mappings.pkl')
    explainer = joblib.load('sleep_disorder_shap_explainer.pkl')
    
    # Debug model information
    print("\n=== MODEL INFORMATION ===")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'classes_'):
        print(f"Model classes: {model.classes_}")
    if hasattr(model, 'n_classes_'):
        print(f"Model n_classes: {model.n_classes_}")
    print("=======================\n")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise e

# Add this function to your code (you can place it near your other utility functions)
def generate_shap_plot(input_data, model, explainer, feature_names):
    """Generate SHAP force plot for the given input data"""
    try:
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_data)
        
        # Get the predicted class
        prediction = model.predict(input_data)[0]
        class_names = ['No Sleep Disorder', 'Insomnia', 'Sleep Apnea']  # Update with your actual class names
        
        # Create the force plot
        plt.figure()
        shap.force_plot(explainer.expected_value[prediction],
                       shap_values[prediction][0],
                       input_data[0],
                       feature_names=feature_names,
                       matplotlib=True,
                       show=False)
        
        # Save plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Encode the image for HTML
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}", class_names[prediction]
    
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None, None

def process_form_data(form_data):
    """Convert form data to the format expected by the model"""
    try:
        print("Starting to process form data")
        # Create a dictionary to hold all processed values
        processed_dict = {}
        # 1. Gender
        gender = form_data.get('gender', 'Male').capitalize()
        processed_dict['Gender'] = categorical_mappings['Gender'].get(gender, 0)
        # 2. Age
        processed_dict['Age'] = float(form_data.get('age', 30))
        # 3. Occupation
        occupation = form_data.get('occupation', 'Accountant')
        processed_dict['Occupation'] = categorical_mappings['Occupation'].get(occupation, 0)
        # 4. Sleep Duration
        processed_dict['Sleep Duration'] = float(form_data.get('sleep_duration', 7))
        # 5. Quality of Sleep
        processed_dict['Quality of Sleep'] = float(form_data.get('sleep_quality', 5))
        # 6. Physical Activity Level
        processed_dict['Physical Activity Level'] = float(form_data.get('physical_activity', 30))
        # 7. Stress Level
        processed_dict['Stress Level'] = float(form_data.get('stress_level', 5))
        # 8. BMI Category
        bmi_cat = form_data.get('BMI Category', 'Normal')
        processed_dict['BMI Category'] = categorical_mappings['BMI Category'].get(bmi_cat, 0)
        # 9. Heart Rate
        processed_dict['Heart Rate'] = float(form_data.get('Heart Rate', 72))
        # 10. Daily Steps
        processed_dict['Daily Steps'] = float(form_data.get('daily_steps', 5000))
        # 11-12. Blood Pressure
        bp = form_data.get('Blood Pressure', '120/80').split('/')
        processed_dict['BP_Systolic'] = float(bp[0])  # Systolic
        processed_dict['BP_Diastolic'] = float(bp[1])  # Diastolic
        
        # Convert to DataFrame for consistent feature names
        processed_data = pd.DataFrame([processed_dict], columns=feature_names)
        
        # Scale the input data
        input_scaled = scaler.transform(processed_data)

        # Print debug information
        print("\n=== PREPROCESSED DATA ===")
        print("Feature Order:", feature_names)
        print("Processed Values:")
        for name, value in zip(feature_names, processed_data.values[0]):
            print(f"  {name}: {value}")
        print("\nFinal Array Shape:", input_scaled.shape)
        print("Final Array:")
        print(input_scaled)
        print("=======================\n")
        
        return input_scaled
        
    except Exception as e:
        print(f"Error processing form data: {e}")
        print("Problematic form data:", form_data)
        return None
    
    
@app.before_request
def check_session():
    print("Current session:", session)

# Routes
# ----- HEADERS -----
@app.route('/', methods=['GET'])
def home():
    session.clear()
    return render_template('index.html')

@app.route('/aboutOSA', methods=['GET'])
def aboutOSA():
    return render_template('aboutOSA.html')

@app.route('/aboutus', methods=['GET'])
def aboutus():
    return render_template('aboutus.html')

# ----- FOOOTER -----
@app.route('/privacypolicy', methods=['GET'])
def privacypolicy():
    return render_template('privacypolicy.html')
@app.route('/contactus', methods=['GET'])
def contactus():
    return render_template('contactus.html')
@app.route('/faq', methods=['GET'])
def faq():
    return render_template('faq.html')

# ----- BODY -----
@app.route('/consent', methods=['GET'])
def consent():
    return render_template('consent.html')

# ----- FORMS -----
@app.route('/demographic', methods=['GET', 'POST'])
def demographic():
    print(f"Rendering demographic.html from: {app.template_folder}")
    if request.method == 'POST':
        print("Form data received:", request.form) 
        session['demographic_data'] = request.form.to_dict()
        return redirect(url_for('sleep'))
    return render_template('demographic.html')

@app.route('/sleep', methods=['GET', 'POST'])
def sleep():
    print(f"Rendering sleep.html from: {app.template_folder}")
    if request.method == 'POST':
        print("Form data received:", request.form)
        session['sleep_data'] = request.form.to_dict()
        return redirect(url_for('lifestyle'))
    return render_template('sleep.html')

@app.route('/lifestyle', methods=['GET', 'POST'])
def lifestyle():
    print(f"Rendering lifestyle.html from: {app.template_folder}")
    if request.method == 'POST':
        print("Form data received:", request.form)
        session['lifestyle_data'] = request.form.to_dict()
        return redirect(url_for('health'))
    return render_template('lifestyle.html')

@app.route('/health', methods=['GET', 'POST'])
def health():
    print(f"Rendering health.html from: {app.template_folder}")
    if request.method == 'POST':
        print("Form data received:", request.form)
        session['health_data'] = request.form.to_dict()
        return redirect(url_for('shap_results'))
    return render_template('health.html')

@app.route('/result')
def result():
    
    return render_template('result.html')
   
# Add this route to show SHAP results
@app.route('/shap_results', methods=['GET'])
def shap_results():
    try:
        print("\n=== STARTING SHAP ANALYSIS ===")
        
        # Verify all required session data exists
        required_sessions = ['demographic_data', 'sleep_data', 'lifestyle_data', 'health_data']
        if not all(key in session for key in required_sessions):
            missing = [key for key in required_sessions if key not in session]
            print(f"Missing session data: {missing}")
            return redirect(url_for('demographic'))

        # Combine all form data
        form_data = {
            **session['demographic_data'],
            **session['sleep_data'],
            **session['lifestyle_data'],
            **session['health_data']
        }
        print("Combined form data:", form_data)

        # Process form data
        processed_df = process_form_data(form_data)
        if processed_df is None:
            return render_template('error.html', message="Invalid input data"), 400

        # Get prediction probabilities
        print("\n=== MODEL PREDICTIONS ===")
        prediction_probs = model.predict(processed_df)
        print("Raw prediction probabilities:", prediction_probs)
        print("Prediction probabilities shape:", prediction_probs.shape)
        
        predicted_class = np.argmax(prediction_probs, axis=1)[0]
        predicted_label = {0: 'Insomnia', 1: 'Sleep Apnea', 2: 'No Sleep Disorder'}.get(predicted_class, 'Unknown')
        print(f"Predicted class: {predicted_class} ({predicted_label})")
        print("Confidence:", np.max(prediction_probs)*100, "%")
        class_names = ['Insomnia', 'Sleep Apnea', 'No Sleep Disorder']
        for i, prob in enumerate(prediction_probs[0]):
            print(f"{class_names[i]}: {prob*100:.2f}%")

        # Prepare class probabilities for display
        class_probabilities = {name: float(prob)*100 for name, prob in zip(class_names, prediction_probs[0])}

        # SHAP Analysis
        print("\n🔍 Generating SHAP explanation...")
        explainer = joblib.load('sleep_disorder_shap_explainer.pkl')
        shap_values = explainer.shap_values(processed_df)

        # Debug SHAP values shape
        print(f"SHAP Values type: {type(shap_values)}")
        print(f"SHAP Values shape: {shap_values.shape}")

        # For multiclass, shap_values is a list of arrays or a 3D array
        if isinstance(shap_values, list):
            # List of arrays format (one per class)
            sample_shap = shap_values[predicted_class][0]  # Get first sample for predicted class
        elif len(np.array(shap_values).shape) == 3:
            # 3D array format (samples × features × classes)
            sample_shap = shap_values[0, :, predicted_class]  # Get first sample for predicted class
        else:
            raise ValueError("Unexpected SHAP values format")
        
        # Get the expected value for the predicted class
        expected_value = explainer.expected_value[predicted_class]

        # Create SHAP force plot
        plt.switch_backend('Agg') # Set backend to non-interactive
        force_plot = shap.force_plot(
            expected_value,
            sample_shap,
            processed_df[0],
            feature_names=feature_names,
            out_names=class_names[predicted_class],
            matplotlib=True,
            show=False,
            plot_cmap='coolwarm'
        )

        # Save force plot to a temporary buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        shap_force_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Create SHAP summary plot (for top features)
        plt.figure()
        shap.summary_plot(
            shap_values[predicted_class],
            processed_df,
            feature_names=feature_names,
            plot_type = 'bar',
            max_display=10,
            show=False
        )
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        shap_summary_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Get top contributing features
        sample_shap = shap_values[predicted_class][0]
        feature_info = list(zip(feature_names, sample_shap, processed_df[0]))
        feature_info.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_info[:10]

        # Calculate precentage for visualization
        total_impact = sum(abs(x[1]) for x in feature_info)
        top_features_with_percent = []

        for name, shap_val, value in top_features:
            percentage = (abs(shap_val) / total_impact) * 100

            # Format feature names and values nicely
            if name == "BMI Category" :
                bmi_mapping = {v: k for k, v in categorical_mappings['BMI Category'].items()}
                formatted_name = f"BMI ({bmi_mapping[int(value)]})"
            elif name == "Sleep Duration":
                formatted_name = f"Sleep Duration ({value:.1f} hrs)"
            elif name in ['BP_Systolic', 'BP_Diastolic']:
                # Find both BP values
                systolic_idx =  feature_names.index('BP_Systolic')
                diastolic_idx = feature_names.index('BP_Diastolic')
                formatted_name = f"Blood Pressure ({processed_df[0][systolic_idx]:.0f}/{processed_df[0][diastolic_idx]:.0f})"
            elif name == "Stress Level":
                formatted_name = f"Stress Level ({value:.0f}/10)"
            elif name == "Physical Activity Level":
                formatted_name = f"Physical Activity Level ({value:.0f}/10)"
            else:
                formatted_name = name.replace('_', ' ').title()

            # Determine if feature increases or decreases risk
            effect = "increases" if shap_val > 0 else "decreases"
            if predicted_class == 2: # for no sleep disorder, reverse the interpretation
                effect = "decreases" if shap_val > 0 else "increases"
            
            top_features_with_percent.append({
                "name": formatted_name,
                "percentage": percentage,
                "effect": effect,
                'impact': abs(shap_val)
            })

        # Prepare data for the template
        result_data = {
            'prediction': predicted_label,
            'confidence': f"{np.max(prediction_probs)*100:.1f}%",
            'class_probabilities': class_probabilities,
            'shap_force_plot': shap_force_plot,
            'shap_summary_plot': shap_summary_plot,
            'top_features': top_features_with_percent,
            'feature_names': [f['name'] for f in top_features_with_percent],
            'feature_percentages': [f['percentage'] for f in top_features_with_percent],
            'feature_effects': [f['effect'] for f in top_features_with_percent]
        }

        return render_template('results.html', **result_data)
    except Exception as e:
        print (f"\n!!! ERROR IN THE SHAP ANALYSIS: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', message="Error in SHAP analysis"), 500

        # # Initialize SHAP explainer with error handling
        # try:
        #     print("\n=== SHAP EXPLAINER ===")
        #     explainer = shap.TreeExplainer(model, model_output="raw")
        #     print("Explainer created successfully")
            
        #     # Calculate SHAP values with robust error handling
        #     shap_values = explainer.shap_values(processed_df)
        #     print(f"Initial SHAP values type: {type(shap_values)}")
            
        #     # Handle different SHAP output formats
        #     if isinstance(shap_values, np.ndarray):
        #         print(f"SHAP values as array with shape: {shap_values.shape}")
        #         if len(shap_values.shape) == 3 and shap_values.shape[0] == 3:
        #             shap_values = [shap_values[i] for i in range(3)]
        #             print("Converted 3D array to list of 2D arrays")
            
        #     if not isinstance(shap_values, list) or len(shap_values) != 3:
        #         print("Unexpected SHAP values format, creating dummy values for visualization")
        #         shap_values = [np.zeros_like(processed_df) for _ in range(3)]
            
        #     # Debug SHAP values
        #     for i, sv in enumerate(shap_values):
        #         print(f"Class {i} SHAP values shape: {sv.shape}")
            
        #     expected_value = explainer.expected_value
        #     if isinstance(expected_value, np.ndarray):
        #         expected_value = expected_value[predicted_class]
        #     print(f"Expected value for class {predicted_class}: {expected_value}")
            
        # except Exception as e:
        #     print(f"Error in SHAP calculation: {e}")
        #     # Create dummy SHAP values if real ones fail
        #     shap_values = [np.zeros_like(processed_df) for _ in range(3)]
        #     expected_value = 0
        #     print("Using dummy SHAP values for visualization")

        # # Generate SHAP explanation panel
        # panel = create_shap_explanation_panel(
        #     shap_values, 
        #     0,  # sample index
        #     predicted_class,
        #     feature_names, 
        #     ['Insomnia', 'Sleep Apnea', 'No Sleep Disorder'], 
        #     processed_df[0],
        #     expected_value
        # )
        
        # # Save panel to buffer
        # panel_buf = BytesIO()
        # panel.savefig(panel_buf, format='png', dpi=300, bbox_inches='tight')
        # plt.close(panel)
        # panel_b64 = base64.b64encode(panel_buf.getvalue()).decode('utf-8')

        # # Generate force plot
        # plt.figure()
        # try:
        #     shap.force_plot(
        #         expected_value, 
        #         shap_values[predicted_class][0], 
        #         processed_df[0],
        #         feature_names=feature_names,
        #         show=False,
        #         matplotlib=True
        #     )
        #     force_buf = BytesIO()
        #     plt.savefig(force_buf, format='png', bbox_inches='tight')
        #     plt.close()
        #     force_b64 = base64.b64encode(force_buf.getvalue()).decode('utf-8')
        # except Exception as e:
        #     print(f"Error generating force plot: {e}")
        #     force_b64 = ""  # Empty if fails

        # # Format probabilities
        # prob_percentages = {
        #     'Insomnia': round(prediction_probs[0][0] * 100, 1),
        #     'Sleep Apnea': round(prediction_probs[0][1] * 100, 1),
        #     'No Sleep Disorder': round(prediction_probs[0][2] * 100, 1)
        # }

        # print("\n=== ANALYSIS COMPLETE ===")
        # return render_template('shap_results.html',
        #                     shap_panel_plot=panel_b64,
        #                     shap_force_plot=force_b64,
        #                     feature_names=feature_names,
        #                     feature_values=processed_df[0],
        #                     shap_values=shap_values[predicted_class][0],
        #                     prediction=predicted_label,
        #                     prediction_proba=prob_percentages,
        #                     form_data=form_data)

    except Exception as e:
        print(f"\n!!! ERROR IN SHAP RESULTS: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', message="An error occurred during analysis"), 500





# def create_shap_explanation_panel(shap_values, sample_idx, predicted_class_idx,
#                                 feature_names, class_names, sample_values,
#                                 expected_value, max_features=5):
#     """
#     Creates a SHAP explanation panel similar to the reference image.
#     """
#     # Get SHAP values for this sample and predicted class
#     sample_shap = shap_values[predicted_class_idx][sample_idx]

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
#                    sample_shap,
#                    sample_values,
#                    feature_names=feature_names,
#                    out_names=class_names[predicted_class_idx],
#                    matplotlib=True,
#                    show=False,
#                    plot_cmap=['#77dd77', '#f99191'],
#                    text_rotation=15)

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
#     for i, (name, shap_val, value) in enumerate(top_features):
#         # Format specific features to match the image
#         if name == 'BMI Category':
#             bmi_mapping = {v: k for k, v in categorical_mappings['BMI Category'].items()}
#             formatted_name = f"BMI ({bmi_mapping[int(value)]})"
#         elif name == 'Sleep Duration':
#             formatted_name = f"Sleep Duration (<{int(value)}hrs)"
#         elif name in ['BP_Systolic', 'BP_Diastolic']:
#             # Find both BP values in the sample
#             systolic_idx = feature_names.index('BP_Systolic')
#             diastolic_idx = feature_names.index('BP_Diastolic')
#             formatted_name = f"BP ({int(sample_values[systolic_idx])}/{int(sample_values[diastolic_idx])})"
#         elif name == 'Stress Level':
#             formatted_name = f"Stress Level ({int(value)}/10)"
#         else:
#             formatted_name = name

#         feature_details.append(f"{formatted_name} [{int(abs(shap_val)*100)}]")

#     ax4.text(0, 0.8, "\n".join(feature_details), fontsize=12, va='top')

#     plt.suptitle(f"SHAP Explanation for {class_names[predicted_class_idx]} Prediction",
#                 fontsize=14, y=1.05)
#     plt.tight_layout()
#     return fig
    

if __name__ == '__main__':
    app.run(port=3100, debug=True)