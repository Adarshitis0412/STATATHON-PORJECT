import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback # For detailed error logging

# --- Import your custom classes ---
from data_cleaning import DataCleaner
from data_analysis import DataAnalyzer
from ai_report import GoogleAIReport

# --- Initialize Flask App ---
app = Flask(__name__)
# --- FIX: Make CORS more specific to allow requests from your frontend's origin ---
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allows all origins for simplicity, can be restricted.

# --- Configuration ---
# Create directories to store uploaded files and generated outputs if they don't exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'analysis_outputs'
CLEANED_DATA_PATH = os.path.join(UPLOAD_FOLDER, 'cleaned_data.csv')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Global variables to store state (simple approach for this project) ---
# In a larger app, you might use a database or more robust session management
# For the statathon, this will work perfectly.
latest_cleaning_log = []
latest_statistics = None

# --- API Endpoints ---

@app.route('/api/upload', methods=['POST'])
def upload_and_clean_data():
    """
    Endpoint to upload a CSV, clean it, and store the results.
    This combines the upload and clean steps for simplicity.
    """
    global latest_cleaning_log # Use the global variable to store the log

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            # Save the uploaded file
            raw_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(raw_filepath)

            # --- Data Cleaning Step ---
            df = pd.read_csv(raw_filepath)
            cleaner = DataCleaner(df)
            
            # The clean_data method saves the cleaned file and returns the log
            cleaned_df, log = cleaner.clean_data(export_path=CLEANED_DATA_PATH)
            
            latest_cleaning_log = log # Store the log for the report step

            # Return the log to the frontend
            return jsonify({"cleaningLog": log})

        except Exception as e:
            traceback.print_exc() # Print detailed error to the console
            return jsonify({"error": f"An error occurred during cleaning: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV."}), 400


@app.route('/api/analyze', methods=['GET'])
def analyze_data():
    """
    Endpoint to run data analysis on the cleaned data.
    """
    global latest_statistics # Use the global variable to store stats

    if not os.path.exists(CLEANED_DATA_PATH):
        return jsonify({"error": "Cleaned data not found. Please upload and clean a file first."}), 404
    
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        # Pass the output directory to the analyzer
        analyzer = DataAnalyzer(df, output_dir=OUTPUT_FOLDER)
        
        # Run the full analysis
        analysis_results, _ = analyzer.run_full_analysis()

        # --- Prepare the response ---
        # Convert statistics DataFrame to JSON
        stats_json = analysis_results["statistics"].reset_index().to_json(orient='split')
        latest_statistics = analysis_results["statistics"] # Save for the report step

        # Create URLs for the generated plot images
        # The frontend will use these URLs to display the images
        base_url = request.host_url
        visualizations = [
            {"title": "Correlation Heatmap", "url": f"{base_url}outputs/{os.path.basename(analysis_results['correlation_heatmap'])}"},
            {"title": "Missing Values Heatmap", "url": f"{base_url}outputs/{os.path.basename(analysis_results['missing_value_heatmap'])}"},
        ]
        # Add distribution plots
        for dist_plot_path in analysis_results.get("distributions", []):
            title = f"{os.path.basename(dist_plot_path).replace('dist_', '').replace('.png', '').capitalize()} Distribution"
            visualizations.append({
                "title": title,
                "url": f"{base_url}outputs/{os.path.basename(dist_plot_path)}"
            })

        return jsonify({
            "statistics": stats_json,
            "visualizations": visualizations
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500


@app.route('/api/report', methods=['GET'])
def generate_ai_report():
    """
    Endpoint to generate the AI report using the cleaning log and statistics.
    """
    if latest_statistics is None or not latest_cleaning_log:
        return jsonify({"error": "Statistics or cleaning log not available. Run previous steps first."}), 404
    
    try:
        reporter = GoogleAIReport()
        report_text = reporter.generate_report(stats_df=latest_statistics, logs=latest_cleaning_log)
        
        return jsonify({"aiReport": report_text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred generating the report: {str(e)}"}), 500


# --- Static File Route ---
@app.route('/outputs/<filename>')
def serve_output_image(filename):
    """
    This route allows the frontend to access the generated plot images.
    """
    return send_from_directory(OUTPUT_FOLDER, filename)


# --- Run the App ---
if __name__ == '__main__':
    # Use port 5001 to avoid conflicts with the React dev server (often on 3000 or 5000)
    app.run(debug=True, port=5001)
