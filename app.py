"""
app.py
Flask web server and REST API for House Price Prediction Embedded MLP.
Provides real-time price estimation, feature contribution breakdown,
and model performance dashboard.
"""

import os
import sys
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, render_template, send_from_directory
from preprocess import HousingDataPreprocessor, AMENITIES_LIST, CONTINUOUS_COLS, MONTHS_LIST
from model import EmbeddedMLP
from geo_data import GEO_HIERARCHY

app = Flask(__name__, template_folder='templates', static_folder='static')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Global model state
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocessor = None
model = None
metadata = {}
metrics = {}

def load_system():
    global preprocessor, model, metadata, metrics
    
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    prep_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    model_path = os.path.join(MODELS_DIR, 'best_model.pt')
    metrics_path = os.path.join(REPORTS_DIR, 'metrics_summary.json')
    
    if os.path.exists(meta_path) and os.path.exists(prep_path) and os.path.exists(model_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        preprocessor = HousingDataPreprocessor.load(prep_path)
        
        model = EmbeddedMLP(
            cat_dims=metadata['cat_dims'],
            emb_dims=metadata['emb_dims'],
            num_continuous=len(metadata['continuous_cols']),
            hidden_units=[256, 128, 64, 32]
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model and preprocessor loaded successfully.")
        
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

load_system()

@app.route('/')
def index():
    return render_template(
        'index.html',
        geo_hierarchy=GEO_HIERARCHY,
        states=list(GEO_HIERARCHY.keys()),
        months=MONTHS_LIST,
        property_types=metadata.get('classes', {}).get('property_type', []),
        furnishing_options=metadata.get('classes', {}).get('furnishing_status', []),
        amenities=AMENITIES_LIST,
        metrics=metrics
    )

@app.route('/api/geo-hierarchy', methods=['GET'])
def get_geo_hierarchy():
    return jsonify(GEO_HIERARCHY)

def format_indian_currency(val):
    val = round(val)
    s = str(abs(val))
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ','.join(groups) + ',' + last3
    formatted = f"₹{'-' if val < 0 else ''}{grouped}"

    abs_v = abs(val)
    if abs_v >= 10000000:
        short = f"₹{abs_v / 10000000:.2f} Crore"
    elif abs_v >= 100000:
        short = f"₹{abs_v / 100000:.2f} Lakhs"
    elif abs_v >= 1000:
        short = f"₹{abs_v / 1000:.1f} K"
    else:
        short = formatted
    return formatted, short

@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        area = float(data.get('area_sqft') or data.get('area') or 2000)
        bedrooms = float(data.get('bedrooms', 3))
        bathrooms = float(data.get('bathrooms', 2))
        
        # Hierarchical geographic parameters
        state = data.get('state', 'Telangana')
        city = data.get('city', 'Hyderabad')
        locality = data.get('locality', 'HITEC City & Cyber Towers')
        
        prop_type = data.get('property_type', 'Apartment / Gated High-Rise Flat')
        furnishing = data.get('furnishing_status', 'Semi-Furnished')
        year_built = float(data.get('year_built', 2024))
        month_built = str(data.get('month_built', 'October'))
        distance = float(data.get('distance_to_city_center_km', 8.5))
        school_rating = float(data.get('school_rating', 8.0))
        safety_index = float(data.get('safety_index', 80.0))
        garage = float(data.get('garage_spaces', 2))
        amenities = data.get('amenities', [])
        
        raw_input = {
            'state': state,
            'city': city,
            'locality': locality,
            'property_type': prop_type,
            'furnishing_status': furnishing,
            'month_built': month_built,
            'area_sqft': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'garage_spaces': garage,
            'year_built': year_built,
            'distance_to_city_center_km': distance,
            'school_rating': school_rating,
            'safety_index': safety_index,
            'amenities': amenities
        }
        
        cat_tensor, cont_tensor = preprocessor.transform_single(raw_input)
        x_cat = torch.tensor(cat_tensor, dtype=torch.long).to(device)
        x_cont = torch.tensor(cont_tensor, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(x_cat, x_cont).cpu().numpy()
            
        pred_price = float(preprocessor.inverse_transform_target(pred_scaled)[0, 0])
        pred_price = max(80000.0, pred_price)
        
        # Calculate confidence interval from model RMSE (~1 standard error)
        rmse_val = metrics.get('RMSE', 78000)
        price_low = max(50000.0, pred_price - (0.85 * rmse_val))
        price_high = pred_price + (0.85 * rmse_val)
        price_per_sqft = pred_price / area if area > 0 else 0
        
        fmt_price, short_price = format_indian_currency(pred_price)
        fmt_low, short_low = format_indian_currency(price_low)
        fmt_high, short_high = format_indian_currency(price_high)
        fmt_sqft, _ = format_indian_currency(price_per_sqft)

        return jsonify({
            'success': True,
            'predicted_price': round(pred_price, 2),
            'formatted_price': fmt_price,
            'short_price': short_price,
            'currency': 'INR',
            'currency_symbol': '₹',
            'price_range': {
                'low': round(price_low, 2),
                'high': round(price_high, 2),
                'formatted': f"{fmt_low} - {fmt_high}",
                'short_formatted': f"{short_low} - {short_high}"
            },
            'price_per_sqft': round(price_per_sqft, 2),
            'formatted_sqft': f"{fmt_sqft}/sq.ft",
            'input_summary': {
                'area': f"{area:,.0f} sq.ft",
                'beds_baths': f"{int(bedrooms)} bed, {bathrooms:g} bath",
                'state': state,
                'city': city,
                'locality': locality,
                'location': f"{locality}, {city}, {state}",
                'property_type': prop_type,
                'year_built': int(year_built),
                'month_built': month_built,
                'amenities_count': len(amenities)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

@app.route('/reports/<path:filename>')
def serve_report_image(filename):
    return send_from_directory(REPORTS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
