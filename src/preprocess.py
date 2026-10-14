"""
preprocess.py
Data cleaning, feature transformation, missing value imputation,
and encoding pipeline for House Price Prediction Embedded MLP.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

AMENITIES_LIST = [
    'Swimming Pool', 'Gym/Fitness Center', 'Smart Home Automation',
    'EV Charging Station', 'Solar Panel Array', 'Private Garden',
    'Rooftop Terrace', 'Clubhouse Access', '24/7 Security Concierge'
]

MONTHS_LIST = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

CATEGORICAL_COLS = ['state', 'city', 'locality', 'property_type', 'furnishing_status', 'month_built']

CONTINUOUS_COLS = [
    'area_sqft', 'bedrooms', 'bathrooms', 'garage_spaces',
    'property_age', 'distance_to_city_center_km', 'school_rating',
    'safety_index', 'bed_to_bath_ratio', 'amenity_score'
] + [f"has_{a.lower().replace('/', '_').replace(' ', '_')}" for a in AMENITIES_LIST]

class HousingDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler_continuous = StandardScaler()
        self.scaler_target = StandardScaler()
        self.imputation_values = {}
        self.cat_dims = {}
        self.emb_dims = {}
        
    def fit_transform(self, df: pd.DataFrame, target_col='price'):
        df_clean = df.copy()
        
        # 1. Missing Value Imputation with Pandas
        # Numerical missing values
        median_year = df_clean['year_built'].median()
        self.imputation_values['year_built'] = median_year
        df_clean['year_built'] = df_clean['year_built'].fillna(median_year)
        
        mode_month = df_clean['month_built'].mode()[0] if 'month_built' in df_clean and not df_clean['month_built'].dropna().empty else 'October'
        self.imputation_values['month_built'] = mode_month
        df_clean['month_built'] = df_clean['month_built'].fillna(mode_month)
        
        median_safety = df_clean['safety_index'].median()
        self.imputation_values['safety_index'] = median_safety
        df_clean['safety_index'] = df_clean['safety_index'].fillna(median_safety)
        
        # Categorical missing values
        mode_furn = df_clean['furnishing_status'].mode()[0]
        self.imputation_values['furnishing_status'] = mode_furn
        df_clean['furnishing_status'] = df_clean['furnishing_status'].fillna(mode_furn)
        
        # 2. Feature Engineering with Exact Month & Year Built
        MONTH_MAP = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month_nums = df_clean['month_built'].map(MONTH_MAP).fillna(10)
        df_clean['property_age'] = (2026.0 - df_clean['year_built']) + (10.0 - month_nums) / 12.0
        df_clean['property_age'] = df_clean['property_age'].clip(lower=0.0)
        df_clean['bed_to_bath_ratio'] = df_clean['bedrooms'] / (df_clean['bathrooms'] + 0.1)
        
        # Parse amenities string into binary indicator columns
        for amenity in AMENITIES_LIST:
            col_name = f"has_{amenity.lower().replace('/', '_').replace(' ', '_')}"
            df_clean[col_name] = df_clean['amenities'].apply(
                lambda x: 1.0 if isinstance(x, str) and amenity in x else 0.0
            )
            
        amenity_cols = [f"has_{a.lower().replace('/', '_').replace(' ', '_')}" for a in AMENITIES_LIST]
        df_clean['amenity_score'] = df_clean[amenity_cols].sum(axis=1)
        
        # 3. Label Encoding for Entity Embeddings
        encoded_cats = []
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le
            num_classes = len(le.classes_)
            self.cat_dims[col] = num_classes
            # Rule of thumb for embedding dimension: min(50, max(2, (num_classes + 1) // 2))
            emb_dim = min(16, max(3, int(np.ceil(num_classes ** 0.5 * 2))))
            self.emb_dims[col] = emb_dim
            encoded_cats.append(df_clean[col].values[:, np.newaxis])
            
        cat_array = np.hstack(encoded_cats).astype(np.int64)
        
        # 4. Standard Scaling for Continuous Features
        cont_features = df_clean[CONTINUOUS_COLS].values
        cont_array = self.scaler_continuous.fit_transform(cont_features).astype(np.float32)
        
        # 5. Target Scaling
        y_raw = df_clean[target_col].values.reshape(-1, 1)
        y_array = self.scaler_target.fit_transform(y_raw).astype(np.float32)
        
        return cat_array, cont_array, y_array, y_raw
        
    def transform_single(self, raw_input: dict):
        """
        Transforms a single raw feature dictionary into tensors for live inference.
        """
        # Impute missing if needed
        year = raw_input.get('year_built', self.imputation_values['year_built'])
        safety = raw_input.get('safety_index', self.imputation_values['safety_index'])
        furn = raw_input.get('furnishing_status', self.imputation_values['furnishing_status'])
        
        # Categorical encoding
        cat_indices = []
        for col in CATEGORICAL_COLS:
            val = str(raw_input.get(col, self.imputation_values.get(col, '')))
            le = self.label_encoders[col]
            if val in le.classes_:
                idx = int(le.transform([val])[0])
            else:
                idx = 0
            cat_indices.append(idx)
            
        # Continuous calculation with exact month precision
        month_name = str(raw_input.get('month_built', self.imputation_values.get('month_built', 'October')))
        MONTH_MAP = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month_num = MONTH_MAP.get(month_name, 10)
        property_age = max(0.0, (2026.0 - float(year)) + (10.0 - month_num) / 12.0)
        bedrooms = float(raw_input.get('bedrooms', 3))
        bathrooms = float(raw_input.get('bathrooms', 2))
        bed_bath_ratio = bedrooms / (bathrooms + 0.1)
        
        amenities = raw_input.get('amenities', [])
        if isinstance(amenities, str):
            amenities = [a.strip() for a in amenities.split(';') if a.strip()]
            
        amenity_flags = []
        for amenity in AMENITIES_LIST:
            flag = 1.0 if any(amenity.lower() in a.lower() for a in amenities) else 0.0
            amenity_flags.append(flag)
            
        amenity_score = sum(amenity_flags)
        
        cont_vals = [
            float(raw_input.get('area_sqft', 1800)),
            bedrooms,
            bathrooms,
            float(raw_input.get('garage_spaces', 1)),
            property_age,
            float(raw_input.get('distance_to_city_center_km', 10)),
            float(raw_input.get('school_rating', 7)),
            float(safety),
            bed_bath_ratio,
            amenity_score
        ] + amenity_flags
        
        cont_scaled = self.scaler_continuous.transform([cont_vals]).astype(np.float32)
        cat_indices = np.array([cat_indices], dtype=np.int64)
        
        return cat_indices, cont_scaled
        
    def inverse_transform_target(self, y_scaled):
        return self.scaler_target.inverse_transform(y_scaled)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

def prepare_data_splits(csv_path='data/housing_raw.csv', test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)
    preprocessor = HousingDataPreprocessor()
    cat_data, cont_data, y_data, y_raw = preprocessor.fit_transform(df)
    
    # Train / Val / Test split (70% train, 15% val, 15% test)
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio_adjusted, random_state=random_state)
    
    splits = {
        'train': (cat_data[train_idx], cont_data[train_idx], y_data[train_idx], y_raw[train_idx]),
        'val': (cat_data[val_idx], cont_data[val_idx], y_data[val_idx], y_raw[val_idx]),
        'test': (cat_data[test_idx], cont_data[test_idx], y_data[test_idx], y_raw[test_idx])
    }
    
    return splits, preprocessor

if __name__ == '__main__':
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'housing_raw.csv')
    splits, preprocessor = prepare_data_splits(csv_file)
    print("Preprocessing completed successfully!")
    print(f"Training samples: {len(splits['train'][0])}")
    print(f"Validation samples: {len(splits['val'][0])}")
    print(f"Test samples: {len(splits['test'][0])}")
    print(f"Categorical dimensions: {preprocessor.cat_dims}")
    print(f"Embedding dimensions: {preprocessor.emb_dims}")
    print(f"Continuous features count: {len(CONTINUOUS_COLS)}")
