"""
generate_dataset.py
Generates a realistic Indian residential real estate dataset with hierarchical
geographic entities (State -> City -> Locality/Place), luxury amenities,
realistic missing values, and market pricing in Indian Rupees (INR).
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from geo_data import GEO_HIERARCHY, PLACE_PREMIUMS, get_all_places_flat

MONTHS_LIST = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

def generate_housing_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    flat_places = get_all_places_flat()
    
    states = list(GEO_HIERARCHY.keys())
    # Balanced distribution across all Indian states in the hierarchy
    state_weights = np.ones(len(states)) / len(states)
    
    property_types = [
        'Apartment / Gated High-Rise Flat',
        'Single Family / Independent House',
        'Gated Community Villa',
        'Luxury Penthouse',
        'Row House / Townhouse'
    ]
    prop_type_multipliers = {
        'Apartment / Gated High-Rise Flat': 0.95,
        'Single Family / Independent House': 1.05,
        'Gated Community Villa': 1.45,
        'Luxury Penthouse': 1.40,
        'Row House / Townhouse': 1.00
    }
    
    furnishing_options = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished', 'Designer Furnished']
    furnishing_multipliers = {
        'Unfurnished': 1.0,
        'Semi-Furnished': 1.06,
        'Fully Furnished': 1.14,
        'Designer Furnished': 1.25
    }
    
    available_amenities = [
        'Swimming Pool', 'Gym/Fitness Center', 'Smart Home Automation',
        'EV Charging Station', 'Solar Panel Array', 'Private Garden',
        'Rooftop Terrace', 'Clubhouse Access', '24/7 Security Concierge'
    ]
    
    records = []
    
    for i in range(n_samples):
        # 1. Hierarchical Geographic Selection: State -> City -> Locality
        state = np.random.choice(states, p=state_weights)
        cities_in_state = list(GEO_HIERARCHY[state].keys())
        city = np.random.choice(cities_in_state)
        
        city_info = GEO_HIERARCHY[state][city]
        city_mult = city_info["multiplier"]
        locality = np.random.choice(city_info["places"])
        place_mult = PLACE_PREMIUMS.get(locality, 1.0)
        
        prop_type = np.random.choice(property_types, p=[0.45, 0.25, 0.12, 0.08, 0.10])
        furnishing = np.random.choice(furnishing_options, p=[0.28, 0.42, 0.22, 0.08])
        
        # Area depending on property type
        if prop_type == 'Apartment / Gated High-Rise Flat':
            area = int(np.random.normal(1250, 320))
            bedrooms = np.random.choice([1, 2, 3], p=[0.20, 0.55, 0.25])
            bathrooms = float(np.random.choice([1.0, 2.0, 3.0], p=[0.25, 0.55, 0.20]))
            lot_size = np.nan
        elif prop_type == 'Row House / Townhouse':
            area = int(np.random.normal(1750, 380))
            bedrooms = np.random.choice([2, 3, 4], p=[0.15, 0.70, 0.15])
            bathrooms = float(np.random.choice([2.0, 2.5, 3.0], p=[0.3, 0.5, 0.2]))
            lot_size = area + int(np.random.uniform(200, 800))
        elif prop_type == 'Gated Community Villa':
            area = int(np.random.normal(3800, 650))
            bedrooms = np.random.choice([4, 5, 6], p=[0.4, 0.4, 0.2])
            bathrooms = float(np.random.choice([3.5, 4.0, 4.5, 5.0], p=[0.2, 0.4, 0.25, 0.15]))
            lot_size = area + int(np.random.uniform(2500, 8000))
        elif prop_type == 'Luxury Penthouse':
            area = int(np.random.normal(3100, 550))
            bedrooms = np.random.choice([3, 4, 5], p=[0.35, 0.50, 0.15])
            bathrooms = float(np.random.choice([3.0, 3.5, 4.0], p=[0.3, 0.4, 0.3]))
            lot_size = np.nan
        else: # Single Family / Independent House
            area = int(np.random.normal(2300, 550))
            bedrooms = np.random.choice([2, 3, 4, 5], p=[0.10, 0.45, 0.35, 0.10])
            bathrooms = float(np.random.choice([2.0, 2.5, 3.0, 3.5], p=[0.3, 0.35, 0.25, 0.10]))
            lot_size = area + int(np.random.uniform(1000, 5000))
            
        area = max(550, min(area, 6500))
        
        # Amenities selection
        n_amenities = int(np.clip(np.random.poisson(3.2), 0, len(available_amenities)))
        chosen_amenities = list(np.random.choice(available_amenities, size=n_amenities, replace=False))
        amenities_str = "; ".join(chosen_amenities) if chosen_amenities else "None"
        
        # Year Built up to recent date (1980 through 2026)
        num_years = 2026 - 1980 + 1  # 47 years
        year_built = int(np.random.choice(range(1980, 2027), p=np.linspace(0.01, 0.04, num_years)/np.sum(np.linspace(0.01, 0.04, num_years))))
        
        # Month Built / Possession Date
        if year_built == 2026:
            # Current year up to October 2026
            month_built = str(np.random.choice(MONTHS_LIST[:10]))
        else:
            month_built = str(np.random.choice(MONTHS_LIST))
            
        month_num = MONTHS_LIST.index(month_built) + 1
        
        # Exact property age with month precision relative to current date (October 2026)
        property_age = round(max(0.0, (2026 - year_built) + (10 - month_num) / 12.0), 2)
        
        # Distance to city center/metro (km)
        if any(w in locality for w in ['HITEC', 'BKC', 'Indiranagar', 'Cyber City', 'Benz Circle']):
            dist_center = round(np.random.uniform(0.8, 5.0), 1)
        elif any(w in locality for w in ['Whitefield', 'Gachibowli', 'Koramangala', 'Bandra', 'Jubilee']):
            dist_center = round(np.random.uniform(2.5, 12.0), 1)
        else:
            dist_center = round(np.random.uniform(8.0, 30.0), 1)
            
        school_rating = round(np.clip(np.random.normal(8.0 if city_mult >= 1.5 else 6.8, 1.2), 2.0, 10.0), 1)
        safety_index = round(np.clip(np.random.normal(80 if city_mult >= 1.5 else 70, 10), 30.0, 99.0), 1)
        garage_spaces = int(np.random.choice([0, 1, 2, 3], p=[0.10, 0.40, 0.40, 0.10])) if prop_type != 'Apartment / Gated High-Rise Flat' else 1
        
        # Price Calculation (in Indian Rupees)
        base_rate_sqft = 6200.0
        prop_mult = prop_type_multipliers[prop_type]
        furn_mult = furnishing_multipliers[furnishing]
        
        # Combined geographic multiplier: City Multiplier * Micro-market Place Premium
        geo_multiplier = city_mult * place_mult
        
        amenity_bonus = len(chosen_amenities) * 220000.0
        age_depreciation = max(0.72, 1.0 - (property_age * 0.0065))
        location_sqft_price = base_rate_sqft * geo_multiplier
        
        raw_price = (area * location_sqft_price * prop_mult * age_depreciation)
        raw_price += (bedrooms * 280000.0) + (bathrooms * 190000.0) + (garage_spaces * 160000.0)
        raw_price += (school_rating * 180000.0) + (safety_index * 12000.0)
        raw_price += (amenity_bonus * geo_multiplier)
        raw_price *= furn_mult
        raw_price -= (dist_center * 32000.0)
        
        # Stochastic market variance
        noise = np.random.normal(1.0, 0.065)
        price = round(max(3200000.0, raw_price * noise), -3)
        
        composite_location = f"{locality}, {city}, {state}"
        
        records.append({
            'property_id': f"IND-PROP-{10000 + i}",
            'state': state,
            'city': city,
            'locality': locality,
            'location': composite_location,
            'property_type': prop_type,
            'furnishing_status': furnishing,
            'area_sqft': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'garage_spaces': garage_spaces,
            'year_built': year_built,
            'month_built': month_built,
            'month_num': month_num,
            'property_age': property_age,
            'lot_size_sqft': lot_size,
            'distance_to_city_center_km': dist_center,
            'school_rating': school_rating,
            'safety_index': safety_index,
            'amenities': amenities_str,
            'amenities_count': len(chosen_amenities),
            'price': price
        })
        
    df = pd.DataFrame(records)
    
    # Missing values
    missing_mask_year = np.random.rand(len(df)) < 0.035
    df.loc[missing_mask_year, 'year_built'] = np.nan
    
    missing_mask_month = np.random.rand(len(df)) < 0.020
    df.loc[missing_mask_month, 'month_built'] = np.nan
    
    missing_mask_safety = np.random.rand(len(df)) < 0.028
    df.loc[missing_mask_safety, 'safety_index'] = np.nan
    
    missing_mask_furn = np.random.rand(len(df)) < 0.025
    df.loc[missing_mask_furn, 'furnishing_status'] = np.nan
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, 'housing_raw.csv')
    df = generate_housing_data(5000)
    df.to_csv(output_path, index=False)
    print(f"Generated hierarchical Indian housing dataset with {len(df)} rows.")
    print(f"Saved to: {output_path}")
    print("\nStates Included:", df['state'].unique().tolist())
    print("Cities Included:", df['city'].unique().tolist())
    print(f"Unique Localities/Places Count: {df['locality'].nunique()}")
    print("\nSample preview:")
    print(df[['state', 'city', 'locality', 'property_type', 'area_sqft', 'price']].head())
