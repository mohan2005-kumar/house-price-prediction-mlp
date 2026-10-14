"""
geo_data.py
Comprehensive Indian Real Estate Geographic Hierarchy:
States -> Cities -> Localities/Places with realistic market price multipliers.
"""

GEO_HIERARCHY = {
    "Telangana": {
        "Hyderabad": {
            "multiplier": 1.45,
            "places": [
                "HITEC City & Cyber Towers",
                "Gachibowli Financial District",
                "Jubilee Hills & Film Nagar",
                "Banjara Hills Road No 12",
                "Madhapur & Durgam Cheruvu",
                "Kondapur & Botanical Garden",
                "Kukatpally Housing Board (KPHB)",
                "Manikonda & Puppalguda"
            ]
        },
        "Warangal": {
            "multiplier": 0.78,
            "places": [
                "Hanamkonda Kazipet Corridor",
                "Subedari Commercial Hub",
                "Kakatiya University Enclave"
            ]
        }
    },
    "Andhra Pradesh": {
        "Amaravati / Vijayawada": {
            "multiplier": 0.95,
            "places": [
                "Amaravati Capital Tech City & VIT-AP Zone",
                "Benz Circle & MG Road",
                "Tadepalli & Kanaka Durga Bypass",
                "Guntur Inner Ring Road Corridor",
                "Mangalagiri AIIMS Suburb"
            ]
        },
        "Visakhapatnam (Vizag)": {
            "multiplier": 1.10,
            "places": [
                "Rushikonda IT SEZ & Beach Road",
                "Madhurawada Tech Township",
                "Siripuram & Waltair Uplands",
                "Gajuwaka Industrial Belt"
            ]
        }
    },
    "Karnataka": {
        "Bengaluru": {
            "multiplier": 1.55,
            "places": [
                "Whitefield & ITPL Main Road",
                "Indiranagar 100ft Road",
                "Koramangala 4th Block",
                "Electronic City Phase 1 & 2",
                "Bellandur & Outer Ring Road (ORR)",
                "HSR Layout Sectors 1-7",
                "Hebbal & Manyata Tech Park",
                "Sarjapur Road Tech Corridor"
            ]
        },
        "Mysuru": {
            "multiplier": 0.85,
            "places": [
                "Gokulam & Jayalakshmipuram",
                "Vijayanagar 1st-4th Stage",
                "Hebbal Electronic City Mysuru"
            ]
        }
    },
    "Maharashtra": {
        "Mumbai": {
            "multiplier": 2.25,
            "places": [
                "Bandra West & Bandra-Kurla Complex (BKC)",
                "Andheri West & Lokhandwala",
                "Juhu Beach Enclave",
                "Powai & Hiranandani Gardens",
                "Worli Sea Face & Lower Parel",
                "Thane West & Ghodbunder Road",
                "Navi Mumbai - Palm Beach Road"
            ]
        },
        "Pune": {
            "multiplier": 1.20,
            "places": [
                "Hinjawadi IT Park Phase 1-3",
                "Baner & Balewadi High Street",
                "Koregaon Park & Kalyani Nagar",
                "Viman Nagar & Magarpatta Cybercity",
                "Wakad & Pimple Saudagar"
            ]
        }
    },
    "Delhi NCR": {
        "Gurgaon (Gurugram)": {
            "multiplier": 1.60,
            "places": [
                "Cyber City & DLF Phase 1-5",
                "Golf Course Road & Extension",
                "Sohna Road Commercial Belt",
                "Sector 57 & Sushant Lok"
            ]
        },
        "Delhi": {
            "multiplier": 1.75,
            "places": [
                "South Extension & Greater Kailash",
                "Hauz Khas & Green Park",
                "Vasant Kunj & Shanti Niketan",
                "Dwarka Expressway Sectors"
            ]
        },
        "Noida": {
            "multiplier": 1.25,
            "places": [
                "Sector 62 & 128 Tech Corridor",
                "Noida Expressway Sector 137-144",
                "Sector 18 Commercial Hub"
            ]
        }
    },
    "Tamil Nadu": {
        "Chennai": {
            "multiplier": 1.30,
            "places": [
                "OMR IT Expressway (Tidel Park to Sholinganallur)",
                "Velachery & Guindy Industrial Hub",
                "Anna Nagar West Extension",
                "Adyar & Besant Nagar Coastal",
                "ECR Coastal Villa Belt"
            ]
        },
        "Coimbatore": {
            "multiplier": 0.88,
            "places": [
                "RS Puram & Race Course",
                "Saravanampatti IT Corridor",
                "Avinashi Road Peelamedu"
            ]
        }
    },
    "West Bengal": {
        "Kolkata": {
            "multiplier": 0.92,
            "places": [
                "Salt Lake Sector V (IT Hub)",
                "New Town Action Area 1-3",
                "Ballygunge & Park Street",
                "Rajarhat Expressway Corridor"
            ]
        }
    },
    "Gujarat": {
        "Ahmedabad": {
            "multiplier": 1.05,
            "places": [
                "SG Highway IT Corridor",
                "Prahlad Nagar & Satellite",
                "Bodakdev & Thaltej Enclave",
                "GIFT City Corridor Gandhinagar"
            ]
        },
        "Surat": {
            "multiplier": 0.88,
            "places": [
                "Vesu & VIP Road Corridor",
                "Piplod & Dumas Beach Road"
            ]
        }
    },
    "Kerala": {
        "Kochi": {
            "multiplier": 1.02,
            "places": [
                "Kakkanad InfoPark Zone",
                "Marine Drive Waterfront",
                "Panampilly Nagar Boulevard",
                "Edappally Metro Corridor"
            ]
        },
        "Thiruvananthapuram": {
            "multiplier": 0.92,
            "places": [
                "Technopark Kazhakoottam IT Hub",
                "Kowdiar Royal Enclave",
                "Pattom Central Suburb"
            ]
        }
    },
    "Rajasthan": {
        "Jaipur": {
            "multiplier": 0.86,
            "places": [
                "Malviya Nagar & Jagatpura",
                "Vaishali Nagar Highstreet",
                "Mansarovar Tech Corridor"
            ]
        }
    },
    "Uttar Pradesh": {
        "Lucknow": {
            "multiplier": 0.84,
            "places": [
                "Gomti Nagar Extension IT City",
                "Hazratganj Prime Heritage",
                "Shaheed Path Commercial Belt"
            ]
        }
    }
}

# Micro-market place modifiers within each city
PLACE_PREMIUMS = {
    # Mumbai
    "Bandra West & Bandra-Kurla Complex (BKC)": 1.35,
    "Worli Sea Face & Lower Parel": 1.30,
    "Juhu Beach Enclave": 1.28,
    "Powai & Hiranandani Gardens": 1.05,
    "Andheri West & Lokhandwala": 1.05,
    "Thane West & Ghodbunder Road": 0.75,
    "Navi Mumbai - Palm Beach Road": 0.82,
    # Hyderabad
    "Jubilee Hills & Film Nagar": 1.40,
    "Banjara Hills Road No 12": 1.35,
    "HITEC City & Cyber Towers": 1.15,
    "Gachibowli Financial District": 1.15,
    "Madhapur & Durgam Cheruvu": 1.10,
    "Kondapur & Botanical Garden": 1.02,
    "Manikonda & Puppalguda": 0.95,
    "Kukatpally Housing Board (KPHB)": 0.90,
    # Bengaluru
    "Indiranagar 100ft Road": 1.35,
    "Koramangala 4th Block": 1.30,
    "Whitefield & ITPL Main Road": 1.10,
    "Bellandur & Outer Ring Road (ORR)": 1.12,
    "HSR Layout Sectors 1-7": 1.10,
    "Hebbal & Manyata Tech Park": 1.05,
    "Electronic City Phase 1 & 2": 0.88,
    "Sarjapur Road Tech Corridor": 0.95,
    # Gurgaon / Delhi
    "Golf Course Road & Extension": 1.35,
    "Cyber City & DLF Phase 1-5": 1.25,
    "South Extension & Greater Kailash": 1.30,
    "Vasant Kunj & Shanti Niketan": 1.28,
    "Hauz Khas & Green Park": 1.20,
    "Noida Expressway Sector 137-144": 0.90,
    "Sector 62 & 128 Tech Corridor": 0.95,
    # Amaravati / Vijayawada
    "Amaravati Capital Tech City & VIT-AP Zone": 1.05,
    "Benz Circle & MG Road": 1.20,
    "Tadepalli & Kanaka Durga Bypass": 1.00,
    # Chennai
    "Adyar & Besant Nagar Coastal": 1.30,
    "Anna Nagar West Extension": 1.18,
    "OMR IT Expressway (Tidel Park to Sholinganallur)": 1.05,
    "ECR Coastal Villa Belt": 1.25
}

def get_all_places_flat():
    all_places = []
    for state, cities in GEO_HIERARCHY.items():
        for city, data in cities.items():
            for place in data["places"]:
                all_places.append({
                    "state": state,
                    "city": city,
                    "place": place,
                    "composite_location": f"{place}, {city}, {state}"
                })
    return all_places
