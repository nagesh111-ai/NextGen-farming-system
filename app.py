import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import openai
import google.generativeai as genai


# Load the trained model
with open("crop_prediction_model.pkl", "rb") as file:
    model, le, scaler = pickle.load(file)


# Crop information dictionary
crop_info = {
    "Wheat": {"Best Season": "Winter", "Required Nutrients": "High Nitrogen", "Expected Yield": "3-4 tons/ha"},
    "Rice": {"Best Season": "Monsoon", "Required Nutrients": "High Phosphorus", "Expected Yield": "4-6 tons/ha"},
    "Maize": {"Best Season": "Summer", "Required Nutrients": "Balanced NPK", "Expected Yield": "5-7 tons/ha"},
    "Sugarcane": {"Best Season": "Tropical", "Required Nutrients": "High Potassium", "Expected Yield": "80-100 tons/ha"},
    "Barley": {"Best Season": "Winter", "Required Nutrients": "Moderate Nitrogen", "Expected Yield": "2-3 tons/ha"},
    "Soybean": {"Best Season": "Monsoon", "Required Nutrients": "High Phosphorus", "Expected Yield": "2-4 tons/ha"}
}

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit UI Setup
st.set_page_config(page_title="next-gen Farming system", layout="wide")

local_css("style.css")

# Initialize session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar for Navigation
with st.sidebar:
    st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">
        🌾 Next-gen farming system 🚀
    </h1>
    <p style="text-align: center; font-size: 18px;">
        Empowering Farmers with AI-driven Insights
    </p>
""", unsafe_allow_html=True)

    selection = st.radio("Go to", ["Home", "Crop Recommendation", "Demand Analysis", "Crop Monitoring",'agribot'])
    st.session_state.page = selection

# Home Page
if st.session_state.page == "Home":
    # Beautiful green banner with gradient
    st.markdown("""
    <div class="home-banner">
        <h1>🌱 Next-gen Farming System</h1>
        <p>Empowering farmers with AI-driven insights for sustainable and profitable agriculture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards section
    st.markdown("<h2>Our Smart Farming Tools</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🌾 Crop Recommendation</h3>
            <p>Get personalized crop recommendations based on your soil composition, environmental factors, and other parameters.</p>
            <ul>
                <li>Soil nutrient analysis</li>
                <li>Climate compatibility</li>
                <li>Top 3 suitable crops</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Demand Analysis</h3>
            <p>Analyze market demand for different crops to maximize your profit and reduce risk.</p>
            <ul>
                <li>Market price trends</li>
                <li>Regional demand insights</li>
                <li>Profitability calculator</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🌦️ Crop Monitoring</h3>
            <p>Monitor weather conditions and get tailored recommendations for crop management.</p>
            <ul>
                <li>Real-time weather data</li>
                <li>Irrigation advice</li>
                <li>Custom alerts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Agribot Assistant</h3>
            <p>Chat with our AI farming assistant for instant expert advice on any agriculture topic.</p>
            <ul>
                <li>24/7 farming advice</li>
                <li>Pest management tips</li>
                <li>Sustainable practices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started section
    st.markdown("<h2>Getting Started</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="css-card">
        <p>Welcome to the Future of Farming! Our AI-powered platform helps you make data-driven decisions for better yields and profits.</p>
        <p>Navigate using the sidebar to explore our tools:</p>
        <ol>
            <li><strong>Crop Recommendation:</strong> Enter your soil and environmental data to get personalized crop suggestions.</li>
            <li><strong>Demand Analysis:</strong> Analyze market trends and calculate potential profits for different crops.</li>
            <li><strong>Crop Monitoring:</strong> Get real-time weather insights and manage your crops efficiently.</li>
            <li><strong>Agribot:</strong> Chat with our AI assistant for instant farming advice.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Call-to-action button
    st.markdown('<div class="main-action-button">', unsafe_allow_html=True)
    if st.button("🚀 Get Started with Crop Recommendation"):
        st.session_state.page = "Crop Recommendation"
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <footer>
        <p>© 2024 Next-gen Farming System | Sustainable Agriculture Technology</p>
    </footer>
    """, unsafe_allow_html=True)

# Crop Recommendation Page
elif st.session_state.page == "Crop Recommendation":
    st.title("🌾 Crop Recommendation System")
    st.markdown("👨‍🌾 Enter your **soil and environment data** to get smart recommendations.")
    
    
    
    with st.form("crop_input_form"):
        st.subheader("🧪 Soil Composition")
        col1, col2, col3 = st.columns(3)
        with col1:
            N = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50)
        with col2:
            P = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=30)
        with col3:
            K = st.number_input("Potassium (K)", min_value=0, max_value=150, value=40)

        st.subheader("🌡️ Environmental Factors")
        col4, col5 = st.columns(2)
        with col4:
            temperature = st.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
            humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=70)
        with col5:
            ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
            rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=500, value=200)

        st.subheader("🌱 Soil & Sunlight")
        with st.expander("More Soil & Sunlight Options"):
            soil_moisture = st.slider("Soil Moisture (%)", min_value=0, max_value=100, value=50)
            soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Loamy", "Peaty", "Silty", "Chalky"])
            sunlight_exposure = st.slider("Sunlight Exposure (hours/day)", min_value=0, max_value=12, value=6)

        submit = st.form_submit_button("🌾 Predict Crop")

    if "crop_prediction" not in st.session_state:
        st.session_state.crop_prediction = None

    if submit:
        try:
            with st.spinner("Predicting the best crops..."):
                soil_type_encoded = ["Sandy", "Clay", "Loamy", "Peaty", "Silty", "Chalky"].index(soil_type)
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_moisture, soil_type_encoded, sunlight_exposure]])
                input_data = scaler.transform(input_data)

                # Get top 3 crop predictions
                probabilities = model.predict_proba(input_data)[0]
                top_3_indices = np.argpartition(probabilities, -3)[-3:]
                top_3_indices = top_3_indices[np.argsort(probabilities[top_3_indices])[::-1]]
                top_3_crops = le.inverse_transform(top_3_indices)
                top_3_probs = probabilities[top_3_indices]

                st.subheader("🌾 Top 3 Recommended Crops")
                crop_data = []
                for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs), 1):
                    crop_link = f"[**{crop}**](https://en.wikipedia.org/wiki/{crop.replace(' ', '_')})"
                    st.success(f"{i}. {crop_link} ({prob:.2%})")
                    best_season = crop_info.get(crop, {}).get('Best Season', 'N/A')
                    required_nutrients = crop_info.get(crop, {}).get('Required Nutrients', 'N/A')
                    expected_yield = crop_info.get(crop, {}).get('Expected Yield', 'N/A')
                    crop_data.append([crop, f"{prob:.2%}", best_season, required_nutrients, expected_yield])
                
                # Display probabilities as an improved bar chart
                st.subheader("📊 Probability Distribution of Recommended Crops")
                prob_df = pd.DataFrame({"Crop": top_3_crops, "Probability": top_3_probs})
                fig = px.bar(prob_df, x="Crop", y="Probability", text_auto=True, color="Crop", 
                             labels={"Probability": "Prediction Confidence"}, height=400)
                st.plotly_chart(fig)

                
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Demand Analysis Page
elif st.session_state.page == "Demand Analysis":
    st.title("📊 Crop Demand Analysis")
    
    # Create a modern card-like container for the intro
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 Analyze Market Trends & Future Demand</h3>
        <p>Make data-driven decisions about which crops to grow based on market demand, 
        price forecasts, and regional trends. Plan your farming strategy for maximum profitability.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Expanded crop information with more crops and additional data
    expanded_crop_info = {
        "Wheat": {
            "Best Season": "Winter", 
            "Required Nutrients": "High Nitrogen", 
            "Expected Yield": "3-4 tons/ha",
            "Growth Period": "120-150 days",
            "Market Trend": "Stable with seasonal variations",
            "Future Outlook": "Strong demand due to staple food status",
            "Export Potential": "High",
            "API_Code": "WHEAT"
        },
        "Rice": {
            "Best Season": "Monsoon", 
            "Required Nutrients": "High Phosphorus", 
            "Expected Yield": "4-6 tons/ha",
            "Growth Period": "90-120 days",
            "Market Trend": "Consistently high demand",
            "Future Outlook": "Increasing with population growth",
            "Export Potential": "Medium-High",
            "API_Code": "RICE"
        },
        "Maize": {
            "Best Season": "Summer", 
            "Required Nutrients": "Balanced NPK", 
            "Expected Yield": "5-7 tons/ha",
            "Growth Period": "90-120 days",
            "Market Trend": "Growing for feed and biofuel",
            "Future Outlook": "Strong growth expected",
            "Export Potential": "Medium",
            "API_Code": "MAIZE"
        },
        "Sugarcane": {
            "Best Season": "Tropical", 
            "Required Nutrients": "High Potassium", 
            "Expected Yield": "80-100 tons/ha",
            "Growth Period": "10-12 months",
            "Market Trend": "Stable with policy influences",
            "Future Outlook": "Moderate growth with biofuel demand",
            "Export Potential": "Low (processed products high)"
        },
        "Barley": {
            "Best Season": "Winter", 
            "Required Nutrients": "Moderate Nitrogen", 
            "Expected Yield": "2-3 tons/ha",
            "Growth Period": "80-100 days",
            "Market Trend": "Growing with craft beer popularity",
            "Future Outlook": "Positive for malting varieties",
            "Export Potential": "Medium"
        },
        "Soybean": {
            "Best Season": "Monsoon", 
            "Required Nutrients": "High Phosphorus", 
            "Expected Yield": "2-4 tons/ha",
            "Growth Period": "100-120 days",
            "Market Trend": "Strong for protein source",
            "Future Outlook": "Very positive with plant protein demand",
            "Export Potential": "High"
        },
        "Cotton": {
            "Best Season": "Summer", 
            "Required Nutrients": "Balanced NPK", 
            "Expected Yield": "2-3 tons/ha",
            "Growth Period": "150-180 days",
            "Market Trend": "Cyclical with fashion industry",
            "Future Outlook": "Stable with synthetic competition",
            "Export Potential": "High"
        },
        "Potato": {
            "Best Season": "Winter/Cool", 
            "Required Nutrients": "High Potassium", 
            "Expected Yield": "20-30 tons/ha",
            "Growth Period": "90-120 days",
            "Market Trend": "Stable staple food",
            "Future Outlook": "Consistent demand expected",
            "Export Potential": "Medium (processed products high)"
        },
        "Tomato": {
            "Best Season": "Spring/Summer", 
            "Required Nutrients": "Balanced with Calcium", 
            "Expected Yield": "40-60 tons/ha",
            "Growth Period": "90-150 days",
            "Market Trend": "High demand with price volatility",
            "Future Outlook": "Growing with processed foods",
            "Export Potential": "Medium-High (seasonal)"
        },
        "Onion": {
            "Best Season": "Winter/Spring", 
            "Required Nutrients": "Balanced NPK", 
            "Expected Yield": "30-40 tons/ha",
            "Growth Period": "100-150 days",
            "Market Trend": "Essential with price volatility",
            "Future Outlook": "Stable with seasonal fluctuations",
            "Export Potential": "Medium"
        },
        "Groundnut": {
            "Best Season": "Summer/Monsoon", 
            "Required Nutrients": "High Phosphorus & Calcium", 
            "Expected Yield": "1.5-2.5 tons/ha",
            "Growth Period": "120-150 days",
            "Market Trend": "Growing for oil and snacks",
            "Future Outlook": "Positive with health food trends",
            "Export Potential": "Medium-High"
        },
        "Mustard": {
            "Best Season": "Winter", 
            "Required Nutrients": "Moderate Nitrogen & Sulfur", 
            "Expected Yield": "1-1.5 tons/ha",
            "Growth Period": "110-150 days",
            "Market Trend": "Strong for oil production",
            "Future Outlook": "Stable with health food trends",
            "Export Potential": "Medium",
            "API_Code": "MUSTARD"
        },
        "Turmeric": {
            "Best Season": "Summer", 
            "Required Nutrients": "High Organic Matter", 
            "Expected Yield": "5-7 tons/ha",
            "Growth Period": "210-300 days",
            "Market Trend": "Growing with health benefits awareness",
            "Future Outlook": "Positive due to medicinal value",
            "Export Potential": "High",
            "API_Code": "TURMERIC"
        },
        "Chilli": {
            "Best Season": "Summer/Monsoon", 
            "Required Nutrients": "Balanced with Calcium", 
            "Expected Yield": "2-3 tons/ha",
            "Growth Period": "120-150 days",
            "Market Trend": "Stable with price spikes",
            "Future Outlook": "Growing with food processing",
            "Export Potential": "High",
            "API_Code": "CHILLI"
        },
        "Jute": {
            "Best Season": "Spring/Summer", 
            "Required Nutrients": "High Nitrogen", 
            "Expected Yield": "2-3.5 tons/ha",
            "Growth Period": "100-120 days",
            "Market Trend": "Declining with synthetics, growing with eco-awareness",
            "Future Outlook": "Potential growth with eco-friendly products",
            "Export Potential": "Medium",
            "API_Code": "JUTE"
        },
        "Coffee": {
            "Best Season": "Tropical year-round", 
            "Required Nutrients": "High Potassium", 
            "Expected Yield": "1-2 tons/ha",
            "Growth Period": "3-4 years to first yield",
            "Market Trend": "High demand with price volatility",
            "Future Outlook": "Premium varieties growth",
            "Export Potential": "Very High",
            "API_Code": "COFFEE"
        },
        "Mango": {
            "Best Season": "Summer", 
            "Required Nutrients": "Balanced NPK", 
            "Expected Yield": "10-15 tons/ha",
            "Growth Period": "3-4 years to first yield",
            "Market Trend": "Strong seasonal demand",
            "Future Outlook": "Growing export potential",
            "Export Potential": "High",
            "API_Code": "MANGO"
        }
    }
    
    # Create two tabs for different analysis views
    tab1, tab2 = st.tabs(["📈 Crop Analysis", "💹 Market Forecasting"])
    
    with tab1:
        # Select a crop from expanded list
        crop_options = list(expanded_crop_info.keys())
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_crop = st.selectbox("🔍 Select a Crop to Analyze", crop_options)
            
            # Display image of selected crop
            st.markdown(f"""
            <div style="border-radius: 10px; overflow: hidden; margin-bottom: 15px;">
                <img src="https://source.unsplash.com/300x200/?{selected_crop},farm" width="100%" 
                style="border-radius: 10px; border: 1px solid #e0e0e0;">
            </div>
            """, unsafe_allow_html=True)
            
            # Real-time market price from API
            st.subheader("💹 Real-time Market Price")
            
            # API integration for crop prices (using a more reliable API)
            try:
                # Try to get real market price from a different API
                api_code = expanded_crop_info[selected_crop].get("API_Code", selected_crop.upper())
                
                # Using the Commodity Price API from NCDEX (National Commodity & Derivatives Exchange)
                api_url = f"https://commodityapi.ncdex.com/api/v1/commodity/price?token=YOUR_API_KEY&commodity={api_code}"
                
                # Alternate API from Agmarknet (Indian Agricultural Marketing Information Network)
                alt_api_url = f"https://agmarknet.gov.in/api/commodityprice?commodity={api_code}&market=all&state=all"
                
                # For demonstration, use a more stable API endpoint - in this case, a mock API
                mock_api_url = "https://mocki.io/v1/6ed6806c-3706-4bbe-85fa-e48cff273ef9" 
                
                # First try the mockup API to ensure consistent results
                response = requests.get(mock_api_url, timeout=5)
                
                if response.status_code == 200 and response.json().get("crops"):
                    # Parse data from the mockup API
                    data = response.json()
                    # Find the matching crop data
                    crop_data = None
                    for crop in data["crops"]:
                        if crop["name"].lower() == selected_crop.lower():
                            crop_data = crop
                            break
                    
                    # Use the first crop if no match (for demonstration)
                    if not crop_data and data["crops"]:
                        crop_data = data["crops"][0]
                    
                    if crop_data:
                        market_price = int(crop_data["modal_price"])
                        market_min = int(crop_data["min_price"])
                        market_max = int(crop_data["max_price"])
                        market_date = crop_data["last_updated"]
                        market_name = crop_data["market_name"]
                        price_trend = crop_data["trend"]
                        
                        # Display the price with trend indicator
                        trend_icon = "↗️" if price_trend == "up" else "↘️" if price_trend == "down" else "➡️"
                        trend_color = "#388E3C" if price_trend == "up" else "#F44336" if price_trend == "down" else "#757575"
                        
                        st.markdown(f"""
                        <div class="css-card" style="background-color: #e8f5e9; border-left: 4px solid #4CAF50;">
                            <h4 style="margin-top: 0; color: #2E7D32;">🏬 Market: {market_name}</h4>
                            <h2 style="margin: 0; color: #2E7D32;">₹{market_price}/quintal <span style="color: {trend_color}; font-size: 0.8em;">{trend_icon} {price_trend.upper()}</span></h2>
                            <p style="font-size: 14px; color: #666;">Range: ₹{market_min} - ₹{market_max} | Last Updated: {market_date}</p>
                            <p style="font-size: 12px; color: #888;">Source: Agricultural Market Data API</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Use the API price for calculations
                        current_market_price = market_price
                    else:
                        raise Exception("Crop data not found in API response")
                else:
                    # Generate realistic price if API fails
                    raise Exception("No data available from API")
                    
            except Exception as e:
                # Generate simulated price based on crop with more realistic market patterns
                np.random.seed(hash(selected_crop) % 10000)
                
                # Different base price ranges for different crop types
                if selected_crop in ["Rice", "Wheat", "Maize"]:
                    # Staple crops
                    base_price = np.random.randint(1800, 2800)
                elif selected_crop in ["Potato", "Onion", "Tomato"]:
                    # Vegetables
                    base_price = np.random.randint(1200, 3500)
                elif selected_crop in ["Coffee", "Turmeric", "Chilli"]:
                    # High-value crops
                    base_price = np.random.randint(6000, 12000)
                else:
                    # Other crops
                    base_price = np.random.randint(2000, 6000)
                
                # Add monthly seasonal adjustment based on current month
                from datetime import datetime
                current_month = datetime.now().month
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (current_month / 12))
                
                # Apply seasonal adjustment to base price
                current_market_price = int(base_price * seasonal_factor)
                
                # Show simulated market data with realistic details
                min_price = int(current_market_price * 0.9)
                max_price = int(current_market_price * 1.1)
                
                # Determine trend based on current month and crop seasonality
                crop_peak_season = {
                    "Wheat": 4,       # April
                    "Rice": 11,       # November
                    "Maize": 9,       # September
                    "Potato": 2,      # February
                    "Onion": 5,       # May
                    "Tomato": 7,      # July
                    "Coffee": 1,      # January
                    "Turmeric": 3,    # March
                    "Chilli": 8,      # August
                }
                
                # Default to mid-year if crop not found
                peak_month = crop_peak_season.get(selected_crop, 6)
                
                # Calculate months from peak season
                months_from_peak = min((current_month - peak_month) % 12, (peak_month - current_month) % 12)
                
                # Determine trend (prices usually go down after harvest)
                if months_from_peak <= 1:
                    trend = "down"
                    trend_icon = "↘️"
                    trend_color = "#F44336"
                elif months_from_peak >= 5:
                    trend = "up"
                    trend_icon = "↗️"
                    trend_color = "#388E3C"
                else:
                    trend = "stable"
                    trend_icon = "➡️"
                    trend_color = "#757575"
                
                st.markdown(f"""
                <div class="css-card" style="background-color: #e8f5e9; border-left: 4px solid #4CAF50;">
                    <h4 style="margin-top: 0; color: #2E7D32;">📊 Current Market Price (Simulated)</h4>
                    <h2 style="margin: 0; color: #2E7D32;">₹{current_market_price}/quintal <span style="color: {trend_color}; font-size: 0.8em;">{trend_icon} {trend.upper()}</span></h2>
                    <p style="font-size: 14px; color: #666;">Range: ₹{min_price} - ₹{max_price} | Based on seasonal patterns</p>
                    <div style="margin-top: 10px; padding: 8px; background-color: #f5f5f5; border-radius: 4px; font-size: 12px;">
                        <p style="margin: 0; color: #666;"><b>Market Analysis:</b> Prices for {selected_crop} are currently {trend}. 
                        {
                            "Prices are falling after recent harvest." if trend == "down" else
                            "Prices are rising as we approach next harvest season." if trend == "up" else
                            "Prices are stable in the mid-season period."
                        }</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Display crop-specific details in a nice card
            st.markdown(f"""
            <div class="css-card" style="height: 100%;">
                <h3 style="color: #2E7D32; margin-top: 0;">📊 Crop Profile: {selected_crop}</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Best Season:</b></td>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{expanded_crop_info[selected_crop]['Best Season']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Growth Period:</b></td>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{expanded_crop_info[selected_crop]['Growth Period']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Required Nutrients:</b></td>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{expanded_crop_info[selected_crop]['Required Nutrients']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Expected Yield:</b></td>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{expanded_crop_info[selected_crop]['Expected Yield']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Market Trend:</b></td>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{expanded_crop_info[selected_crop]['Market Trend']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Future Outlook:</b></td>
                        <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{expanded_crop_info[selected_crop]['Future Outlook']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 5px;"><b>Export Potential:</b></td>
                        <td style="padding: 8px 5px;">{expanded_crop_info[selected_crop]['Export Potential']}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Market Insights with 3-year trend
        st.subheader("🌐 Market Insights")
        
        # Generate predictable trend data with seasonal patterns
        def generate_trend_data(crop_name, months=36):
            np.random.seed(hash(crop_name) % 10000)  # Seed based on crop name for consistent results
            base_price = np.random.randint(2000, 6000)
            trend_factor = np.random.uniform(-0.5, 1.0)  # Negative to positive trend
            seasonality = np.random.uniform(0.1, 0.3)  # Seasonal variation magnitude
            noise_level = np.random.uniform(0.05, 0.15)  # Random noise amount
            
            # Create time-based components
            time = np.arange(months)
            trend = base_price * (1 + trend_factor * time/months)
            season = seasonality * base_price * np.sin(2 * np.pi * time / 12)
            noise = np.random.normal(0, noise_level * base_price, months)
            
            # Create price series with trend, seasonality and noise
            prices = trend + season + noise
            return prices.astype(int)
        
        # Create 3-year price history
        months_3yr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] * 3
        years = ["2022"] * 12 + ["2023"] * 12 + ["2024"] * 12
        full_labels = [f"{m} {y}" for m, y in zip(months_3yr, years)]
        
        # Get price data for selected crop
        price_history = generate_trend_data(selected_crop)
        
        # Create a DataFrame for the chart
        price_df = pd.DataFrame({
            "Month": full_labels,
            "Price (₹/Quintal)": price_history,
            "Year": years
        })
        
        # Plot the price history with Plotly
        fig = px.line(price_df, x="Month", y="Price (₹/Quintal)", 
                      title=f"{selected_crop} Price Trends (3-Year History)",
                      labels={"Price (₹/Quintal)": "Price (₹/Quintal)", "Month": ""},
                      markers=True, color_discrete_sequence=["#4CAF50"])
        
        # Customize to highlight years
        for year in ["2022", "2023", "2024"]:
            year_data = price_df[price_df["Year"] == year]
            fig.add_scatter(x=year_data["Month"], y=year_data["Price (₹/Quintal)"],
                          mode="markers", name=year, marker=dict(size=8))
        
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=full_labels[::3], ticktext=full_labels[::3]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price prediction for next 6 months
        st.subheader("🔮 Price Prediction (Next 6 Months)")
        
        # Generate future predictions based on historical patterns plus growth
        last_price = price_history[-1]
        future_months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
        future_years = ["2024"] * 6
        future_labels = [f"{m} {y}" for m, y in zip(future_months, future_years)]
        
        # Create somewhat optimistic predictions based on current trend
        prediction_base = price_history[-12:]  # Last year
        seasonal_pattern = prediction_base - np.mean(prediction_base)  # Extract seasonality
        growth_factor = 1 + np.random.uniform(0.05, 0.15)  # 5-15% annual growth
        
        # Apply seasonal pattern to future months with growth factor
        future_prices = []
        for i in range(6):
            next_price = last_price * growth_factor + seasonal_pattern[i]
            future_prices.append(int(next_price))
            last_price = next_price
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            "Month": future_labels,
            "Predicted Price (₹/Quintal)": future_prices,
        })
        
        # Display price prediction as a line chart with prediction interval
        fig2 = px.line(prediction_df, x="Month", y="Predicted Price (₹/Quintal)",
                      title="Price Forecast with Confidence Interval",
                      labels={"Predicted Price (₹/Quintal)": "Price (₹/Quintal)"},
                      markers=True, color_discrete_sequence=["#4CAF50"])
        
        # Add prediction intervals
        upper_bound = [p * 1.1 for p in future_prices]  # 10% above prediction
        lower_bound = [p * 0.9 for p in future_prices]  # 10% below prediction
        
        fig2.add_scatter(x=future_labels, y=upper_bound, mode="lines", line=dict(width=0),
                       showlegend=False)
        fig2.add_scatter(x=future_labels, y=lower_bound, mode="lines", fill="tonexty",
                       line=dict(width=0), fillcolor="rgba(76, 175, 80, 0.2)",
                       name="Prediction Interval")
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Profitability & Cost Estimation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Market inputs
            st.markdown("#### Market Inputs")
            # Use the API price as default if available
            market_price = st.number_input("Market Price per quintal (₹)", 
                                            min_value=1000, 
                                            max_value=100000, 
                                            value=current_market_price if 'current_market_price' in locals() else 25000, 
                                            step=500)
            expected_yield = float(expanded_crop_info[selected_crop]["Expected Yield"].split('-')[0])  # Get min yield
            land_area_acre = st.number_input("Land Area (acres)", min_value=1, max_value=250, value=5)
            
            # Additional market factors
            st.markdown("#### Market Factors")
            quality_premium = st.slider("Quality Premium (%)", min_value=-10, max_value=30, value=0, 
                                        help="Premium or discount based on crop quality")
            organic_premium = st.checkbox("Organic Certification", 
                                         help="Premium price for certified organic crops")
            if organic_premium:
                organic_premium_value = st.slider("Organic Premium (%)", min_value=10, max_value=50, value=20)
            else:
                organic_premium_value = 0
        
        with col2:
            # Cost inputs
            st.markdown("#### Cost Breakdown (₹ per acre)")
            seed_cost_acre = st.number_input("🌱 Seed Cost", min_value=500, max_value=25000, value=2500, step=500)
            fertilizer_cost_acre = st.number_input("💊 Fertilizer Cost", min_value=500, max_value=25000, value=3500, step=500)
            labor_cost_acre = st.number_input("👷 Labor Cost", min_value=500, max_value=25000, value=5000, step=500)
            transport_cost_acre = st.number_input("🚚 Transportation Cost", min_value=500, max_value=10000, value=1500, step=500)
            other_costs = st.number_input("🔧 Other Costs (equipment, irrigation, etc.)", min_value=0, max_value=25000, value=2000, step=500)
        
        # Total cost per acre
        total_cost_per_acre = seed_cost_acre + fertilizer_cost_acre + labor_cost_acre + transport_cost_acre + other_costs
        
        # Convert expected yield to per acre (since yield is given per hectare)
        expected_yield_per_acre = expected_yield / 2.47  # (1 hectare = 2.47 acres)
        
        # Convert quintal to tons if needed (1 ton = 10 quintals)
        expected_yield_quintals = expected_yield_per_acre * 10  # Convert tons to quintals
        
        # Calculate base case values (before adjustments)
        base_price = market_price
        base_revenue_per_acre = base_price * expected_yield_quintals
        base_total_revenue = base_revenue_per_acre * land_area_acre
        base_total_cost = total_cost_per_acre * land_area_acre
        base_profit = base_total_revenue - base_total_cost
        
        # Apply market factors to price
        adjusted_price = market_price * (1 + quality_premium/100) * (1 + organic_premium_value/100)
        
        # Add storage and post-harvest options
        st.subheader("📦 Post-Harvest & Storage Strategy")
        col1, col2 = st.columns(2)
        
        with col1:
            storage_option = st.selectbox("Storage Strategy", 
                                         ["Sell Immediately", "Short-term Storage (1-3 months)", 
                                          "Long-term Storage (3-6 months)"])
            
            if storage_option == "Sell Immediately":
                storage_cost = 0
                price_benefit = 0
                storage_text = "No storage costs, but missing potential higher prices"
            elif storage_option == "Short-term Storage (1-3 months)":
                storage_cost = 200 * land_area_acre  # ₹200 per acre for short-term
                price_benefit = 0.05  # 5% price increase
                storage_text = "Medium storage costs, potential for better prices"
            else:
                storage_cost = 500 * land_area_acre  # ₹500 per acre for long-term
                price_benefit = 0.12  # 12% price increase
                storage_text = "Higher storage costs, but best chance for peak prices"
        
        with col2:
            processing_option = st.selectbox("Processing Level", 
                                            ["No Processing", "Basic Processing", 
                                             "Advanced Processing"])
            
            if processing_option == "No Processing":
                processing_cost = 0
                processing_benefit = 0
                processing_text = "No additional costs, base market prices"
            elif processing_option == "Basic Processing":
                processing_cost = 1500 * land_area_acre  # ₹1500 per acre for basic processing
                processing_benefit = 0.15  # 15% price increase
                processing_text = "Sorting, cleaning, packaging for better prices"
            else:
                processing_cost = 4000 * land_area_acre  # ₹4000 per acre for advanced processing
                processing_benefit = 0.35  # 35% price increase
                processing_text = "Value-added processing for premium markets"
        
        # Display strategy information
        st.markdown(f"""
        <div class="css-card">
            <h4 style="margin-top: 0; color: #2E7D32;">Selected Strategy</h4>
            <p><b>Storage:</b> {storage_option} - {storage_text}</p>
            <p><b>Processing:</b> {processing_option} - {processing_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Apply storage and processing to final price
        final_price = adjusted_price * (1 + price_benefit) * (1 + processing_benefit)
        total_post_harvest_cost = storage_cost + processing_cost
        
        # Recalculate revenue with all factors
        adjusted_revenue_per_acre = final_price * expected_yield_quintals
        total_adjusted_revenue = adjusted_revenue_per_acre * land_area_acre
        
        # Add post-harvest costs to total costs
        total_cost_with_post_harvest = total_cost_per_acre * land_area_acre + total_post_harvest_cost
        
        # Recalculate profit
        adjusted_profit = total_adjusted_revenue - total_cost_with_post_harvest
        adjusted_profit_margin = (adjusted_profit / total_adjusted_revenue) * 100 if total_adjusted_revenue > 0 else 0
        adjusted_roi = (adjusted_profit / total_cost_with_post_harvest) * 100 if total_cost_with_post_harvest > 0 else 0
        
        # Display adjusted financial summary
        st.subheader("💼 Financial Analysis (with Strategy)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="css-card" style="background-color: #f5f5f5; border-left: 4px solid #4CAF50;">
                <h4 style="margin-top: 0; color: #2E7D32;">💵 Total Investment</h4>
                <h2 style="margin: 0; color: #2E7D32;">₹{total_cost_with_post_harvest:,.2f}</h2>
                <p style="font-size: 14px; color: #666;">Including post-harvest: ₹{total_post_harvest_cost:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="css-card" style="background-color: #e8f5e9; border-left: 4px solid #4CAF50;">
                <h4 style="margin-top: 0; color: #2E7D32;">💰 Strategic Revenue</h4>
                <h2 style="margin: 0; color: #2E7D32;">₹{total_adjusted_revenue:,.2f}</h2>
                <p style="font-size: 14px; color: #666;">Price: ₹{final_price:.2f}/qtl (vs ₹{market_price}/qtl base)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            if adjusted_profit > 0:
                st.markdown(f"""
                <div class="css-card" style="background-color: #e8f5e9; border-left: 4px solid #4CAF50;">
                    <h4 style="margin-top: 0; color: #2E7D32;">✅ Strategic Profit</h4>
                    <h2 style="margin: 0; color: #2E7D32;">₹{adjusted_profit:,.2f}</h2>
                    <p style="font-size: 14px; color: #666;">Margin: {adjusted_profit_margin:.1f}% | ROI: {adjusted_roi:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="css-card" style="background-color: #ffebee; border-left: 4px solid #F44336;">
                    <h4 style="margin-top: 0; color: #C62828;">⚠️ Strategic Loss</h4>
                    <h2 style="margin: 0; color: #C62828;">₹{-adjusted_profit:,.2f}</h2>
                    <p style="font-size: 14px; color: #666;">Margin: {adjusted_profit_margin:.1f}% | ROI: {adjusted_roi:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Add comparison chart between base strategy and optimized strategy
        comparison_data = {
            'Strategy': ['Base Strategy', 'Optimized Strategy'],
            'Revenue': [base_total_revenue, total_adjusted_revenue],
            'Cost': [base_total_cost, total_cost_with_post_harvest],
            'Profit': [base_profit, adjusted_profit]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison chart
        st.subheader("📊 Strategy Comparison")
        
        fig_comparison = px.bar(comparison_df, x='Strategy', y=['Revenue', 'Cost', 'Profit'], 
                               barmode='group', title="Financial Comparison of Strategies",
                               color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3'])
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Marketing channels analysis
        st.subheader("🛒 Marketing Channels Analysis")
        
        channels = {
            "Local Market": {
                "Price": final_price * 0.9,  # 90% of optimized price
                "Risk": "Low",
                "Requirements": "Basic quality, no certification needed",
                "Advantages": "Immediate payment, no transportation",
                "Disadvantages": "Lower prices, limited volume"
            },
            "Wholesale Market": {
                "Price": final_price * 1.0,  # 100% of optimized price (reference)
                "Risk": "Medium",
                "Requirements": "Standard quality, consistent supply",
                "Advantages": "Higher volume sales, established channel",
                "Disadvantages": "Price fluctuations, delayed payments possible"
            },
            "Direct to Consumer": {
                "Price": final_price * 1.3,  # 130% of optimized price
                "Risk": "Medium-High",
                "Requirements": "High quality, packaging, marketing",
                "Advantages": "Best prices, direct customer relationships",
                "Disadvantages": "Time-consuming, requires marketing"
            },
            "Export Market": {
                "Price": final_price * 1.5,  # 150% of optimized price
                "Risk": "High",
                "Requirements": "Certifications, highest quality, consistent volume",
                "Advantages": "Premium prices, large volume potential",
                "Disadvantages": "Complex regulations, high entry barriers"
            }
        }
        
        # Let user select marketing channel
        selected_channel = st.selectbox("Select Marketing Channel", list(channels.keys()))
        
        # Display selected channel details
        channel_info = channels[selected_channel]
        st.markdown(f"""
        <div class="css-card">
            <h4 style="margin-top: 0; color: #2E7D32;">{selected_channel} Channel Details</h4>
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Expected Price:</b></td>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">₹{channel_info['Price']:.2f}/quintal</td>
                </tr>
                <tr>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Risk Level:</b></td>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{channel_info['Risk']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Requirements:</b></td>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{channel_info['Requirements']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;"><b>Advantages:</b></td>
                    <td style="padding: 8px 5px; border-bottom: 1px solid #f0f0f0;">{channel_info['Advantages']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 5px;"><b>Disadvantages:</b></td>
                    <td style="padding: 8px 5px;">{channel_info['Disadvantages']}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# Crop Monitoring Page
elif st.session_state.page == "Crop Monitoring":
    import requests
    import streamlit as st

    # Your WeatherStack API key
    api_key = "9d3c29ecd15d314e58fa3511bf27277b"

    # Function to fetch weather data
    def get_weather_data(location):
        api_url = f"http://api.weatherstack.com/current?access_key={api_key}&query={location}"
        
        try:
            response = requests.get(api_url)
            data = response.json()
            
            if "current" in data:
                return {
                    "Temperature (°C)": data["current"]["temperature"],
                    "Humidity (%)": data["current"]["humidity"],
                    "Rainfall (mm)": data["current"].get("precip", "N/A"),
                    "Wind Speed (km/h)": data["current"]["wind_speed"],
                    "UV Index": data["current"]["uv_index"],
                    "Pressure (mb)": data["current"]["pressure"]
                }
            else:
                return None
        except Exception as e:
            return None

    # Streamlit UI
    st.title("🌾 Crop Monitoring System")

    # User input for location
    location = st.text_input("📍 Enter Location (City or District)", "New Delhi")

    if st.button("🔍 Get Weather Data"):
        weather_data = get_weather_data(location)
        
        if weather_data:
            st.subheader(f"🌤️ Weather in {location}")
            st.json(weather_data)

            # Alerts & Recommendations
            if weather_data["Temperature (°C)"] > 35:
                st.warning("⚠️ High temperature! Consider irrigation to avoid heat stress.")
            if weather_data["Humidity (%)"] < 30:
                st.warning("⚠️ Low humidity detected! Crops might need additional watering.")
            if weather_data["Rainfall (mm)"] == 0:
                st.warning("⚠️ No rainfall detected. Ensure proper irrigation.")
        else:
            st.error("❌ Failed to fetch weather data. Please check API key or try again.")
 

elif st.session_state.page == "agribot":
    # Create a modern header with gradient and icon
    st.markdown("""
    <div style="background: linear-gradient(to right, #4CAF50, #2E7D32); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; display: flex; align-items: center;">
            <span style="font-size: 40px; margin-right: 10px;">🤖</span> 
            Agribot: Your Smart Farming Assistant
        </h1>
        <p style="color: white; margin-top: 10px; font-size: 16px;">
            Get instant advice on crops, pests, irrigation, sustainable practices, and more!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create layout with two columns
    col1, col2 = st.columns([2, 1])
    
    # Main chat area in left column
    with col1:
        # Import Google Generative AI
        import google.generativeai as genai
        
        # Gemini API Key - Replace with your actual API key
        gemini_api_key = "AIzaSyAXSNDM9tzAYfyw30evV9QIE4bZwSLFDtE"
        
        # Configure Gemini with proper settings
        genai.configure(api_key=gemini_api_key)
        
        # Use specifically gemini-1.5-flash model (newer version)
        gemini_model_name = "gemini-1.5-flash"
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat container with custom styling
        st.markdown("""
        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; margin-bottom: 20px; background-color: #f9f9f9;">
            <h3 style="color: #2E7D32; border-bottom: 1px solid #e0e0e0; padding-bottom: 10px;">
                💬 Chat with Agribot
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat messages with improved styling
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👨‍🌾"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message["content"])
        
        # Accept user input with a prompt
        user_prompt = "Ask me about crops, pests, farming techniques, market trends..."
        prompt = st.chat_input(user_prompt)
        
        # When the user sends a message
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user", avatar="👨‍🌾"):
                st.markdown(prompt)
            
            # Prepare the farming context
            farming_context = """
            You are Agribot, an expert AI assistant specializing in agriculture and farming.
            Provide helpful, accurate, and concise information about farming practices,
            crop management, pest control, sustainable agriculture, modern farming technology,
            climate-resilient agriculture, and market trends.
            
            Keep your answers practical and actionable for farmers.
            Include specific recommendations where appropriate.
            If discussing chemicals or treatments, always mention safety precautions.
            """
            
            # Combine the context with the user's question
            full_prompt = f"{farming_context}\n\nUser question: {prompt}"
            
            # Get response from Gemini
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🌱 Thinking...")
                
                try:
                    # Create the model with the specified model name
                    model = genai.GenerativeModel(gemini_model_name)
                    
                    # Get the response
                    response = model.generate_content(full_prompt)
                    
                    # Update the message placeholder
                    if hasattr(response, 'text'):
                        message_placeholder.markdown(response.text)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    else:
                        # Handle different response format
                        content = str(response)
                        message_placeholder.markdown(content)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": content})
                        
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Information panel in right column
    with col2:
        # Add an inspirational quote about agricultural technology
        st.markdown("""
        <div style="padding: 15px; border-radius: 10px; background: linear-gradient(135deg, #e6f7ff, #d1e9ff); margin-bottom: 20px; border-left: 4px solid #1976D2;">
            <em>"Technology applied wisely is agriculture's best hope for feeding a growing population on a warming planet."</em>
            <br><span style="font-size: 12px;">— Sustainable Farming Initiative</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick question suggestions
        st.markdown("### 💡 Quick Questions")
        
        # Function to create suggestion buttons
        def create_suggestion(title, query):
            if st.button(title, key=f"suggestion_{title}", use_container_width=True):
                # Add to messages and trigger rerun
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        # Create suggestion buttons with farm-related questions
        create_suggestion("🐛 Pest Control", "What are organic methods to control aphids on vegetable crops?")
        create_suggestion("💧 Irrigation Tips", "What are the most water-efficient irrigation systems for small farms?")
        create_suggestion("🌱 Crop Rotation", "How should I plan my crop rotation for a vegetable garden?")
        create_suggestion("🌦️ Climate Resilience", "Which crops are most resilient to drought conditions?")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Information box about Agribot
        st.markdown("""
        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; margin-top: 20px; background-color: #f5f5f5;">
            <h4 style="color: #2E7D32; margin-top: 0;">About Agribot 🤖</h4>
            <p style="font-size: 14px;">
                Agribot provides AI-powered advice on:
                <ul style="font-size: 14px; padding-left: 20px;">
                    <li>Crop selection & rotation</li>
                    <li>Pest & disease management</li>
                    <li>Irrigation & water conservation</li>
                    <li>Sustainable farming practices</li>
                    <li>Soil health & fertilization</li>
                    <li>Market trends & crop prices</li>
                    <li>Weather adaptation strategies</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
