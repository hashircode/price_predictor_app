import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================================
# 1. ⚙️ APP SETUP AND CACHING
# ===============================================

CAR_DATA_PATH = 'car_data.csv'
MOBILE_DATA_PATH = 'mobile_data.csv' 
HOUSE_DATA_PATH = 'house_data.csv' 
CURRENT_YEAR = 2024
RANDOM_SEED = 42

@st.cache_data
def load_car_data_raw():
    try:
        df = pd.read_csv(CAR_DATA_PATH)
    except FileNotFoundError:
        sample_data = {
            'Year': [2015, 2018, 2012, 2020, 2010, 2019, 2016, 2017, 2021, 2013, 2017, 2014, 2022, 2011, 2019, 2018, 2016, 2020, 2015, 2022],
            'Selling_Price': [550000, 1200000, 300000, 1500000, 180000, 950000, 620000, 780000, 1800000, 400000, 850000, 480000, 2100000, 250000, 1050000, 1300000, 500000, 1600000, 650000, 2200000],
            'Kms_Driven': [50000, 25000, 80000, 15000, 110000, 35000, 45000, 40000, 10000, 95000, 30000, 65000, 5000, 120000, 20000, 22000, 60000, 12000, 75000, 3000],
            'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel'],
            'Transmission': ['Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic'],
            'Owner': ['First Owner', 'First Owner', 'Second Owner', 'First Owner', 'Third Owner', 'First Owner', 'Second Owner', 'First Owner', 'First Owner', 'Second Owner', 'First Owner', 'Second Owner', 'First Owner', 'Third Owner', 'First Owner', 'First Owner', 'Second Owner', 'First Owner', 'Second Owner', 'First Owner']
        }
        df = pd.DataFrame(sample_data)
    return df

@st.cache_data
def load_mobile_data_raw():
    sample_data = {
        'Price_k': [15, 30, 8, 45, 12, 60, 20, 35, 90, 18], 
        'RAM_GB': [4, 6, 3, 8, 4, 12, 6, 8, 16, 4],
        'Storage_GB': [64, 128, 32, 256, 64, 512, 128, 256, 1024, 64],
        'Screen_Inches': [6.0, 6.5, 5.5, 6.7, 6.2, 6.8, 6.4, 6.6, 6.9, 6.1],
        'Battery_mAh': [4000, 4500, 3500, 5000, 4200, 5500, 4800, 5200, 6000, 4100],
        'Brand': ['Samsung', 'Apple', 'Xiaomi', 'Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'OnePlus', 'Samsung', 'Xiaomi']
    }
    df = pd.DataFrame(sample_data)
    return df

@st.cache_data
def load_house_data_raw():
    sample_data = {
        'Price_Lakhs': [50, 120, 35, 80, 200, 65, 90, 150, 40, 110], 
        'Area_SqFt': [1000, 2500, 750, 1500, 4000, 1200, 1800, 3000, 850, 2200],
        'Bedrooms': [2, 4, 1, 3, 5, 2, 3, 4, 2, 3],
        'Bathrooms': [1, 3, 1, 2, 4, 2, 2, 3, 1, 3],
        'Age_Years': [5, 1, 15, 8, 2, 10, 3, 5, 12, 1],
        'Modular_Kitchen': [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        'Lawn_Garden': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        'Locality_Score': [8, 9, 3, 7, 9, 4, 8, 8, 5, 9],
        'Location': ['City Center', 'Suburb', 'Rural', 'City Center', 'Suburb', 'Rural', 'City Center', 'Suburb', 'Rural', 'City Center']
    }
    df = pd.DataFrame(sample_data)
    return df

# ===============================================
# 2. 🧠 MODEL TRAINING FUNCTIONS
# ===============================================

@st.cache_data
def train_car_price_model(raw_data_df):
    df = raw_data_df.copy()
    
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    df['Price_in_Lakhs'] = df['Selling_Price'] / 100000
    df.drop(['Year', 'Selling_Price'], axis=1, inplace=True)

    X = df.drop('Price_in_Lakhs', axis=1)
    y = df['Price_in_Lakhs']
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    accuracy_score = r2_score(y_test, y_predict)
    error_margin = mean_absolute_error(y_test, y_predict)
    
    return model, df, feature_names, accuracy_score, error_margin

@st.cache_data
def train_mobile_price_model(raw_data_df):
    df = raw_data_df.copy()
    
    X = df.drop('Price_k', axis=1)
    y = df['Price_k']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=RANDOM_SEED
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    accuracy_score = r2_score(y_test, y_predict)
    error_margin = mean_absolute_error(y_test, y_predict)
    
    return model, df, feature_names, accuracy_score, error_margin

@st.cache_data
def train_house_price_model(raw_data_df):
    df = raw_data_df.copy()
    
    X = df.drop('Price_Lakhs', axis=1)
    y = df['Price_Lakhs']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    accuracy_score = r2_score(y_test, y_predict)
    error_margin = mean_absolute_error(y_test, y_predict)
    
    return model, df, feature_names, accuracy_score, error_margin

# ===============================================
# 3. 🔮 PREDICTION FUNCTIONS
# ===============================================

def get_car_price_estimate(model, features, car_age, kms_driven, fuel, transmission, owner):
    input_data = pd.DataFrame(0, index=[0], columns=features)
    input_data['Car_Age'] = car_age
    input_data['Kms_Driven'] = kms_driven
    
    if fuel == 'Petrol' and 'Fuel_Type_Petrol' in input_data.columns:
        input_data['Fuel_Type_Petrol'] = 1
    if transmission == 'Manual' and 'Transmission_Manual' in input_data.columns:
        input_data['Transmission_Manual'] = 1
    if owner == 'Second Owner' and 'Owner_Second Owner' in input_data.columns:
        input_data['Owner_Second Owner'] = 1
    elif owner == 'Third Owner' and 'Owner_Third Owner' in input_data.columns:
        input_data['Owner_Third Owner'] = 1
    
    estimated_price = model.predict(input_data)[0]
    return estimated_price

def get_mobile_price_estimate(model, features, ram, storage, screen, battery, brand):
    input_data = pd.DataFrame(0, index=[0], columns=features)
    
    input_data['RAM_GB'] = ram
    input_data['Storage_GB'] = storage
    input_data['Screen_Inches'] = screen
    input_data['Battery_mAh'] = battery
    
    brand_column = f'Brand_{brand}'
    if brand_column in input_data.columns:
        input_data[brand_column] = 1
        
    estimated_price = model.predict(input_data)[0]
    return estimated_price

def get_house_price_estimate(model, features, area, num_bedrooms, num_bathrooms, age, has_kitchen, has_lawn, location):
    input_data = pd.DataFrame(0, index=[0], columns=features)
    
    input_data['Area_SqFt'] = area
    input_data['Bedrooms'] = num_bedrooms
    input_data['Bathrooms'] = num_bathrooms
    input_data['Age_Years'] = age
    
    if 'Locality_Score' in input_data.columns:
        input_data['Locality_Score'] = 8 
    
    if 'Modular_Kitchen' in input_data.columns: 
        input_data['Modular_Kitchen'] = 1 if has_kitchen else 0
    if 'Lawn_Garden' in input_data.columns: 
        input_data['Lawn_Garden'] = 1 if has_lawn else 0
    
    location_column = f'Location_{location}'
    if location_column in input_data.columns:
        input_data[location_column] = 1
        
    estimated_price = model.predict(input_data)[0]
    return estimated_price

# ===============================================
# 4. 🖥️ MAIN STREAMLIT APPLICATION LOGIC
# ===============================================
def main():
    st.set_page_config(
        page_title="Price Predictor Multi App"
    )

    st.title("💰 Multi-Product Price Estimator")
    st.markdown("""
    Select an option below to get a price **estimate** for either a **Used Car**, a **Mobile Phone**, or a **House**.
    """)
    
    with st.sidebar:
        st.header("Model Setup & Selection")
        
        car_raw_df = load_car_data_raw()
        car_model, car_df, car_features, car_accuracy, car_error = train_car_price_model(car_raw_df)
        
        mobile_raw_df = load_mobile_data_raw()
        mobile_model, mobile_df, mobile_features, mobile_accuracy, mobile_error = train_mobile_price_model(mobile_raw_df)

        house_raw_df = load_house_data_raw()
        house_model, house_df, house_features, house_accuracy, house_error = train_house_price_model(house_raw_df)

        st.success("✅ All prediction models are ready.")
        
        predictor_choice = st.radio(
            "Select Predictor:",
            ('Used Car Price', 'Mobile Phone Price', 'House Price'),
            index=0
        )

    st.divider()

    # 3 car predictor
    if predictor_choice == 'Used Car Price':
        st.header("🚗 Used Car Price Estimator")
        st.markdown("Enter details to predict the selling price in **Lakhs** (₹100,000s).")
        
        fuel_opts = car_df['Fuel_Type'].unique()
        trans_opts = car_df['Transmission'].unique()
        owner_opts = car_df['Owner'].unique()

        with st.form("car_estimation_form"):
            col1, col2 = st.columns(2)
            with col1:
                car_age = st.slider("Age of the Car (Years)", min_value=car_df['Car_Age'].min(), max_value=car_df['Car_Age'].max(), value=5)
                fuel_type = st.selectbox("Fuel Type", fuel_opts)
                owner = st.selectbox("Current Owner Status", owner_opts)
            with col2:
                kms_driven = st.number_input("Kilometers Driven (KMs)", min_value=1000, max_value=200000, value=40000, step=1000)
                transmission = st.selectbox("Gearbox Type", trans_opts)
                st.write("") 
            
            estimate_button = st.form_submit_button("Get Car Price Estimate", type="primary")

            if estimate_button:
                estimated_lakhs = get_car_price_estimate(
                    car_model, car_features, car_age, kms_driven, fuel_type, transmission, owner
                )
                
                st.success(f"**Your Car Price Estimate:**")
                st.metric(label="Estimated Amount (in Lakhs)", value=f"{estimated_lakhs:,.2f} Lakhs")
                st.caption(f"This is approximately ₹{estimated_lakhs * 100000:,.0f}")

    # 2 mobile predictor
    elif predictor_choice == 'Mobile Phone Price':
        st.header("📱 Mobile Phone Price Estimator")
        st.markdown("Enter specifications to predict the price in **Thousands (k)**.")
        
        brand_opts = mobile_df['Brand'].unique()

        with st.form("mobile_estimation_form"):
            col1, col2 = st.columns(2)
            with col1:
                brand = st.selectbox("Brand Name", brand_opts)
                ram = st.selectbox("RAM (Gigabytes)", [4, 6, 8, 12, 16], index=2)
                battery = st.number_input("Battery Capacity (mAh)", min_value=3000, max_value=7000, value=4500, step=100)
            with col2:
                storage = st.selectbox("Storage (Gigabytes)", [64, 128, 256, 512, 1024], index=1)
                screen = st.slider("Screen Size (Inches)", min_value=5.0, max_value=7.0, value=6.5, step=0.1)
                st.write("") 
            
            estimate_button = st.form_submit_button("Get Mobile Price Estimate", type="primary")

            if estimate_button:
                estimated_k = get_mobile_price_estimate(
                    mobile_model, mobile_features, ram, storage, screen, battery, brand
                )
                
                st.success(f"**Your Mobile Price Estimate:**")
                st.metric(label="Estimated Amount (in Thousands)", value=f"{estimated_k:,.2f} k")
                st.caption(f"This is approximately ₹{estimated_k * 1000:,.0f}")

    # 1 house predictor
    elif predictor_choice == 'House Price':
        st.header("🏠 House Price Estimator")
        st.markdown("Enter property details to predict the selling price in **Lakhs** (₹100,000s).")
        
        location_opts = house_df['Location'].unique()

        with st.form("house_estimation_form"):
            st.subheader("🏡 Property Layout")
            col1, col2 = st.columns(2)
            with col1:
                area = st.number_input("Area (Square Feet)", min_value=500, max_value=10000, value=1500, step=100)
                num_bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5], index=2)
                age = st.slider("Property Age (Years)", min_value=0, max_value=50, value=5)
            with col2:
                num_bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4], index=1)
                has_modular_kitchen = st.checkbox("Has Modular Kitchen", value=True)
                has_lawn_garden = st.checkbox("Has Lawn/Garden", value=False)
            
            st.subheader("📍 Location Details")
            col3, col4 = st.columns(2)
            with col3:
                location = st.selectbox("Location Type", location_opts)
            # col4 is empty as Locality Score input was removed
            
            estimate_button = st.form_submit_button("Get House Price Estimate", type="primary")

            if estimate_button:
                estimated_lakhs = get_house_price_estimate(
                    house_model, 
                    house_features, 
                    area, num_bedrooms, num_bathrooms, age, 
                    has_modular_kitchen, has_lawn_garden, 
                    location
                )
                
                st.success(f"**Your House Price Estimate:**")
                display_lakhs = max(0.01, estimated_lakhs)
                st.metric(label="Estimated Amount (in Lakhs)", value=f"{display_lakhs:,.2f} Lakhs")
                st.caption(f"This is approximately ₹{display_lakhs * 100000:,.0f}")
            
    st.divider()
    st.markdown("---")

if __name__ == '__main__':
    main()