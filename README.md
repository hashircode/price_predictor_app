Multi-Product Price Estimator App

This project is a modular web application built with Streamlit that provides price estimates for three different product categories: Used Cars, Mobile Phones, and Houses. It uses distinct Linear Regression models for each product, demonstrating an efficient, data-driven approach to price prediction.

![img alt](https://github.com/hashircode/price_predictor_app/blob/main/predictor.png)

## Architecture 
![Architecture](https://github.com/hashircode/price_predictor_app/blob/main/arcitecture.png)

✨ Features


Three Estimators in One: Predicts prices for Used Cars, Mobile Phones, and Houses from a single application interface.

Modular ML Approach: Uses separate Linear Regression models for each product category for specialized predictions.

Indian Market Context: Provides price outputs scaled for the Indian market: Lakhs (₹100,000s) for cars and houses, and Thousands (k) for mobile phones.

Efficient Caching: Utilizes Streamlit's @st.cache_data decorator to ensure data loading and model training are performed only once, maximizing application speed.

🛠️ Technologies Used

User Interface	: Streamlit

Backend / Data	: Python / Pandas

Machine Learning : Scikit-learn


🏗️ Technical Flow
Start: When you run the app, sample data is loaded for all three products.

Training: The data is processed (e.g., text categories like 'Petrol' are converted to numbers), and three separate Linear Regression models are quickly trained.

Prediction: When you enter your item's details (e.g., Car Age, RAM), the app feeds those numbers to the correct model to instantly return the estimated price.


Requirments:

Python installed on your system.

streamlit 

pandas 

Hpw to run the app:

1 : Go to cmd 
2:write cd"specific path to app"
3:write streamlit run and name of given app
4:Access the Dashboard
The application will open automatically in your browser (typically at http://localhost:8501).

Use the sidebar radio buttons to switch between Car, Mobile, and House prediction modes.

Enter the required features for the chosen product.

Click the "Get Estimate" button to see the result.

## Authors

* **[Muhammad Hashir](https://github.com/account)**
