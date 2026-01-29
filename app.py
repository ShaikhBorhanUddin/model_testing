
import streamlit as st
import pandas as pd
import joblib
import xgboost # Required to load XGBoost model

# --- File Paths ---
model_path = 'nyc_property_sales_model.pkl'
feature_names_path = 'feature_names.pkl'
unique_categorical_values_path = 'unique_categorical_values.pkl'

# --- Load Model and Artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    return model

@st.cache_resource
def load_artifacts():
    feature_names = joblib.load(feature_names_path)
    unique_categorical_values = joblib.load(unique_categorical_values_path)
    return feature_names, unique_categorical_values

model = load_model()
feature_names, unique_categorical_values = load_artifacts()

# --- Streamlit App --- 
st.title('NYC Property Sale Price Prediction')
st.write('Enter the property details to predict its adjusted sale price.')

# Input fields for numerical features
st.header('Numerical Features')
borough = st.number_input('BOROUGH', min_value=1, max_value=5, value=1, help='Borough (1-Manhattan, 2-Bronx, 3-Brooklyn, 4-Queens, 5-Staten Island)')
block = st.number_input('BLOCK', min_value=1, value=374, help='Block number')
lot = st.number_input('LOT', min_value=1, value=46, help='Lot number')
zip_code = st.number_input('ZIP CODE', min_value=10001, max_value=11697, value=10009, help='Property ZIP code')
residential_units = st.number_input('RESIDENTIAL UNITS', min_value=0, value=1, help='Number of residential units')
commercial_units = st.number_input('COMMERCIAL UNITS', min_value=0, value=0, help='Number of commercial units')
land_square_feet = st.number_input('LAND SQUARE FEET', min_value=0.0, value=2116.0, help='Land area in square feet')
gross_square_feet = st.number_input('GROSS SQUARE FEET', min_value=0.0, value=4400.0, help='Gross building area in square feet')
year_built = st.number_input('YEAR BUILT', min_value=1700, max_value=2024, value=1900, help='Year property was built')
sale_year = st.number_input('SALE YEAR', min_value=2000, max_value=2024, value=2022, help='Year of sale')
sale_month = st.number_input('SALE MONTH', min_value=1, max_value=12, value=9, help='Month of sale (1-12)')

# Input fields for categorical features using selectbox
st.header('Categorical Features')
selected_neighborhood = st.selectbox(
    'NEIGHBORHOOD',
    options=unique_categorical_values['NEIGHBORHOOD'],
    index=unique_categorical_values['NEIGHBORHOOD'].index('ALPHABET CITY') if 'ALPHABET CITY' in unique_categorical_values['NEIGHBORHOOD'] else 0,
    help='Neighborhood of the property'
)
selected_building_class = st.selectbox(
    'BUILDING CLASS AT TIME OF SALE',
    options=unique_categorical_values['BUILDING CLASS AT TIME OF SALE'],
    index=unique_categorical_values['BUILDING CLASS AT TIME OF SALE'].index('A4') if 'A4' in unique_categorical_values['BUILDING CLASS AT TIME OF SALE'] else 0,
    help='Building class at time of sale'
)
selected_building_category = st.selectbox(
    'BUILDING CLASS CATEGORY DESCRIPTION',
    options=unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'],
    index=unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'].index('ONE FAMILY DWELLINGS') if 'ONE FAMILY DWELLINGS' in unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'] else 0,
    help='Category of the building class'
)

# Create a button to make predictions
if st.button('Predict Sale Price'):
    # Prepare the input data as a dictionary
    input_data_dict = {
        'BOROUGH': borough,
        'BLOCK': block,
        'LOT': lot,
        'ZIP CODE': zip_code,
        'RESIDENTIAL UNITS': residential_units,
        'COMMERCIAL UNITS': commercial_units,
        'LAND SQUARE FEET': land_square_feet,
        'GROSS SQUARE FEET': gross_square_feet,
        'YEAR BUILT': year_built,
        'SALE YEAR': sale_year,
        'SALE MONTH': sale_month,
        'NEIGHBORHOOD': selected_neighborhood,
        'BUILDING CLASS AT TIME OF SALE': selected_building_class,
        'BUILDING CLASS CATEGORY DESCRIPTION': selected_building_category
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data_dict])

    # Apply one-hot encoding to categorical columns (matching training)
    col_to_ohe = ['NEIGHBORHOOD', 'BUILDING CLASS AT TIME OF SALE', 'BUILDING CLASS CATEGORY DESCRIPTION']
    input_df_ohe = pd.get_dummies(input_df, columns=col_to_ohe, drop_first=True)

    # Align columns with the training data's feature names
    # Create a DataFrame with all expected feature_names and fill with 0
    final_input_df = pd.DataFrame(columns=feature_names)
    # Populate with the one-hot encoded user input
    for col in input_df_ohe.columns:
        if col in final_input_df.columns:
            final_input_df[col] = input_df_ohe[col]
    # Fill any remaining NaNs (for features not present in user input's OHE columns) with 0
    final_input_df = final_input_df.fillna(0)

    # Ensure the order of columns matches X_train
    final_input_df = final_input_df[feature_names]

    # Make prediction
    prediction = model.predict(final_input_df)[0]

    st.success(f'Predicted Adjusted Sale Price: ${prediction:,.2f}')
