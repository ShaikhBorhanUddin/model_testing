
import streamlit as st
import pandas as pd
import joblib
import xgboost # Required to load XGBoost model
import folium
from streamlit_folium import st_folium

# --- File Paths ---
model_path = 'xgboost_model.pkl'
feature_names_path = 'feature_names.pkl'
unique_categorical_values_path = 'unique_categorical_values.pkl'
combined_location_mapping_path = 'combined_location_mapping.pkl'
location_coordinates_mapping_path = 'location_coordinates_mapping.pkl'

# --- Load Model and Artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    return model

@st.cache_resource
def load_artifacts():
    feature_names = joblib.load(feature_names_path)
    unique_categorical_values = joblib.load(unique_categorical_values_path)
    combined_location_mapping = joblib.load(combined_location_mapping_path)
    location_coordinates_mapping = joblib.load(location_coordinates_mapping_path)
    return feature_names, unique_categorical_values, combined_location_mapping, location_coordinates_mapping

model = load_model()
feature_names, unique_categorical_values, combined_location_mapping, location_coordinates_mapping = load_artifacts()

# --- Define Mappings and Initial Defaults ---
borough_names_map = {
    1: 'Manhattan',
    2: 'Bronx',
    3: 'Brooklyn',
    4: 'Queens',
    5: 'Staten Island'
}

initial_borough_id = 1
initial_neighborhood = 'ALPHABET CITY'
initial_zip_code = 10009
initial_latitude = 40.7128 # Default NYC latitude
initial_longitude = -74.0060 # Default NYC longitude

# Helper function to update lat/lon based on current selections
def update_lat_lon():
    key = (st.session_state.selected_borough_id, st.session_state.selected_neighborhood_name, st.session_state.selected_zip_code)
    if key in location_coordinates_mapping:
        st.session_state.selected_latitude, st.session_state.selected_longitude = location_coordinates_mapping[key]
    else:
        # Fallback to a default or previous value if lookup fails
        # For now, let's keep the last valid or initial default
        st.session_state.selected_latitude = initial_latitude
        st.session_state.selected_longitude = initial_longitude

# --- Helper Function for Selectbox Index ---
def get_select_box_index(options_list, selected_value):
    try:
        return options_list.index(selected_value)
    except ValueError:
        return 0 # Default to first option if selected_value is not in the list

# --- Session State Callbacks ---
def on_borough_change():
    st.session_state.selected_borough_id = st.session_state.borough_select
    # Reset neighborhood and zip code to valid defaults for the new borough
    new_neighborhood_options = combined_location_mapping['borough_to_neighborhoods'].get(st.session_state.selected_borough_id, [])
    st.session_state.selected_neighborhood_name = new_neighborhood_options[0] if new_neighborhood_options else ''

    new_zip_options = sorted(list(set(combined_location_mapping['borough_to_zipcodes'].get(st.session_state.selected_borough_id, [])) &
                                  set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))))
    st.session_state.selected_zip_code = new_zip_options[0] if new_zip_options else 0
    update_lat_lon()

def on_neighborhood_change():
    st.session_state.selected_neighborhood_name = st.session_state.neighborhood_select
    # Reset zip code to valid default for the current borough and new neighborhood
    new_zip_options = sorted(list(set(combined_location_mapping['borough_to_zipcodes'].get(st.session_state.selected_borough_id, [])) &
                                  set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))))
    st.session_state.selected_zip_code = new_zip_options[0] if new_zip_options else 0
    update_lat_lon()

def on_zip_code_change():
    st.session_state.selected_zip_code = st.session_state.zip_code_select
    update_lat_lon()

# --- Initialize Session State ---
# Ensure initial values are consistent with the loaded mapping
if 'selected_borough_id' not in st.session_state:
    st.session_state.selected_borough_id = initial_borough_id

if 'selected_neighborhood_name' not in st.session_state:
    valid_initial_neighborhoods = combined_location_mapping['borough_to_neighborhoods'].get(initial_borough_id, [])
    if initial_neighborhood in valid_initial_neighborhoods:
        st.session_state.selected_neighborhood_name = initial_neighborhood
    elif valid_initial_neighborhoods:
        st.session_state.selected_neighborhood_name = valid_initial_neighborhoods[0]
    else:
        st.session_state.selected_neighborhood_name = '' # Fallback

if 'selected_zip_code' not in st.session_state:
    valid_initial_zips_borough = set(combined_location_mapping['borough_to_zipcodes'].get(initial_borough_id, []))
    valid_initial_zips_neighborhood = set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))
    valid_initial_zips = sorted(list(valid_initial_zips_borough & valid_initial_zips_neighborhood))
    if initial_zip_code in valid_initial_zips:
        st.session_state.selected_zip_code = initial_zip_code
    elif valid_initial_zips:
        st.session_state.selected_zip_code = valid_initial_zips[0]
    else:
        st.session_state.selected_zip_code = 0 # Fallback

# Set initial lat/lon based on the initialized location selections
if 'selected_latitude' not in st.session_state or 'selected_longitude' not in st.session_state:
    update_lat_lon()


# --- Streamlit App ---
st.title('NYC Property Sale Price Prediction')
st.write('Enter the property details to predict its adjusted sale price.')

st.header('Location Details')

# BOROUGH Selectbox
selected_borough_id_display = st.selectbox(
    'BOROUGH',
    options=list(borough_names_map.keys()),
    index=get_select_box_index(list(borough_names_map.keys()), st.session_state.selected_borough_id),
    format_func=lambda x: borough_names_map[x],
    key='borough_select',
    on_change=on_borough_change,
    help='Borough (1-Manhattan, 2-Bronx, 3-Brooklyn, 4-Queens, 5-Staten Island)'
)

# NEIGHBORHOOD Selectbox
current_neighborhood_options = combined_location_mapping['borough_to_neighborhoods'].get(st.session_state.selected_borough_id, [])
if not current_neighborhood_options:
    current_neighborhood_options = ['No neighborhoods found'] # Placeholder

selected_neighborhood_display = st.selectbox(
    'NEIGHBORHOOD',
    options=current_neighborhood_options,
    index=get_select_box_index(current_neighborhood_options, st.session_state.selected_neighborhood_name),
    key='neighborhood_select',
    on_change=on_neighborhood_change,
    help='Neighborhood of the property'
)

# ZIP CODE Selectbox
current_zip_options_borough = set(combined_location_mapping['borough_to_zipcodes'].get(st.session_state.selected_borough_id, []))
current_zip_options_neighborhood = set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))
current_zip_options = sorted(list(current_zip_options_borough & current_zip_options_neighborhood))

if not current_zip_options:
    current_zip_options = [0] # Placeholder

selected_zip_code_display = st.selectbox(
    'ZIP CODE',
    options=current_zip_options,
    index=get_select_box_index(current_zip_options, st.session_state.selected_zip_code),
    key='zip_code_select',
    on_change=on_zip_code_change,
    help='Property ZIP code'
)

st.header('Property Details')

block = st.number_input('BLOCK', min_value=1, value=374, help='Block number')
lot = st.number_input('LOT', min_value=1, value=46, help='Lot number')
residential_units = st.number_input('RESIDENTIAL UNITS', min_value=0, value=1, help='Number of residential units')
commercial_units = st.number_input('COMMERCIAL UNITS', min_value=0, value=0, help='Number of commercial units')
land_square_feet = st.number_input('LAND SQUARE FEET', min_value=0.0, value=2116.0, help='Land area in square feet')
gross_square_feet = st.number_input('GROSS SQUARE FEET', min_value=0.0, value=4400.0, help='Gross building area in square feet')
year_built = st.number_input('YEAR BUILT', min_value=1700, max_value=2024, value=1900, help='Year property was built')
sale_year = st.number_input('SALE YEAR', min_value=2000, max_value=2024, value=2022, help='Year of sale')
sale_month = st.number_input('SALE MONTH', min_value=1, max_value=12, value=9, help='Month of sale (1-12)')

# Latitude and Longitude inputs for map visualization (now dynamically updated)
st.subheader('Property Location (for Map Visualization)')
# Display current lat/lon from session state, user can still override if needed
latitude = st.number_input('LATITUDE', min_value=-90.0, max_value=90.0, value=float(st.session_state.selected_latitude), format="%.6f", help='Latitude coordinate of the property')
longitude = st.number_input('LONGITUDE', min_value=-180.0, max_value=180.0, value=float(st.session_state.selected_longitude), format="%.6f", help='Longitude coordinate of the property')

# Store current lat/lon in session state to maintain user edits if any
st.session_state.selected_latitude = latitude
st.session_state.selected_longitude = longitude

# Display map
st.subheader('Property Location Map')
# Create a Folium map centered at the entered coordinates
m = folium.Map(location=[st.session_state.selected_latitude, st.session_state.selected_longitude], zoom_start=15)
# Add a marker for the property
folium.Marker([st.session_state.selected_latitude, st.session_state.selected_longitude], popup='Property Location').add_to(m)
# Display the map in Streamlit
st_folium(m, width=700, height=500)


st.header('Building Characteristics')
selected_building_class = st.selectbox(
    'BUILDING CLASS AT TIME OF SALE',
    options=unique_categorical_values['BUILDING CLASS AT TIME OF SALE'],
    index=get_select_box_index(unique_categorical_values['BUILDING CLASS AT TIME OF SALE'], 'A4'),
    help='Building class at time of sale'
)
selected_building_category = st.selectbox(
    'BUILDING CLASS CATEGORY DESCRIPTION',
    options=unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'],
    index=get_select_box_index(unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'], 'ONE FAMILY DWELLINGS'),
    help='Category of the building class'
)

# Create a button to make predictions
if st.button('Predict Sale Price'):
    # Prepare the input data as a dictionary (excluding latitude and longitude for model prediction)
    input_data_dict = {
        'BOROUGH': st.session_state.selected_borough_id,
        'BLOCK': block,
        'LOT': lot,
        'ZIP CODE': st.session_state.selected_zip_code,
        'RESIDENTIAL UNITS': residential_units,
        'COMMERCIAL UNITS': commercial_units,
        'LAND SQUARE FEET': land_square_feet,
        'GROSS SQUARE FEET': gross_square_feet,
        'YEAR BUILT': year_built,
        'SALE YEAR': sale_year,
        'SALE MONTH': sale_month,
        'NEIGHBORHOOD': st.session_state.selected_neighborhood_name,
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
    final_input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    # Populate with the one-hot encoded user input
    for col in input_df_ohe.columns:
        if col in final_input_df.columns:
            final_input_df[col] = input_df_ohe[col].values

    # Ensure the order of columns matches X_train
    final_input_df = final_input_df[feature_names]

    # Make prediction
    prediction = model.predict(final_input_df)[0]

    st.success(f'Predicted Adjusted Sale Price: ${prediction:,.2f}')
