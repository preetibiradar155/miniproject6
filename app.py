import pickle
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Neural Network-based house price estimation")

# Load model
model = load_model("house_price_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.subheader("Enter House Details")

area = st.number_input("Area (sq ft)", 100, 10000, 1500, step=100)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
age = st.number_input("Age (years)", 0, 100, 5)

if st.button("Predict Price 🏠", type="primary", use_container_width=True):
    df = pd.DataFrame([[area, bedrooms, age]], columns=["Area", "Bedrooms", "Age"])
    scaled = scaler.transform(df)
    price = float(model.predict(scaled, verbose=0)[0][0])

    st.divider()
    
    # --- IMAGE LOGIC ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Price ke hisab se image
        if price < 40:
            # Small house
            st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=500", caption="Budget House")
        elif price < 80:
            # Medium house
            st.image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=500", caption="Standard House")
        else:
            # Luxury house
            st.image("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=500", caption="Luxury House")
    
    with col2:
        st.metric("Estimated Price", f"₹ {price:.2f} lakhs")
        st.write(f"**Area:** {area} sq ft")
        st.write(f"**Bedrooms:** {bedrooms}")
        st.write(f"**Age:** {age} years")
        
        if price < 40:
            st.info("💰 Budget Friendly")
        elif price < 80:
            st.success("🏡 Good Value")
        else:
            st.warning("✨ Premium Property")

st.divider()
st.caption("ML Mini Project | House Price with Image")