import streamlit as st

# ------------------------------
# Page Configuration
# ------------------------------
# Only set page config when running as main script to avoid Streamlit warnings
if __name__ == "__main__":
    st.set_page_config(
        page_title='Attendance System',  # Set browser tab title
        layout='wide'                    # Use full page width
    )

# ------------------------------
# Application Header
# ------------------------------
# Main title for the application
st.header('Attendance System using Face Recognition')

# ------------------------------
# System Initialization
# ------------------------------
# Show loading spinner while importing and setting up components
with st.spinner("Loading Models and Connecting to Redis db..."):
    # Import face recognition module (contains Redis connection and ML models)
    import face_rec
    
# Display success messages after initialization
st.success('Model loaded successfully')  # Confirmation of model loading
st.success('Redis db successfully connected')  # Confirmation of database connection