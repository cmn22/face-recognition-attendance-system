import streamlit as st

# Only set page config if this is the main file being run
if __name__ == "__main__":
    st.set_page_config(page_title='Attendance System', layout='wide')

st.header('Attendance System using Face Recognition')


with st.spinner("Loading Models and Conneting to Redis db ..."):
    import face_rec
    
st.success('Model loaded sucesfully')
st.success('Redis db sucessfully connected')