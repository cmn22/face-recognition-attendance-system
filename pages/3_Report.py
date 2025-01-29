import streamlit as st

st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

from Home import face_rec
import pandas as pd
from redis import Redis, RedisError

# Retrieve logs data and show in Report.py
# Extract data from Redis list
name = 'attendance:logs'

def load_logs(name, end=-1):
    try:
        logs_list = redis_conn.lrange(name, start=0, end=end)
        return logs_list if logs_list else []
    except RedisError as e:
        st.error(f"Error connecting to Redis: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []
redis_conn = face_rec.get_redis_connection()
# Tabs to show the info
tab1, tab2, tab3 = st.tabs(['Registered Data', 'Logs', "Attendance Report"])

with tab1:
    if st.button('Refresh Data'):
        # Retrieve the data from Redis Database
        with st.spinner('Retrieving Data from Redis DB ...'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))

with tab3:
    
    st.subheader('Attendance Report')

    logs_list = load_logs(name=name)
        
    if not logs_list:
            st.warning("No attendance logs found.")
    else:
            # Convert bytes to string
            logs_list_string = [log.decode('utf-8') for log in logs_list]
            
            # Split string into Name, Role, and Timestamp
            logs_nested_list = [log.split('@') for log in logs_list_string]
            
            # Ensure data is in the correct format
            if len(logs_nested_list) > 0 and len(logs_nested_list[0]) == 3:
                logs_df = pd.DataFrame(logs_nested_list, columns=['Name', 'Role', 'Timestamp'])
                
                # Clean and process data
                logs_df = logs_df.apply(lambda x: x.str.strip())
                logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'], errors='coerce')
                logs_df = logs_df.dropna(subset=['Timestamp'])
                logs_df['Date'] = logs_df['Timestamp'].dt.date
                
                # Generate attendance report
                report_df = logs_df.groupby(['Date', 'Name', 'Role']).agg(
                    In_time=pd.NamedAgg(column='Timestamp', aggfunc='min'),
                    Out_time=pd.NamedAgg(column='Timestamp', aggfunc='max')
                ).reset_index()
                
                # Calculate and format duration
                report_df['Duration'] = report_df['Out_time'] - report_df['In_time']
                
                def format_duration(td):
                    days = td.days
                    hours, remainder = divmod(td.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    return f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"
                
                report_df['Duration'] = report_df['Duration'].apply(format_duration)
                
                # Display the report
                st.dataframe(report_df)
            else:
                st.warning("No valid logs found for processing.")
