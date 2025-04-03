import streamlit as st
from Home import face_rec
import pandas as pd
from redis import Redis, RedisError

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

# ==============================================
# CONSTANTS & INITIALIZATION
# ==============================================
REDIS_LOG_KEY = 'attendance:logs'
redis_conn = face_rec.get_redis_connection()

# ==============================================
# DATA LOADING FUNCTIONS
# ==============================================
def load_logs(log_key: str, end: int = -1) -> list:
    """
    Retrieve attendance logs from Redis
    
    Args:
        log_key: Redis key for logs
        end: Index of last item to retrieve (-1 for all)
    
    Returns:
        List of log entries or empty list on error
    """
    try:
        logs_list = redis_conn.lrange(log_key, start=0, end=end)
        return logs_list if logs_list else []
    except RedisError as e:
        st.error(f"Redis connection error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error loading logs: {e}")
        return []

def process_attendance_logs(logs: list) -> pd.DataFrame:
    """
    Process raw log data into formatted attendance report
    
    Args:
        logs: List of raw log entries from Redis
    
    Returns:
        DataFrame with processed attendance data
    """
    if not logs:
        return pd.DataFrame()
    
    # Convert and clean log data
    logs_decoded = [log.decode('utf-8') for log in logs]
    logs_split = [log.split('@') for log in logs_decoded if len(log.split('@')) == 3]
    
    if not logs_split:
        return pd.DataFrame()
    
    # Create DataFrame and process timestamps
    df = pd.DataFrame(logs_split, columns=['Name', 'Role', 'Timestamp'])
    df = df.apply(lambda x: x.str.strip())
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    
    # Generate attendance statistics
    report = df.groupby(['Date', 'Name', 'Role']).agg(
        Check_In=('Timestamp', 'min'),
        Check_Out=('Timestamp', 'max')
    ).reset_index()
    
    # Format time duration
    report['Duration'] = report['Check_Out'] - report['Check_In']
    report['Duration'] = report['Duration'].apply(_format_timedelta)
    
    return report

def _format_timedelta(td: pd.Timedelta) -> str:
    """Helper to format timedelta as human-readable string"""
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes = remainder // 60
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"

# ==============================================
# PAGE LAYOUT - TABBED INTERFACE
# ==============================================
tab1, tab2, tab3 = st.tabs(['Registered Users', 'Raw Logs', 'Attendance Report'])

with tab1:
    st.subheader('Registered Users')
    if st.button('Refresh User Data', key='refresh_users'):
        with st.spinner('Loading user data...'):
            user_data = face_rec.retrive_data(name='academy:register')
            st.dataframe(user_data[['Name', 'Role']])

with tab2:
    st.subheader('Raw Attendance Logs')
    if st.button('Refresh Log Data', key='refresh_logs'):
        raw_logs = load_logs(REDIS_LOG_KEY)
        st.write(raw_logs)

with tab3:
    st.subheader('Processed Attendance Report')
    
    # Load and process attendance data
    attendance_logs = load_logs(REDIS_LOG_KEY)
    attendance_report = process_attendance_logs(attendance_logs)
    
    if attendance_report.empty:
        st.warning("No valid attendance data found")
    else:
        st.dataframe(attendance_report)