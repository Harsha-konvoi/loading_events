import os
import pandas as pd
import pytz
import clickhouse_connect
from datetime import datetime, timedelta


def initialize_clickhouse_client():
    """Initialize ClickHouse client with environment variables."""
    clickhouse_url = os.environ.get("CLICKHOUSE_URL", "")
    _parsed = clickhouse_url.replace("https://", "").replace("http://", "").split(":")
    host = _parsed[0] if _parsed[0] else os.environ.get("CLICKHOUSE_HOST", "localhost")
    port = int(_parsed[1]) if len(_parsed) > 1 else int(os.environ.get("CLICKHOUSE_PORT", "8443"))
    secure = clickhouse_url.startswith("https")
    user = os.environ.get("CLICKHOUSE_USER", "default")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")

    if not all([clickhouse_url, host, port, user, password]):
        raise ValueError("Missing required ClickHouse environment variables")

    ch_client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        secure=secure,
    )
    print("ClickHouse client initialized successfully")
    return ch_client


def adjust_query_hour(timezone='Europe/Berlin'):
    """Adjust query time for the previous hour with timezone conversion."""
    try:
        tz = pytz.timezone(timezone)

        now_berlin = datetime.now(tz)
        current_hour_start = now_berlin.replace(minute=0, second=0, microsecond=0)
        previous_hour_start = current_hour_start - timedelta(hours=1)
        previous_hour_end = current_hour_start

        adjusted_start = previous_hour_start.astimezone(pytz.utc)
        adjusted_end = previous_hour_end.astimezone(pytz.utc)

        return adjusted_start.strftime('%Y-%m-%dT%H:%M:%SZ'), adjusted_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        raise


def query_hourly_data(truck, start_datetime, end_datetime, ch_client, table="shock_sensor"):
    """
    Query shock sensor data from ClickHouse for a specific hour range.

    Parameters
    ----------
    truck : str              – e.g. "Lindt_1"
    start_datetime : str or datetime – start of range (Berlin time or naive)
    end_datetime : str or datetime   – end of range (Berlin time or naive)
    ch_client                – clickhouse_connect client
    table : str              – table name in iot_data database
    """
    try:
        tz = pytz.timezone('Europe/Berlin')

        if isinstance(start_datetime, str):
            start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
        if isinstance(end_datetime, str):
            end_datetime = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")

        if start_datetime.tzinfo is None:
            start_utc = tz.localize(start_datetime).astimezone(pytz.utc)
        else:
            start_utc = start_datetime.astimezone(pytz.utc)

        if end_datetime.tzinfo is None:
            end_utc = tz.localize(end_datetime).astimezone(pytz.utc)
        else:
            end_utc = end_datetime.astimezone(pytz.utc)

        start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%S')

        query = f"""
        SELECT
            timestamp AS Time,
            client_id AS Client_ID,
            accel_x,
            accel_y,
            accel_z,
            label
        FROM iot_data.{table}
        WHERE client_id = %(truck)s
          AND timestamp >= toDateTime64(%(start)s, 9)
          AND timestamp <  toDateTime64(%(stop)s, 9)
        ORDER BY timestamp
        """

        df = ch_client.query_df(
            query,
            parameters={
                "truck": truck,
                "start": start_str,
                "stop": end_str,
            },
        )

        if not df.empty:
            df['Time'] = pd.to_datetime(df['Time'])
            if not isinstance(df['Time'].dtype, pd.DatetimeTZDtype):
                df['Time'] = df['Time'].dt.tz_localize('UTC')

            df['Time'] = df['Time'].dt.tz_convert('Europe/Berlin')

            df = df.sort_values(by='Time').reset_index(drop=True)
            df['Client_ID'] = truck

            return df
        else:
            return pd.DataFrame(columns=['Time', 'Client_ID', 'accel_x', 'accel_y', 'accel_z', 'label'])

    except Exception as e:
        print(f"Error querying data for {truck}: {e}")
        return pd.DataFrame(columns=['Time', 'Client_ID', 'accel_x', 'accel_y', 'accel_z', 'label'])

def analyze_loading_with_continuity_check(df_predict, chunk_start_time, overlap_minutes=5, previous_session_info=None):
    """
    Analyze loading events with awareness of previous session state
    """
    from utilities import find_loading_sessions
    loading_sessions = find_loading_sessions(df_predict)
    
    if not loading_sessions:
        return pd.DataFrame(columns=['Client_ID', 'start_time', 'end_time', 'pattern', 'event', 'pallet_count']), None
    
    client_id = df_predict['Client_ID'].iloc[0] if 'Client_ID' in df_predict.columns and not df_predict.empty else None
    
    if previous_session_info and previous_session_info['ended_with_loading']:
        overlap_start = chunk_start_time - timedelta(minutes=overlap_minutes)
        overlap_region = df_predict[
            (df_predict['Time'] >= overlap_start) & 
            (df_predict['Time'] < chunk_start_time)
        ]
        
        has_loading_in_overlap = (overlap_region['pattern'] == 'loading').any() if not overlap_region.empty else False
        
        if has_loading_in_overlap and loading_sessions:
            first_session = loading_sessions[0]
            
            time_gap_seconds = (first_session['start_time'] - previous_session_info['last_loading_end']).total_seconds()
            
            abs_gap_seconds = abs(time_gap_seconds)
            
            if abs_gap_seconds <= 1800:  # 30 minutes
                first_session['is_continuation'] = True
                first_session['previous_session_end'] = previous_session_info['last_loading_end']
                print(f"Loading session continuation detected. Gap: {time_gap_seconds/60:.1f} minutes (abs: {abs_gap_seconds/60:.1f})")
            else:
                first_session['is_continuation'] = False
        else:
            if loading_sessions:
                loading_sessions[0]['is_continuation'] = False
    
    current_hour_events = []
    for session in loading_sessions:
        if session['start_time'] >= chunk_start_time:
            if client_id:
                session['Client_ID'] = client_id
            current_hour_events.append(session)
        elif session.get('is_continuation', False):
            if client_id:
                session['Client_ID'] = client_id
            current_hour_events.append(session)
    
    current_session_info = None
    if loading_sessions: 
        last_session = loading_sessions[-1]  
        
        end_time = df_predict['Time'].max()
        last_few_minutes = df_predict[df_predict['Time'] >= (end_time - timedelta(minutes=5))]
        ends_with_loading = (last_few_minutes['pattern'] == 'loading').any() if not last_few_minutes.empty else False
        
        if ends_with_loading:
            current_session_info = {
                'ended_with_loading': True,
                'last_loading_end': last_session['end_time'],
                'session_id': len(loading_sessions)
            }
    
    return pd.DataFrame(current_hour_events), current_session_info

def merge_continuation_events(loading_events_list):
    """
    Merge loading events marked as continuations
    """
    final_events = []
    pending_session = None
    
    for loading_info, _ in loading_events_list:
        if loading_info.empty:
            continue
            
        for _, event in loading_info.iterrows():
            if event.get('is_continuation', False) and pending_session is not None:
                pending_session['end_time'] = event['end_time']
                pending_session['pallet_count'] += event['pallet_count']
                print(f"Merged continuation: Total pallets now {pending_session['pallet_count']}")
            else:
                if pending_session is not None:
                    final_events.append(pending_session)
                
                pending_session = {
                    'Client_ID': event.get('Client_ID'),
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'pattern': event['pattern'],
                    'event': event['event'],
                    'pallet_count': event['pallet_count']
                }
    
    if pending_session is not None:
        final_events.append(pending_session)
    
    return pd.DataFrame(final_events) if final_events else pd.DataFrame(columns=['Client_ID', 'start_time', 'end_time', 'pattern', 'event', 'pallet_count'])
