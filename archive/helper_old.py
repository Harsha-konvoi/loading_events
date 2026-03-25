import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from main import StatefulProcessor


def initialize_influxdb_client():
    """Initialize InfluxDB client with environment variables"""
    token = os.environ.get("INFLUXDB_TOKEN")
    org = os.environ.get("INFLUXDB_ORG")
    bucket = os.environ.get("INFLUXDB_BUCKET")
    url = os.environ.get("INFLUXDB_URL")
    
    if not all([token, org, bucket, url]):
        raise ValueError("Missing required environment variables")
    
    client = InfluxDBClient(url=url, token=token, org=org, timeout=600_000)
    query_api = client.query_api()
    
    return client, query_api, org, bucket

def query_hourly_data(truck, start_datetime, end_datetime, query_api, org, bucket, table="shock_sensor_0"):
    """Query shock sensor data from InfluxDB for a specific hour range"""
    try:
        # Convert to UTC for InfluxDB query
        tz = pytz.timezone('Europe/Berlin')
        
        if isinstance(start_datetime, str):
            start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
        if isinstance(end_datetime, str):
            end_datetime = datetime.strptime(end_datetime, "%Y-%m-%d %H:%M:%S")
        
        # Localize to Berlin timezone then convert to UTC
        start_utc = tz.localize(start_datetime).astimezone(pytz.utc)
        end_utc = tz.localize(end_datetime).astimezone(pytz.utc)
        
        start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        query = f'''
        from(bucket:"{bucket}")
          |> range(start: {start_str}, stop: {end_str})
          |> filter(fn: (r) => r._measurement == "{table}" and r.client_id == "{truck}")
          |> filter(fn: (r) => r._field == "accel_x" or r._field == "accel_y" or r._field == "accel_z" or r._field == "label")
          |> pivot(rowKey:["_time", "client_id"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement"])
        '''
        
        result = query_api.query_data_frame(org=org, query=query)
        
        if isinstance(result, list):
            if not result:
                return pd.DataFrame(columns=['Time', 'Client_ID', 'accel_x', 'accel_y', 'accel_z', 'label'])
            df = pd.concat(result)
        else:
            df = result
        
        if not df.empty:
            df.rename(columns={"_time": "Time", "client_id": "Client_ID"}, inplace=True)
            
            if not isinstance(df['Time'].iloc[0], pd.Timestamp):
                df['Time'] = pd.to_datetime(df['Time'])
            
            # Convert back to Berlin timezone
            berlin_timezone = pytz.timezone('Europe/Berlin')
            df['Time'] = df['Time'].dt.tz_convert(berlin_timezone)
            
            # Clean up unnecessary columns
            df = df.drop(columns=['result', 'table'], errors='ignore')
            
            # Sort by time
            df = df.sort_values(by='Time').reset_index(drop=True)
            
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
    
    # Get all loading events from current chunk (including overlap)
    loading_sessions = find_loading_sessions(df_predict)
    
    if not loading_sessions:
        return pd.DataFrame(columns=['start_time', 'end_time', 'pattern', 'event', 'pallet_count']), None
    
    # Check overlap region if we have previous session info
    if previous_session_info and previous_session_info['ended_with_loading']:
        overlap_start = chunk_start_time - timedelta(minutes=overlap_minutes)
        overlap_region = df_predict[
            (df_predict['Time'] >= overlap_start) & 
            (df_predict['Time'] < chunk_start_time)
        ]
        
        # Check if overlap region has loading
        has_loading_in_overlap = (overlap_region['pattern'] == 'loading').any() if not overlap_region.empty else False
        
        if has_loading_in_overlap and loading_sessions:
            first_session = loading_sessions[0]
            
            # Check if first session starts close to chunk boundary (within reasonable gap)
            time_gap_seconds = (first_session['start_time'] - previous_session_info['last_loading_end']).total_seconds()
            
            # Handle negative gaps (overlap detection working correctly)
            abs_gap_seconds = abs(time_gap_seconds)
            
            # Apply business rules: gap < 30 minutes (1800 seconds)
            if abs_gap_seconds <= 1800:  # 30 minutes
                # This is a continuation - modify the first session
                first_session['is_continuation'] = True
                first_session['previous_session_end'] = previous_session_info['last_loading_end']
                print(f"Loading session continuation detected. Gap: {time_gap_seconds/60:.1f} minutes (abs: {abs_gap_seconds/60:.1f})")
            else:
                first_session['is_continuation'] = False
        else:
            # No loading in overlap, treat as separate session
            if loading_sessions:
                loading_sessions[0]['is_continuation'] = False
    
    # Filter to only return events that start in the current hour (not overlap)
    current_hour_events = []
    for session in loading_sessions:
        if session['start_time'] >= chunk_start_time:
            current_hour_events.append(session)
        elif session.get('is_continuation', False):
            # Include continuation events even if they technically started in overlap
            current_hour_events.append(session)
    
    # Prepare session info for next chunk
    current_session_info = None
    if loading_sessions:  # Check all sessions in chunk, not just current_hour_events
        last_session = loading_sessions[-1]  # Last session in the full chunk (including overlap)
        
        # Check if chunk ends with loading pattern
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
                # Merge with pending session
                pending_session['end_time'] = event['end_time']
                pending_session['pallet_count'] += event['pallet_count']
                print(f"Merged continuation: Total pallets now {pending_session['pallet_count']}")
            else:
                # Save pending session if exists
                if pending_session is not None:
                    final_events.append(pending_session)
                
                # Start new session
                pending_session = {
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'pattern': event['pattern'],
                    'event': event['event'],
                    'pallet_count': event['pallet_count']
                }
    
    # Don't forget the last session
    if pending_session is not None:
        final_events.append(pending_session)
    
    return pd.DataFrame(final_events) if final_events else pd.DataFrame(columns=['start_time', 'end_time', 'pattern', 'event', 'pallet_count'])

# def process_parquet_with_overlap(df, predictor_mode, predictor_pattern, main_function, chunk_hours=1, overlap_minutes=5):
#     """
#     Process parquet data in chunks with overlapping windows and proper loading session continuity
#     """
#     # Ensure Time column is datetime
#     df['Time'] = pd.to_datetime(df['Time'])
#     df = df.sort_values('Time').reset_index(drop=True)
    
#     # Get time range
#     start_time = df['Time'].min()
#     end_time = df['Time'].max()
    
#     # Round start time to nearest hour
#     current_time = start_time.replace(minute=0, second=0, microsecond=0)
    
#     all_new_results = []  # Only new data for saving
#     all_loading_events_with_info = []  # For continuity processing
#     label_map = {"parking": 0, "road": 1}
#     previous_session_info = None
    
#     chunk_num = 0
#     while current_time < end_time:
#         chunk_num += 1
#         print(f"Processing chunk {chunk_num}: {current_time}")
        
#         # Define chunk window with overlap
#         chunk_start = current_time
#         if chunk_num > 1 and overlap_minutes > 0:
#             chunk_start = current_time - timedelta(minutes=overlap_minutes)
        
#         chunk_end = current_time + timedelta(hours=chunk_hours)
        
#         # Extract chunk data with overlap
#         chunk_mask = (df['Time'] >= chunk_start) & (df['Time'] < chunk_end)
#         chunk_df = df[chunk_mask].copy()
        
#         if chunk_df.empty:
#             print(f"No data in chunk {chunk_start} to {chunk_end}")
#             current_time += timedelta(hours=chunk_hours)
#             continue
        
#         # Prepare data
#         if 'label' in chunk_df.columns:
#             chunk_df['mode'] = chunk_df['label'].map(label_map)
        
#         # Run pipeline on FULL overlapped data for better accuracy
#         df_predict_full, _ = main_function(chunk_df, predictor_mode, predictor_pattern)
        
#         # Analyze loading with continuity awareness
#         loading_info, current_session_info = analyze_loading_with_continuity_check(
#             df_predict_full, current_time, overlap_minutes, previous_session_info
#         )
        
#         # Extract only NEW data for saving (exclude overlap from previous chunk)
#         if chunk_num > 1 and overlap_minutes > 0:
#             df_predict_new = df_predict_full[df_predict_full['Time'] >= current_time].copy()
#         else:
#             df_predict_new = df_predict_full.copy()
        
#         if not df_predict_new.empty:
#             all_new_results.append(df_predict_new)
        
#         # Store loading events with session info for later merging
#         all_loading_events_with_info.append((loading_info, current_session_info))
        
#         previous_session_info = current_session_info
#         current_time += timedelta(hours=chunk_hours)
    
#     if all_new_results:
#         combined_df = pd.concat(all_new_results, ignore_index=True)
#     else:
#         combined_df = pd.DataFrame()
    
#     combined_loading_info = merge_continuation_events(all_loading_events_with_info)
    
#     return combined_df, combined_loading_info

def process_parquet_with_stateful_processor(df, predictor_mode, predictor_pattern, 
                                          chunk_hours=1, overlap_minutes=5):
    """Process parquet with stateful processor maintaining continuity across chunks"""
    
    # Initialize the stateful processor
    processor = StatefulProcessor(predictor_mode, predictor_pattern)
    
    chunk_duration = pd.Timedelta(hours=chunk_hours)
    overlap_duration = pd.Timedelta(minutes=overlap_minutes)
    
    all_predictions = []
    all_loading_info = []
    
    df = df.sort_values('Time').reset_index(drop=True)
    start_time = df['Time'].min()
    end_time = df['Time'].max()
    
    current_time = start_time
    chunk_number = 0
    
    while current_time < end_time:
        chunk_start = current_time
        chunk_end = min(current_time + chunk_duration, end_time)
        
        # Get chunk data
        chunk_mask = (df['Time'] >= chunk_start) & (df['Time'] < chunk_end)
        df_chunk = df[chunk_mask].copy()
        
        if df_chunk.empty:
            current_time = chunk_end
            continue
        
        chunk_number += 1
        print(f"Processing chunk {chunk_number}: {chunk_start} to {chunk_end} ({len(df_chunk)} samples)")
        
        # Process with maintained state - THIS IS THE KEY DIFFERENCE
        df_pred_new, loading_info = processor.process_chunk_stateful(df_chunk)
        
        all_predictions.append(df_pred_new)
        if not loading_info.empty:
            all_loading_info.append(loading_info)
        
        # Move to next chunk
        current_time = chunk_end - overlap_duration
    
    # Combine results
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        # Remove any potential duplicates from overlapping
        final_predictions = final_predictions.drop_duplicates(subset=['Time'], keep='last').sort_values('Time')
    else:
        final_predictions = pd.DataFrame()
    
    if all_loading_info:
        final_loading_info = pd.concat(all_loading_info, ignore_index=True)
        final_loading_info = final_loading_info.drop_duplicates()
    else:
        final_loading_info = pd.DataFrame()
    
    return final_predictions, final_loading_info

def continuous_hourly_processing(truck, start_date, predictor_mode, predictor_pattern, main_function, hours_to_process=24, overlap_minutes=5):
    """
    Production-ready hourly processing with proper session continuity
    """
    client, query_api, org, bucket = initialize_influxdb_client()
    
    if isinstance(start_date, str):
        current_hour = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    else:
        current_hour = start_date
    
    current_hour = current_hour.replace(minute=0, second=0, microsecond=0)
    
    all_results = []
    all_loading_events_with_info = []
    previous_session_info = None
    label_map = {"parking": 0, "road": 1}
    
    for i in range(hours_to_process):
        print(f"Processing hour {i+1}/{hours_to_process}: {current_hour}")
        
        query_start = current_hour
        if i > 0 and overlap_minutes > 0:
            query_start = current_hour - timedelta(minutes=overlap_minutes)
        
        query_end = current_hour + timedelta(hours=1)
        
        df = query_hourly_data(truck, query_start, query_end, query_api, org, bucket)
        
        if df.empty:
            print(f"No data found for {truck} between {query_start} and {query_end}")
            all_results.append((current_hour, pd.DataFrame(), pd.DataFrame()))
            all_loading_events_with_info.append((pd.DataFrame(), None))
        else:
            cols = ['Client_ID', 'Time', 'accel_x', 'accel_y', 'accel_z'] + (['label'] if 'label' in df.columns else [])
            df = df[cols]
            
            if 'label' in df.columns:
                df['mode'] = df['label'].map(label_map)
            
            df_predict_full, _ = main_function(df, predictor_mode, predictor_pattern)
            
            loading_info, current_session_info = analyze_loading_with_continuity_check(
                df_predict_full, current_hour, overlap_minutes, previous_session_info
            )
            
            df_predict_new = df_predict_full[df_predict_full['Time'] >= current_hour].copy()
            
            all_results.append((current_hour, df_predict_full, df_predict_new))
            all_loading_events_with_info.append((loading_info, current_session_info))
            
            previous_session_info = current_session_info
        
        current_hour += timedelta(hours=1)
    
    client.close()
    
    final_loading_events = merge_continuation_events(all_loading_events_with_info)
    
    return all_results, final_loading_events