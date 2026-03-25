import os
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from ShockInference import ShockInference
from utilities import mode_postprocessing, analyze_loading_events
from helper import initialize_influxdb_client, query_hourly_data
from client_config import client_configs
from supabase import create_client
from dotenv import load_dotenv
import logging
from save_data import HourlyDataManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------
# Initialize models ONCE
# -----------------------
np.random.seed(42)

# Stage 1 model (Mode)
stage_1_dir = "saved_models/stage_1_mode"
mode_window_size = 120
mode_step_size = 26
predictor_mode = ShockInference(stage_1_dir, mode_window_size, mode_step_size, 'Mode')
print("Stage_1 model successfully initiated")

# Stage 2 model (Pattern)
stage_2_dir = "saved_models/stage_2_pattern"
pattern_window_size = 500
pattern_step_size = 26
predictor_pattern = ShockInference(stage_2_dir, pattern_window_size, pattern_step_size, 'Pattern')
print("Stage_2 model successfully initiated")


class LoadingEventPipeline:
    """Pipeline for processing loading events and database operations."""
    
    def __init__(self):
        """Initialize the pipeline with database connections."""
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.device_mapping = self.get_device_mapping()
    
    def get_device_mapping(self):
        """Get device mapping from database."""
        try:
            devices_response = self.supabase.table('devices') \
                .select('device_id, device_name, trailer') \
                .execute()
            
            device_mapping = {}
            for device in devices_response.data:
                device_name = device['device_name']
                device_id = device['device_id']
                device_mapping[device_name] = device_id
            
            logger.info(f"Retrieved mapping for {len(device_mapping)} devices")
            return device_mapping
            
        except Exception as e:
            logger.error(f"Error fetching device mapping: {e}")
            return {}
    
    def prepare_loading_event_row(self, row):
        """Prepare a row for loading_events table insertion."""
        client_id = row.get('Client_ID')
        device_id = self.device_mapping.get(client_id)
        
        if not device_id:
            raise ValueError(f"No device_id found for client: {client_id}")
        
        berlin_tz = pytz.timezone('Europe/Berlin')
        utc_tz = pytz.UTC
        
        start_time = pd.to_datetime(row['start_time'])
        end_time = pd.to_datetime(row['end_time'])
        
        if start_time.tzinfo is None:
            start_time = berlin_tz.localize(start_time)
        if end_time.tzinfo is None:
            end_time = berlin_tz.localize(end_time)
        
        start_time_utc = start_time.astimezone(utc_tz).isoformat()
        end_time_utc = end_time.astimezone(utc_tz).isoformat()
        
        return {
            'device_id': int(device_id),
            'start_time': start_time_utc,
            'end_time': end_time_utc,
            'event': int(row['event']),
            'pallet_count': int(row['pallet_count']),
            'created_at': datetime.now(utc_tz).isoformat()
        }
    
    def upsert_loading_events(self, loading_events_df, batch_size=50):
        """Insert loading events into loading_events table."""
        if loading_events_df.empty:
            logger.info("No loading events to insert")
            return True
        
        logger.info(f"Preparing {len(loading_events_df)} loading events for database insertion...")
        
        data_to_insert = []
        
        for _, row in loading_events_df.iterrows():
            try:
                prepared_row = self.prepare_loading_event_row(row)
                data_to_insert.append(prepared_row)
            except ValueError as e:
                logger.warning(f"Error preparing loading event row: {e}")
                continue
        
        if not data_to_insert:
            logger.warning("No valid loading events to insert after preparation")
            return False
        
        table_name = 'loading_events'
        total_rows = len(data_to_insert)
        success_count = 0
        error_count = 0
        
        logger.info(f"Starting bulk insert of {total_rows} loading events in batches of {batch_size}...")
        
        for i in range(0, total_rows, batch_size):
            batch = data_to_insert[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_rows + batch_size - 1) // batch_size
            
            try:
                response = self.supabase.table(table_name).insert(batch).execute()
                success_count += len(batch)
                logger.info(f"Batch {batch_num}/{total_batches} completed: {len(batch)} loading events inserted")
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_num}: {str(e)}")
                error_count += len(batch)
                
                logger.info(f"Attempting individual processing for failed batch {batch_num}...")
                for row in batch:
                    try:
                        response = self.supabase.table(table_name).insert(row).execute()
                        success_count += 1
                        error_count -= 1
                    except Exception as individual_error:
                        logger.warning(f"Individual loading event failed: Device {row['device_id']} - {row['pallet_count']} pallets")
                        logger.warning(f"Error: {str(individual_error)}")
        
        logger.info("="*50)
        logger.info("LOADING EVENTS INSERT SUMMARY")
        logger.info("="*50)
        logger.info(f"Total loading events: {total_rows}")
        logger.info(f"Successfully inserted: {success_count}")
        logger.info(f"Insert errors: {error_count}")
        if total_rows > 0:
            logger.info(f"Success rate: {(success_count/total_rows*100):.1f}%")
        logger.info("="*50)
        
        return error_count == 0


def main(df, predictor_mode, predictor_pattern):
    """Main prediction pipeline function."""
    client_ids = df['Client_ID'].copy() if 'Client_ID' in df.columns else None
    
    # -------- Stage 1 inference --------
    predictions, confidences, avg_probs = predictor_mode.predict_timestamp_level_multi_scale(df)
    print('Stage_1 predictions done.....')
    
    df_pred = mode_postprocessing(df, predictor_mode, predictions, avg_probs)
    print('Stage_1 post processing done.....')
    
    if client_ids is not None:
        df_pred['Client_ID'] = client_ids
    
    # -------- Stage 2 inference (only on long parking) --------
    df_pred['Time'] = pd.to_datetime(df_pred['Time'], errors='coerce')
    df_pred = df_pred.sort_values('Time').reset_index(drop=True)
        
    mask_parking = df_pred['predictor_mode'].eq('parking')
    
    loading_info = pd.DataFrame(columns=['Client_ID', 'start_time', 'end_time', 'pattern', 'event', 'pallet_count'])
    
    if mask_parking.any(): 
        segment_id = (mask_parking != mask_parking.shift(fill_value=False)).cumsum()
        parking_seg_ids = segment_id[mask_parking]
        parking_groups = df_pred.loc[mask_parking].groupby(parking_seg_ids)
        
        segment_durations = parking_groups['Time'].agg(lambda s: (s.iloc[-1] - s.iloc[0]))
        
        segment_durations_seconds = segment_durations.dt.total_seconds()
        min_duration_seconds = 5 * 60  # 5 minutes in seconds
        
        long_parking_ids = segment_durations_seconds[segment_durations_seconds >= min_duration_seconds].index
        long_parking_mask = mask_parking & segment_id.isin(long_parking_ids)
        df_parking = df_pred.loc[long_parking_mask].copy()
        
        if not df_parking.empty:
            predictions1, confidences1, avg_probs1 = predictor_pattern.predict_timestamp_level_multi_scale(df_parking)
            print('Stage_2 predictions done.....')
            predictions1 = np.asarray(predictions1).ravel()
            
            if len(predictions1) != len(df_parking):
                raise ValueError(
                    f"Predictions length ({len(predictions1)}) != number of qualifying parking rows ({len(df_parking)})"
                )
            
            df_pred.loc[long_parking_mask, 'pattern'] = pd.Series(predictions1, index=df_parking.index)
            df_pred.loc[df_pred["pattern"].isna() | (df_pred["pattern"] == "unknown"), "pattern"] = df_pred.get("predictor_mode", "unknown")
            
            print('Loading info .......')
            loading_info = analyze_loading_events(df_pred)
        else:
            print('No long parking segments found, skipping Stage_2...')
            df_pred['pattern'] = df_pred.get('predictor_mode', 'unknown')
    else:
        print('No parking segments found, skipping Stage_2...')
        df_pred['pattern'] = df_pred.get('predictor_mode', 'unknown')
    
    return df_pred, loading_info


def adjust_query_hour(timezone='Europe/Berlin'):
    """Adjust query time for the previous hour with timezone conversion"""
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

def process_previous_hour_data():
    """Process all clients for the previous hour's data - AUTOMATED."""
    start_time_utc, end_time_utc = adjust_query_hour()
    
    tz = pytz.timezone('Europe/Berlin')
    now_berlin = datetime.now(tz)
    current_hour_start = now_berlin.replace(minute=0, second=0, microsecond=0)
    previous_hour_start = current_hour_start - timedelta(hours=1)
    previous_hour_end = current_hour_start
    
    logger.info(f"Processing data from {previous_hour_start.strftime('%Y-%m-%d %H:%M:%S')} to {previous_hour_end.strftime('%Y-%m-%d %H:%M:%S')} Berlin time")
    logger.info(f"UTC query range: {start_time_utc} to {end_time_utc}")
    
    client, query_api, org, bucket = initialize_influxdb_client()
    
    loading_pipeline = LoadingEventPipeline()
    
    all_predictions = []
    all_loading_events = []
    label_map = {"parking": 0, "road": 1}
    
    for client_name, config in client_configs.items():
        logger.info(f"Processing client: {client_name}")
        
        truck_range = config['truck_range']
        client_ids = [f"{client_name}_{i}" for i in truck_range]
        
        for client_id in client_ids:
            logger.info(f"Processing truck: {client_id}")
            
            try:
                df = query_hourly_data(client_id, previous_hour_start, previous_hour_end, query_api, org, bucket)
                
                if df.empty:
                    logger.info(f"No data found for {client_id}")
                    continue
                
                cols = ['Client_ID', 'Time', 'accel_x', 'accel_y', 'accel_z'] + (['label'] if 'label' in df.columns else [])
                df = df[cols]
                df['Client_ID'] = client_id
                
                if 'label' in df.columns:
                    df['mode'] = df['label'].map(label_map)
                
                df_predict, loading_info = main(df, predictor_mode, predictor_pattern)
                
                if not df_predict.empty:
                    df_predict['Client_ID'] = client_id
                    all_predictions.append(df_predict)
                
                if not loading_info.empty:
                    loading_info['Client_ID'] = client_id
                    all_loading_events.append(loading_info)
                    
                logger.info(f"Processed {len(df_predict)} predictions and {len(loading_info)} loading events for {client_id}")
                
            except Exception as e:
                logger.error(f"Error processing {client_id}: {e}")
                continue
    
    client.close()
    
    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()
    
    if not final_loading_events.empty:
        success = loading_pipeline.upsert_loading_events(final_loading_events)
        if success:
            logger.info("Successfully inserted all loading events to database")
        else:
            logger.warning("Some errors occurred inserting loading events")
    
    logger.info("="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total predictions: {len(final_predictions)}")
    logger.info(f"Total loading events: {len(final_loading_events)}")
    
    if not final_loading_events.empty:
        logger.info("Loading events by client:")
        client_breakdown = final_loading_events['Client_ID'].value_counts()
        for client, count in client_breakdown.items():
            logger.info(f"  {client}: {count} events")
        
        total_pallets = final_loading_events['pallet_count'].sum()
        logger.info(f"Total pallets loaded: {total_pallets}")
    
    logger.info("="*50)
    
    return final_predictions, final_loading_events

def process_previous_hour_data():
    """Process all clients for the previous hour's data - AUTOMATED WITH DATA SAVING."""
    start_time_utc, end_time_utc = adjust_query_hour()
    
    tz = pytz.timezone('Europe/Berlin')
    now_berlin = datetime.now(tz)
    current_hour_start = now_berlin.replace(minute=0, second=0, microsecond=0)
    previous_hour_start = current_hour_start - timedelta(hours=1)
    previous_hour_end = current_hour_start
    
    processing_timestamp = previous_hour_start
    
    logger.info(f"Processing data from {previous_hour_start.strftime('%Y-%m-%d %H:%M:%S')} to {previous_hour_end.strftime('%Y-%m-%d %H:%M:%S')} Berlin time")
    logger.info(f"UTC query range: {start_time_utc} to {end_time_utc}")
    
    client, query_api, org, bucket = initialize_influxdb_client()
    loading_pipeline = LoadingEventPipeline()
    data_manager = HourlyDataManager() 
    
    all_predictions = []
    all_loading_events = []
    label_map = {"parking": 0, "road": 1}
    
    for client_name, config in client_configs.items():
        logger.info(f"Processing client: {client_name}")
        
        truck_range = config['truck_range']
        client_ids = [f"{client_name}_{i}" for i in truck_range]
        
        for client_id in client_ids:
            logger.info(f"Processing truck: {client_id}")
            
            try:
                df = query_hourly_data(client_id, previous_hour_start, previous_hour_end, query_api, org, bucket)
                
                if df.empty:
                    logger.info(f"No data found for {client_id}")
                    continue
                
                cols = ['Client_ID', 'Time', 'accel_x', 'accel_y', 'accel_z'] + (['label'] if 'label' in df.columns else [])
                df = df[cols]
                df['Client_ID'] = client_id
                
                if 'label' in df.columns:
                    df['mode'] = df['label'].map(label_map)
                
                df_predict, loading_info = main(df, predictor_mode, predictor_pattern)
                
                if not df_predict.empty:
                    df_predict['Client_ID'] = client_id
                    all_predictions.append(df_predict)
                
                if not loading_info.empty:
                    loading_info['Client_ID'] = client_id
                    all_loading_events.append(loading_info)
                    
                logger.info(f"Processed {len(df_predict)} predictions and {len(loading_info)} loading events for {client_id}")
                
            except Exception as e:
                logger.error(f"Error processing {client_id}: {e}")
                continue
    
    client.close()
    
    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()
    
    logger.info("="*50)
    logger.info("SAVING PREDICTIONS DATA TO FILES")
    logger.info("="*50)
    
    predictions_saved = data_manager.save_predictions(final_predictions, processing_timestamp)
    
    if predictions_saved:
        logger.info("Successfully saved all prediction data to files")
    else:
        logger.warning("Some errors occurred saving prediction data to files")
    
    if not final_loading_events.empty:
        success = loading_pipeline.upsert_loading_events(final_loading_events)
        if success:
            logger.info("Successfully inserted all loading events to database")
        else:
            logger.warning("Some errors occurred inserting loading events")
    
    logger.info("="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total predictions: {len(final_predictions)}")
    logger.info(f"Total loading events: {len(final_loading_events)}")
    logger.info(f"Processing timestamp: {processing_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    if not final_loading_events.empty:
        logger.info("Loading events by client:")
        client_breakdown = final_loading_events['Client_ID'].value_counts()
        for client, count in client_breakdown.items():
            logger.info(f"  {client}: {count} events")
        
        total_pallets = final_loading_events['pallet_count'].sum()
        logger.info(f"Total pallets loaded: {total_pallets}")
    
    logger.info("="*50)
    
    return final_predictions, final_loading_events


if __name__ == '__main__':
    
    logger.info("="*50)
    logger.info("AUTOMATED LOADING EVENT PIPELINE")
    logger.info("="*50)
    logger.info("Processing ALL clients from client_configs")
    logger.info("Querying PREVIOUS HOUR data automatically")
    
    # Process the previous hour's data automatically
    predictions, loading_events = process_previous_hour_data()
    
    logger.info("Pipeline completed successfully!")