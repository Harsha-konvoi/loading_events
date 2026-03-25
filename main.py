import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

from ShockInference import ShockInference
from utilities import mode_postprocessing, analyze_loading_events, correct_short_road_segments, \
    remove_false_positive_loading
from helper import initialize_clickhouse_client, query_hourly_data, adjust_query_hour
from client_config import client_configs
from supabase import create_client
from dotenv import load_dotenv
from save_data import HourlyDataManager
from pathlib import Path

load_dotenv()

# ============================================================
# LOGGING SETUP
# ============================================================
LOG_BASE_DIR = Path(r"D:\Logs\Shock_model_log\logs")


def setup_logger(name="shock_pipeline", log_base_dir=LOG_BASE_DIR):
    """
    Configure logging to write to both console and an hourly .log file
    inside a date-based folder.

    Folder structure:
      D:\Logs\Shock_model_log\
        2026-03-24\
          shock_pipeline_2026-03-24_13-00.log   ← one file per hourly run
          shock_pipeline_2026-03-24_14-00.log
        2026-03-25\
          shock_pipeline_2026-03-25_08-00.log
          ...

    Each run creates a fresh log file named by the hour it was triggered.
    Old date folders are kept — clean up externally if needed (e.g. 30-day cron).
    """
    logger = logging.getLogger(name)

    # Remove any existing handlers (allows re-calling per hourly run)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    # --- Create date folder: D:\Logs\Shock_model_log\YYYY-MM-DD\ ---
    now = datetime.now(pytz.timezone('Europe/Berlin'))
    date_str = now.strftime('%Y-%m-%d')
    hour_str = now.strftime('%H-%M')

    log_dir = log_base_dir / date_str
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"shock_pipeline_{date_str}_{hour_str}.log"

    # --- Formatter ---
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- File handler: one file per hourly run ---
    file_handler = logging.FileHandler(
        filename=str(log_file),
        mode='a',
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.__stdout__)  # use original stdout
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized — log file: {log_file}")
    return logger


# Initialize logger
logger = setup_logger()


# ============================================================
# Redirect stdout so that third-party print() calls also get logged
# ============================================================
class LoggerWriter:
    """Redirect print() from third-party libs to the logger."""

    def __init__(self, logger, level=logging.INFO):
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, msg):
        if msg and msg.strip():
            self._logger.log(self._level, msg.strip())

    def flush(self):
        pass


sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.WARNING)

# -----------------------
# Initialize models ONCE
# -----------------------
np.random.seed(42)

# Stage 1 model (Mode)
stage_1_dir = "saved_models/stage_1_mode/v2"
mode_window_size = 120
mode_step_size = 26
predictor_mode = ShockInference(stage_1_dir, mode_window_size, mode_step_size, 'Mode')
logger.info("Stage_1 model successfully initiated")

# Stage 2 model (Pattern)
stage_2_dir = "saved_models/stage_2_pattern/v2"
pattern_window_size = 500
pattern_step_size = 26
predictor_pattern = ShockInference(stage_2_dir, pattern_window_size, pattern_step_size, 'Pattern')
logger.info("Stage_2 model successfully initiated")


class LoadingEventPipeline:
    """Pipeline for processing loading events and database operations with daily CSV export."""

    def __init__(self, csv_output_dir=None):
        """Initialize the pipeline with database connections and CSV output directory."""
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.device_mapping = self.get_device_mapping()

        # Set up CSV output directory
        if csv_output_dir is None:
            self.csv_output_dir = Path(r"D:\Logs\Shock_model_log\Results")
        else:
            self.csv_output_dir = Path(csv_output_dir)

        self.csv_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CSV output directory set to: {self.csv_output_dir}")

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

    def get_recent_loading_events_for_consolidation(self, device_id, hours_back=2):
        try:
            tz = pytz.timezone('Europe/Berlin')
            utc_tz = pytz.UTC
            now = datetime.now(tz)
            lookback_time = now - timedelta(hours=hours_back)
            lookback_utc = lookback_time.astimezone(utc_tz).isoformat()

            response = self.supabase.table('loading_events').select('*').filter(
                'device_id', 'eq', device_id
            ).filter(
                'end_time', 'gte', lookback_utc
            ).order('end_time', desc=True).limit(5).execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Error fetching recent loading events for device {device_id}: {e}")
            return []

    def should_consolidate_loading_events(self, new_event, existing_event, df_predict):
        try:
            if new_event.get('stop_id') != existing_event.get('stop_id'):
                logger.debug(
                    f"Different stop_ids: new={new_event.get('stop_id')}, existing={existing_event.get('stop_id')}")
                return False

            if new_event.get('stop_id') is None:
                logger.debug("Both events have no stop_id - not consolidating for safety")
                return False

            existing_end = pd.to_datetime(existing_event['end_time'])
            new_start = pd.to_datetime(new_event['start_time'])

            gap_hours = (new_start - existing_end).total_seconds() / 3600

            if gap_hours > 1.0:
                logger.debug(f"Time gap too large: {gap_hours:.2f} hours")
                return False

            if gap_hours < 0:
                logger.info(f"Events overlap by {abs(gap_hours):.2f} hours - treating as continuation")
                return True

            if not df_predict.empty and 'pattern' in df_predict.columns:
                loading_mask = df_predict['pattern'] == 'loading'
                if loading_mask.any():
                    first_loading_idx = df_predict[loading_mask].index[0]

                    before_loading = df_predict.iloc[:first_loading_idx]
                    if not before_loading.empty and 'road' in before_loading['pattern'].values:
                        logger.info("Found 'road' pattern before first loading - not consolidating")
                        return False

            logger.info(
                f"Events should be consolidated - gap: {gap_hours:.2f} hours, same stop_id: {new_event.get('stop_id')}")
            return True

        except Exception as e:
            logger.error(f"Error checking consolidation: {e}")
            return False

    def consolidate_loading_events(self, new_event, existing_event):
        try:
            existing_end = pd.to_datetime(existing_event['end_time'])
            new_end = pd.to_datetime(new_event['end_time'])
            consolidated_end = max(existing_end, new_end)

            existing_pallets = existing_event.get('pallet_count', 0)
            new_pallets = new_event.get('pallet_count', 0)
            total_pallets = existing_pallets + new_pallets

            update_data = {
                'end_time': consolidated_end.isoformat(),
                'pallet_count': total_pallets,
                'created_at': datetime.now(pytz.utc).isoformat()
            }

            response = self.supabase.table('loading_events').update(update_data).eq(
                'id', existing_event['id']
            ).execute()

            logger.info(f"Successfully consolidated loading events:")
            logger.info(f"  Event ID: {existing_event['id']}")
            logger.info(f"  Original: {existing_pallets} pallets, ended at {existing_end.strftime('%H:%M:%S')}")
            logger.info(f"  Updated: {total_pallets} pallets, ended at {consolidated_end.strftime('%H:%M:%S')}")
            logger.info(f"  Added: {new_pallets} pallets from new event")

            return True

        except Exception as e:
            logger.error(f"Error consolidating loading events: {e}")
            return False

    def get_recent_parking_event_for_loading(self, device_id, loading_start_time, loading_end_time, hours_back=48):
        try:
            berlin_tz = pytz.timezone('Europe/Berlin')
            utc_tz = pytz.UTC

            if loading_start_time.tzinfo is None:
                loading_start_time = berlin_tz.localize(loading_start_time).astimezone(utc_tz)
            else:
                loading_start_time = loading_start_time.astimezone(utc_tz)

            if loading_end_time.tzinfo is None:
                loading_end_time = berlin_tz.localize(loading_end_time).astimezone(utc_tz)
            else:
                loading_end_time = loading_end_time.astimezone(utc_tz)

            search_bound = loading_start_time - timedelta(hours=hours_back)

            logger.debug(f"Searching for parking events for device {device_id}")
            logger.debug(
                f"Loading event: {loading_start_time.strftime('%H:%M:%S')} to {loading_end_time.strftime('%H:%M:%S')}")

            response = self.supabase.table('trailer_event_log').select('*').filter(
                'device_id', 'eq', device_id
            ).filter(
                'event', 'eq', 0
            ).filter(
                'start_time', 'lte', loading_start_time.isoformat()
            ).filter(
                'end_time', 'gte', loading_end_time.isoformat()
            ).filter(
                'end_time', 'gte', search_bound.isoformat()
            ).order('start_time', desc=True).limit(5).execute()

            if response.data:
                parking_event = response.data[0]
                parking_start = pd.to_datetime(parking_event['start_time'])
                parking_end = pd.to_datetime(parking_event['end_time'])

                logger.info(f"Found containing parking event: ID {parking_event['id']}")
                logger.debug(f"  Parking: {parking_start.strftime('%H:%M:%S')} to {parking_end.strftime('%H:%M:%S')}")
                logger.debug(
                    f"  Loading: {loading_start_time.strftime('%H:%M:%S')} to {loading_end_time.strftime('%H:%M:%S')}")
                return parking_event
            else:
                logger.warning(f"No parking events found that contain the loading event for device {device_id}")

            return None

        except Exception as e:
            logger.error(f"Error fetching parking event for loading: {e}")
            return None

    def prepare_loading_event_row(self, row, df_predict=None):
        """Prepare a row for loading_events table insertion with stop_id linking and consolidation check."""
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

        start_time_utc = start_time.astimezone(utc_tz)
        end_time_utc = end_time.astimezone(utc_tz)

        parking_event = self.get_recent_parking_event_for_loading(
            device_id, start_time_utc, end_time_utc
        )

        stop_id = None
        if parking_event:
            stop_id = parking_event['id']
            logger.info(f"Linking loading event to parking stop ID: {stop_id}")
        else:
            logger.warning("No matching parking event found - loading event will have null stop_id")

        new_event_data = {
            'device_id': int(device_id),
            'start_time': start_time_utc.isoformat(),
            'end_time': end_time_utc.isoformat(),
            'event': int(row['event']),
            'pallet_count': int(row['pallet_count']),
            'stop_id': stop_id,
            'created_at': datetime.now(utc_tz).isoformat()
        }

        recent_events = self.get_recent_loading_events_for_consolidation(device_id)

        for existing_event in recent_events:
            if self.should_consolidate_loading_events(new_event_data, existing_event, df_predict):
                success = self.consolidate_loading_events(new_event_data, existing_event)
                if success:
                    return None
                else:
                    logger.warning(f"Failed to consolidate loading events - will insert as new event")
                    break

        return new_event_data

    def get_daily_csv_file_path(self, timestamp):
        """Get the daily CSV file path based on timestamp."""
        date_str = timestamp.strftime('%Y-%m-%d')
        filename = f"loading_events_{date_str}.csv"
        return self.csv_output_dir / filename

    def save_loading_events_to_daily_csv(self, loading_events_df, processing_timestamp=None):
        """Save loading events to daily CSV file with appending and sorting by Client_ID."""
        try:
            if loading_events_df.empty:
                logger.info("No loading events to save to CSV")
                return True

            if processing_timestamp is None:
                processing_timestamp = datetime.now(pytz.timezone('Europe/Berlin'))

            csv_file_path = self.get_daily_csv_file_path(processing_timestamp)

            csv_data = loading_events_df.copy()
            csv_data['processed_at'] = processing_timestamp.isoformat()

            if csv_file_path.exists():
                existing_df = pd.read_csv(csv_file_path)
                logger.info(f"Found existing daily CSV with {len(existing_df)} rows: {csv_file_path.name}")

                combined_df = pd.concat([existing_df, csv_data], ignore_index=True)

                combined_df = combined_df.drop_duplicates(
                    subset=['Client_ID', 'start_time', 'end_time'],
                    keep='last'
                )

                logger.info(f"After deduplication: {len(combined_df)} rows")
            else:
                combined_df = csv_data
                logger.info(f"Creating new daily CSV file: {csv_file_path.name}")

            combined_df = combined_df.sort_values(['Client_ID', 'start_time']).reset_index(drop=True)
            combined_df.to_csv(csv_file_path, index=False)

            logger.info(f"Successfully saved {len(csv_data)} new loading events to daily CSV")
            logger.info(f"Total events in daily CSV: {len(combined_df)}")
            logger.info(f"Daily CSV file: {csv_file_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving loading events to daily CSV: {e}")
            return False

    def upsert_loading_events(self, loading_events_df, df_predict=None, batch_size=50, save_csv=True,
                              processing_timestamp=None):
        """Insert loading events into database AND save to daily CSV with stop_id linking and consolidation."""
        if loading_events_df.empty:
            logger.info("No loading events to process")
            return True

        logger.info(f"Preparing {len(loading_events_df)} loading events for database insertion and daily CSV export...")

        csv_success = True
        if save_csv:
            csv_success = self.save_loading_events_to_daily_csv(loading_events_df, processing_timestamp)
            if not csv_success:
                logger.warning("Failed to save to daily CSV, but continuing with database operations")

        data_to_insert = []
        linked_count = 0
        unlinked_count = 0
        consolidated_count = 0

        for _, row in loading_events_df.iterrows():
            try:
                client_predictions = None
                if df_predict is not None and not df_predict.empty:
                    client_id = row.get('Client_ID')
                    client_predictions = df_predict[
                        df_predict['Client_ID'] == client_id] if 'Client_ID' in df_predict.columns else df_predict

                prepared_row = self.prepare_loading_event_row(row, client_predictions)

                if prepared_row is None:
                    consolidated_count += 1
                    continue

                data_to_insert.append(prepared_row)

                if prepared_row['stop_id'] is not None:
                    linked_count += 1
                else:
                    unlinked_count += 1

            except ValueError as e:
                logger.error(f"Error preparing loading event row: {e}")
                continue

        if not data_to_insert:
            if consolidated_count > 0:
                logger.info(f"All {consolidated_count} loading events were consolidated with existing events")
                return True
            else:
                logger.warning("No valid loading events to insert after preparation")
                return False

        table_name = 'loading_events'
        total_rows = len(data_to_insert)
        success_count = 0
        error_count = 0

        logger.info(f"Starting bulk insert of {total_rows} new loading events in batches of {batch_size}...")

        for i in range(0, total_rows, batch_size):
            batch = data_to_insert[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_rows + batch_size - 1) // batch_size

            try:
                response = self.supabase.table(table_name).insert(batch).execute()
                success_count += len(batch)
                logger.info(f"Batch {batch_num}/{total_batches} completed: {len(batch)} loading events inserted")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                error_count += len(batch)

                logger.info(f"Attempting individual processing for failed batch {batch_num}...")
                for row in batch:
                    try:
                        response = self.supabase.table(table_name).insert(row).execute()
                        success_count += 1
                        error_count -= 1
                    except Exception as individual_error:
                        logger.error(
                            f"Individual loading event failed: Device {row['device_id']} - {row['pallet_count']} pallets")
                        logger.error(f"Error: {str(individual_error)}")

        logger.info("=" * 50)
        logger.info("LOADING EVENTS WITH CONSOLIDATION AND DAILY CSV EXPORT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total loading events processed: {len(loading_events_df)}")
        logger.info(f"Consolidated with existing events: {consolidated_count}")
        logger.info(f"New events inserted to database: {success_count}")
        logger.info(f"Linked to parking events: {linked_count}")
        logger.info(f"Not linked (no matching parking): {unlinked_count}")
        logger.info(f"Database insert errors: {error_count}")
        if save_csv:
            logger.info(f"Daily CSV export: {'Success' if csv_success else 'Failed'}")
            if processing_timestamp:
                daily_file = self.get_daily_csv_file_path(processing_timestamp)
                logger.info(f"Daily CSV file: {daily_file}")

        total_processed = len(loading_events_df)
        if total_processed > 0:
            logger.info(f"Success rate: {((success_count + consolidated_count) / total_processed * 100):.1f}%")
            logger.info(f"Consolidation rate: {(consolidated_count / total_processed * 100):.1f}%")
            if linked_count + unlinked_count > 0:
                logger.info(f"Linking rate: {(linked_count / (linked_count + unlinked_count) * 100):.1f}%")
        logger.info("=" * 50)

        return error_count == 0


def main(df, predictor_mode, predictor_pattern):
    """Main prediction pipeline function."""
    client_ids = df['Client_ID'].copy() if 'Client_ID' in df.columns else None

    # -------- Stage 1 inference --------
    predictions, confidences, avg_probs = predictor_mode.predict_timestamp_level_multi_scale(df)
    logger.info('Stage_1 predictions done')

    df_pred = mode_postprocessing(df, predictor_mode, predictions, avg_probs)
    logger.info('Stage_1 post processing done')

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
        min_duration_seconds = 5 * 60

        long_parking_ids = segment_durations_seconds[segment_durations_seconds >= min_duration_seconds].index
        long_parking_mask = mask_parking & segment_id.isin(long_parking_ids)
        df_parking = df_pred.loc[long_parking_mask].copy()

        if not df_parking.empty:
            predictions1, confidences1, avg_probs1 = predictor_pattern.predict_timestamp_level_multi_scale(df_parking)
            logger.info('Stage_2 predictions done')
            predictions1 = np.asarray(predictions1).ravel()

            if len(predictions1) != len(df_parking):
                raise ValueError(
                    f"Predictions length ({len(predictions1)}) != number of qualifying parking rows ({len(df_parking)})"
                )

            df_pred.loc[long_parking_mask, 'pattern'] = pd.Series(predictions1, index=df_parking.index)
            df_pred.loc[df_pred["pattern"].isna() | (df_pred["pattern"] == "unknown"), "pattern"] = df_pred.get(
                "predictor_mode", "unknown")

            # CORRECTION STEP 1: Fix short road blips between loading
            logger.info('Correcting short road segments...')
            df_pred = correct_short_road_segments(
                df_pred,
                pattern_col='pattern',
                time_col='Time',
                max_duration_sec=60,
            )

            # CORRECTION STEP 2: Remove false-positive loading labels
            logger.info('Removing false-positive loading segments...')
            df_pred = remove_false_positive_loading(
                df_pred,
                pattern_col='pattern',
                time_col='Time',
                accel_cols=('accel_x', 'accel_y', 'accel_z'),
                energy_method='std',
                threshold=None,
                auto_threshold_factor=2.0,
                relabel_to='parking',
            )

            logger.info('Analyzing loading events...')
            loading_info = analyze_loading_events(df_pred)
        else:
            logger.info('No long parking segments found, skipping Stage_2')
            df_pred['pattern'] = df_pred.get('predictor_mode', 'unknown')
    else:
        logger.info('No parking segments found, skipping Stage_2')
        df_pred['pattern'] = df_pred.get('predictor_mode', 'unknown')

    return df_pred, loading_info


def process_previous_hour_data():
    """Process all clients for the previous hour's data."""
    start_time_utc, end_time_utc = adjust_query_hour()

    tz = pytz.timezone('Europe/Berlin')
    now_berlin = datetime.now(tz)
    current_hour_start = now_berlin.replace(minute=0, second=0, microsecond=0)
    previous_hour_start = current_hour_start - timedelta(hours=1)
    previous_hour_end = current_hour_start

    processing_timestamp = previous_hour_start

    logger.info(
        f"Processing data from {previous_hour_start.strftime('%Y-%m-%d %H:%M:%S')} to {previous_hour_end.strftime('%Y-%m-%d %H:%M:%S')} Berlin time")
    logger.info(f"UTC query range: {start_time_utc} to {end_time_utc}")

    ch_client = initialize_clickhouse_client()

    csv_output_dir = r"D:\Logs\Shock_model_log\Results"
    loading_pipeline = LoadingEventPipeline(csv_output_dir=csv_output_dir)

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
                df = query_hourly_data(client_id, previous_hour_start, previous_hour_end, ch_client)

                if df.empty:
                    logger.warning(f"No data found for {client_id}")
                    continue

                cols = ['Client_ID', 'Time', 'accel_x', 'accel_y', 'accel_z'] + (
                    ['label'] if 'label' in df.columns else [])
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

                logger.info(
                    f"Processed {len(df_predict)} predictions and {len(loading_info)} loading events for {client_id}")

            except Exception as e:
                logger.error(f"Error processing {client_id}: {e}")
                logger.debug(traceback.format_exc())
                continue

    ch_client.close()

    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()

    logger.info("=" * 50)
    logger.info("SAVING PREDICTIONS DATA TO FILES")
    logger.info("=" * 50)

    predictions_saved = data_manager.save_predictions(final_predictions, processing_timestamp)

    if predictions_saved:
        logger.info("Successfully saved all prediction data to files")
    else:
        logger.error("Some errors occurred saving prediction data to files")

    if not final_loading_events.empty:
        success = loading_pipeline.upsert_loading_events(
            final_loading_events,
            final_predictions,
            save_csv=True,
            processing_timestamp=processing_timestamp
        )
        if success:
            logger.info("Successfully inserted all loading events to database and exported to CSV")
        else:
            logger.error("Some errors occurred inserting loading events or exporting CSV")

    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total predictions: {len(final_predictions)}")
    logger.info(f"Total loading events: {len(final_loading_events)}")
    logger.info(f"Processing timestamp: {processing_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"CSV output path: {csv_output_dir}")

    if not final_loading_events.empty:
        logger.info("Loading events by client:")
        client_breakdown = final_loading_events['Client_ID'].value_counts()
        for client, count in client_breakdown.items():
            logger.info(f"  {client}: {count} events")

        total_pallets = final_loading_events['pallet_count'].sum()
        logger.info(f"Total pallets loaded: {total_pallets}")

    logger.info("=" * 50)

    return final_predictions, final_loading_events


if __name__ == '__main__':

    BACKTEST_MODE = False

    if BACKTEST_MODE:
        # ==========================================
        # BACKTEST CONFIGURATION
        # ==========================================

        SINGLE_CLIENT = True
        SINGLE_DATE = True

        CLIENT_ID = "Lindt_7"

        DATE = "2026-03-18"
        START_DATE = "2026-03-17"
        END_DATE = "2026-03-23"

        SAVE_CSV = True
        CSV_OUTPUT_DIR = r"D:\Logs\Shock_model_log"

        from backtest import (
            run_backtest_single,
            run_backtest_multi_date,
            run_backtest_all_clients,
            run_backtest_all_clients_multi_date
        )

        if SINGLE_CLIENT and SINGLE_DATE:
            predictions, loading_events = run_backtest_single(
                client_id=CLIENT_ID,
                date_str=DATE,
                predictor_mode=predictor_mode,
                predictor_pattern=predictor_pattern,
                main_func=main,
                save_csv=SAVE_CSV,
                csv_output_dir=CSV_OUTPUT_DIR
            )

        elif SINGLE_CLIENT and not SINGLE_DATE:
            predictions, loading_events = run_backtest_multi_date(
                client_id=CLIENT_ID,
                start_date=START_DATE,
                end_date=END_DATE,
                predictor_mode=predictor_mode,
                predictor_pattern=predictor_pattern,
                main_func=main,
                save_csv=SAVE_CSV,
                csv_output_dir=CSV_OUTPUT_DIR
            )

        elif not SINGLE_CLIENT and SINGLE_DATE:
            predictions, loading_events = run_backtest_all_clients(
                date_str=DATE,
                predictor_mode=predictor_mode,
                predictor_pattern=predictor_pattern,
                main_func=main,
                save_csv=SAVE_CSV,
                csv_output_dir=CSV_OUTPUT_DIR
            )

        else:
            predictions, loading_events = run_backtest_all_clients_multi_date(
                start_date=START_DATE,
                end_date=END_DATE,
                predictor_mode=predictor_mode,
                predictor_pattern=predictor_pattern,
                main_func=main,
                save_csv=SAVE_CSV,
                csv_output_dir=CSV_OUTPUT_DIR
            )

    else:
        # ==========================================
        # PRODUCTION MODE - AUTOMATED PIPELINE
        # ==========================================
        logger.info("=" * 50)
        logger.info("AUTOMATED LOADING EVENT PIPELINE WITH CONSOLIDATION")
        logger.info("=" * 50)
        logger.info("Processing ALL clients from client_configs")
        logger.info("Querying PREVIOUS HOUR data automatically")

        try:
            predictions, loading_events = process_previous_hour_data()
            logger.info("Pipeline completed successfully!")
        except Exception as e:
            logger.critical(f"Pipeline failed with unhandled exception: {e}")
            logger.critical(traceback.format_exc())
            sys.exit(1)