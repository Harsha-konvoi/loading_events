import os
import numpy as np
import pandas as pd
from ShockInference import ShockInference
from utilities import mode_postprocessing, analyze_loading_events
from helper_old import continuous_hourly_processing, process_parquet_with_overlap

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


def main(df, predictor_mode, predictor_pattern):
    client_ids = df['Client_ID'].copy() if 'Client_ID' in df.columns else None
    
    # -------- Stage 1 inference --------
    predictions, confidences, avg_probs = predictor_mode.predict_timestamp_level_multi_scale(df)
    print('Stage_1 predictions done.....')
    
    df_pred = mode_postprocessing(df, predictor_mode, predictions, avg_probs)
    print('Stage_1 post processing done.....')
    
    # -------- Stage 2 inference (only on long parking) --------
    df_pred['Time'] = pd.to_datetime(df_pred['Time'], errors='coerce')
    df_pred = df_pred.sort_values('Time').reset_index(drop=True)
        
    mask_parking = df_pred['predictor_mode'].eq('parking')
    
    loading_info = pd.DataFrame(columns=['Client_ID' ,'start_time', 'end_time', 'pattern', 'event', 'pallet_count'])
    
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
            df_pred['pattern'] = df_pred.get('label', 'unknown')
    else:
        print('No parking segments found, skipping Stage_2...')
        df_pred['pattern'] = df_pred.get('label', 'unknown')
    
    return df_pred, loading_info


if __name__ == '__main__':
    # Configuration
    use_parquet = True  # Set to False to use hourly processing from InfluxDB
    overlap_minutes = 5  # Minutes of overlap between processing windows
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    if use_parquet:
        data_path = "validation/Ups_2_2025-07-31.parquet"
        
        df = pd.read_parquet(data_path)
        cols = ['Client_ID', 'Time', 'accel_x', 'accel_y', 'accel_z'] + (['label'] if 'label' in df.columns else [])
        df = df[cols]
        
        print(f"Processing {len(df)} records with {overlap_minutes}-minute overlap...")
        
        df_predict, loading_info = process_parquet_with_overlap(
            df, predictor_mode, predictor_pattern, main, 
            chunk_hours=1, overlap_minutes=overlap_minutes
        )
        
        print(f"Processed {len(df_predict)} records")
        if not loading_info.empty:
            print(f"Found {len(loading_info)} loading events")
        

        if not df_predict.empty:
            predictions_file = os.path.join(results_dir, "predictions_output.csv")
            df_predict.to_csv(predictions_file, index=False)
            print(f"Predictions saved to: {predictions_file}")
        
        if not loading_info.empty:
            loading_file = os.path.join(results_dir, "loading_events_output.csv")
            loading_info.to_csv(loading_file, index=False)
            print(f"Loading events saved to: {loading_file}")
        else:
            print("No loading events to save")
    
    else:
        # Option 2: Hourly processing from InfluxDB with overlap
        truck_id = "your_truck_id"
        start_time = "2025-01-15 08:00:00"
        hours_to_process = 12
        
        print(f"Starting hourly processing with {overlap_minutes}-minute overlap...")
        
        results = continuous_hourly_processing(
            truck_id, start_time, predictor_mode, predictor_pattern, main, 
            hours_to_process, overlap_minutes=overlap_minutes
        )
        
        # Process results - you can choose what to save
        all_predictions_new = []  # For saving/appending (no duplicates)
        all_predictions_full = [] # For analysis (includes overlaps)
        all_loading_info = []
        
        for hour, df_predict_full, df_predict_new, loading_info in results:
            print(f"Hour: {hour}")
            print(f"Full predictions (with overlap): {df_predict_full.shape}")
            print(f"New predictions (for saving): {df_predict_new.shape}")
            if not loading_info.empty:
                print(f"Loading events: {len(loading_info)}")
            print("-" * 50)
            
            # Collect new data for saving/appending
            if not df_predict_new.empty:
                all_predictions_new.append(df_predict_new)
                
            # Optionally collect full data for analysis
            if not df_predict_full.empty:
                all_predictions_full.append(df_predict_full)
                
            if not loading_info.empty:
                all_loading_info.append(loading_info)
        
        # Combine results for saving (no duplicates)
        if all_predictions_new:
            combined_predictions_new = pd.concat(all_predictions_new, ignore_index=True)
            print(f"Total NEW predictions for saving: {len(combined_predictions_new)} records")
        
        # Combine loading events
        if all_loading_info:
            combined_loading_info = pd.concat(all_loading_info, ignore_index=True)
            print(f"Total loading events: {len(combined_loading_info)} events")
            
        combined_predictions_new.to_parquet("output_predictions.parquet")
        combined_loading_info.to_parquet("output_loading_events.parquet")