import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from pathlib import Path
import calendar
from utilities import analyze_loading_events


BACKTEST_CLIENT_CONFIGS = {
    'Nagel': {
        'truck_range': range(21, 46),
    },
    'Bode': {
        'truck_range': range(21, 40),
    },
    'Greilmeier': {
        'truck_range': range(3, 5),
    },
    'Kipfer': {
        'truck_range': range(3, 5),
    },
    'Login': {
        'truck_range': range(1, 4),
    },
    'Aircargo': {
        'truck_range': range(1, 6),
    },
    'Ups': {
        'truck_range': range(1, 3),
    },
    'Sti': {
        'truck_range': range(1, 3),
    },
    'Ktn': {
        'truck_range': range(1, 5),
    },
    
}


class BacktestDataLoader:
    """Load historical data for backtesting from parquet files."""
    
    def __init__(self, main_path=None):
        if main_path is None:
            self.main_path = Path(r"C:\Users\HarshavardhanMoravap\KONVOI GmbH\Communication site - Product Development\Data")
        else:
            self.main_path = Path(main_path)
        
        print(f"Initialized BacktestDataLoader with path: {self.main_path}")
    
    def extract_client_name(self, client_id):
        """Extract client name from client_id (e.g., 'ClientA_1' -> 'ClientA')."""
        if '_' in client_id:
            return client_id.split('_')[0]
        else:
            return client_id
    
    def read_shock_data(self, client_id, date_str):
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.strftime('%Y')
            month_name = f"{date_obj.strftime('%m')}_{calendar.month_name[date_obj.month]}"
            
            client_name = self.extract_client_name(client_id)
            
            shock_folder = os.path.join(
                self.main_path, "Local_database", year, month_name,
                client_name, "Shock data", client_id
            )
            
            file_path = os.path.join(shock_folder, f"{client_id}_{date_str}.parquet")
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
            if not df.empty:
                if 'Time' in df.columns:
                    df['Time'] = pd.to_datetime(df['Time'])
                
                df['Client_ID'] = client_id
                
                print(f"Loaded {len(df)} rows for {client_id} on {date_str}")
                print(f"Time range: {df['Time'].min()} to {df['Time'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading shock data for {client_id} on {date_str}: {e}")
            return pd.DataFrame()


def generate_date_range(start_date, end_date):
    """Generate list of dates between start_date and end_date (inclusive)."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def run_backtest_single(client_id, date_str, predictor_mode, predictor_pattern, main_func, 
                       save_csv=False, csv_output_dir=None):
    print("="*50)
    print("BACKTEST MODE - SINGLE CLIENT, SINGLE DATE")
    print("="*50)
    print(f"Client: {client_id}")
    print(f"Date: {date_str}")
    print("="*50)
    
    loader = BacktestDataLoader()
    
    df = loader.read_shock_data(client_id, date_str)
    
    if df.empty:
        print("No data found for backtesting")
        return pd.DataFrame(), pd.DataFrame()
    
    required_cols = ['accel_x', 'accel_y', 'accel_z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Running predictions on {len(df)} data points...")
    
    df_predict, loading_info = main_func(df, predictor_mode, predictor_pattern)
    
    if not df_predict.empty:
        loading_info = analyze_loading_events(df_predict)
    
    _print_backtest_results(df_predict, loading_info)
    
    if save_csv:
        _save_backtest_csv(df_predict, loading_info, client_id, date_str, csv_output_dir)
    
    return df_predict, loading_info


def run_backtest_multi_date(client_id, start_date, end_date, predictor_mode, predictor_pattern, 
                            main_func, save_csv=False, csv_output_dir=None):
    print("="*50)
    print("BACKTEST MODE - SINGLE CLIENT, MULTIPLE DATES")
    print("="*50)
    print(f"Client: {client_id}")
    print(f"Date range: {start_date} to {end_date}")
    print("="*50)
    
    dates = generate_date_range(start_date, end_date)
    print(f"Processing {len(dates)} dates...")
    
    all_predictions = []
    all_loading_events = []
    
    for date_str in dates:
        print(f"\nProcessing date: {date_str}")
        print("-"*30)
        
        df_predict, loading_info = run_backtest_single(
            client_id, date_str, predictor_mode, predictor_pattern, main_func,
            save_csv=False, csv_output_dir=None
        )
        
        if not df_predict.empty:
            all_predictions.append(df_predict)
        
        if not loading_info.empty:
            all_loading_events.append(loading_info)
    
    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()
    
    print("\n" + "="*50)
    print("MULTI-DATE BACKTEST SUMMARY")
    print("="*50)
    _print_backtest_results(final_predictions, final_loading_events)
    
    if save_csv:
        date_range_str = f"{start_date}_to_{end_date}"
        _save_backtest_csv(final_predictions, final_loading_events, client_id, date_range_str, csv_output_dir)
    
    return final_predictions, final_loading_events


def run_backtest_all_clients(date_str, predictor_mode, predictor_pattern, main_func, 
                             client_configs=None, save_csv=False, csv_output_dir=None):
    if client_configs is None:
        client_configs = BACKTEST_CLIENT_CONFIGS
    
    print("="*50)
    print("BACKTEST MODE - ALL CLIENTS, SINGLE DATE")
    print("="*50)
    print(f"Date: {date_str}")
    print(f"Clients to process: {list(client_configs.keys())}")
    print("="*50)
    
    all_predictions = []
    all_loading_events = []
    
    for client_name, config in client_configs.items():
        truck_range = config['truck_range']
        client_ids = [f"{client_name}_{i}" for i in truck_range]
        
        print(f"\n{'='*50}")
        print(f"Processing client: {client_name} ({len(client_ids)} trucks)")
        print('='*50)
        
        for client_id in client_ids:
            print(f"\nTruck: {client_id}")
            print("-"*30)
            
            df_predict, loading_info = run_backtest_single(
                client_id, date_str, predictor_mode, predictor_pattern, main_func,
                save_csv=False, csv_output_dir=None
            )
            
            if not df_predict.empty:
                all_predictions.append(df_predict)
            
            if not loading_info.empty:
                all_loading_events.append(loading_info)
    
    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()
    
    print("\n" + "="*50)
    print("ALL CLIENTS BACKTEST SUMMARY")
    print("="*50)
    _print_backtest_results(final_predictions, final_loading_events)
    
    if save_csv:
        _save_backtest_csv(final_predictions, final_loading_events, "all_clients", date_str, csv_output_dir)
    
    return final_predictions, final_loading_events


def run_backtest_all_clients_multi_date(start_date, end_date, predictor_mode, predictor_pattern, 
                                       main_func, client_configs=None, save_csv=False, csv_output_dir=None):
    if client_configs is None:
        client_configs = BACKTEST_CLIENT_CONFIGS
    
    print("="*50)
    print("BACKTEST MODE - ALL CLIENTS, MULTIPLE DATES")
    print("="*50)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Clients to process: {list(client_configs.keys())}")
    print("="*50)
    
    dates = generate_date_range(start_date, end_date)
    print(f"Processing {len(dates)} dates...")
    
    all_predictions = []
    all_loading_events = []
    
    for date_str in dates:
        print(f"\n{'='*50}")
        print(f"Processing date: {date_str}")
        print('='*50)
        
        df_predict, loading_info = run_backtest_all_clients(
            date_str, predictor_mode, predictor_pattern, main_func,
            client_configs=client_configs, save_csv=False, csv_output_dir=None
        )
        
        if not df_predict.empty:
            all_predictions.append(df_predict)
        
        if not loading_info.empty:
            all_loading_events.append(loading_info)
    
    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()
    
    print("\n" + "="*50)
    print("ALL CLIENTS MULTI-DATE BACKTEST SUMMARY")
    print("="*50)
    _print_backtest_results(final_predictions, final_loading_events)
    
    if save_csv:
        date_range_str = f"{start_date}_to_{end_date}"
        _save_backtest_csv(final_predictions, final_loading_events, "all_clients", date_range_str, csv_output_dir)
    
    return final_predictions, final_loading_events


def _print_backtest_results(df_predict, loading_info):
    """Helper function to print backtest results."""
    print(f"Predictions generated: {len(df_predict)}")
    print(f"Loading events detected: {len(loading_info)}")
    
    if not df_predict.empty and 'pattern' in df_predict.columns:
        pattern_counts = df_predict['pattern'].value_counts()
        print("\nPattern distribution:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count} ({count/len(df_predict)*100:.1f}%)")
    
    if not loading_info.empty:
        total_pallets = loading_info['pallet_count'].sum()
        print(f"\nTotal pallets detected: {total_pallets}")
        print(f"Number of loading events: {len(loading_info)}")
        
        if 'Client_ID' in loading_info.columns:
            client_breakdown = loading_info.groupby('Client_ID').agg({
                'pallet_count': 'sum',
                'start_time': 'count'
            }).rename(columns={'start_time': 'event_count'})
            print("\nLoading events by client:")
            for client_id, row in client_breakdown.iterrows():
                print(f"  {client_id}: {row['event_count']} events, {row['pallet_count']} pallets")


def _save_backtest_csv(df_predict, loading_info, client_id, date_identifier, csv_output_dir):
    """Helper function to save backtest results - predictions as parquet, loading events as CSV."""
    print("\n" + "="*50)
    print("SAVING BACKTEST RESULTS")
    print("="*50)
    
    if csv_output_dir is None:
        csv_output_dir = Path('./backtest_results')
    else:
        csv_output_dir = Path(csv_output_dir)
    
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = csv_output_dir / 'Backtest'/ 'Predictions'
    results_dir = csv_output_dir / 'Backtest'/ 'Results'
    
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not df_predict.empty:
        essential_cols = ['Client_ID', 'Time', 'pattern']
        if 'predictor_mode' in df_predict.columns:
            essential_cols.append('predictor_mode')
        
        available_cols = [col for col in essential_cols if col in df_predict.columns]
        df_to_save = df_predict[available_cols].copy()
        
        predictions_filename = f"{client_id}_{date_identifier}_predictions_{timestamp_str}.parquet"
        predictions_path = predictions_dir / predictions_filename
        df_to_save.to_parquet(predictions_path, index=False)
        print(f"✓ Predictions PARQUET saved to: {predictions_path}")
        print(f"  Columns: {', '.join(available_cols)}")
        print(f"  Rows: {len(df_to_save)}")
    
    if not loading_info.empty:
        events_filename = f"{client_id}_{date_identifier}_loading_events_{timestamp_str}.csv"
        events_path = results_dir / events_filename
        loading_info.to_csv(events_path, index=False)
        print(f"✓ Loading events CSV saved to: {events_path}")
        print(f"  Columns: {', '.join(loading_info.columns)}")
        print(f"  Rows: {len(loading_info)}")
    
    print(f"\nFiles saved to:")
    print(f"  Predictions (parquet): {predictions_dir.absolute()}")
    print(f"  Results (csv): {results_dir.absolute()}")