import os
import pandas as pd
import pytz
from datetime import datetime
from pathlib import Path
import calendar

class HourlyDataManager:
    
    def __init__(self, base_data_dir=None):
        if base_data_dir is None:
            self.base_data_dir = Path(r"C:\Users\HarshavardhanMoravap\KONVOI GmbH\Communication site - Product Development\Data\Local_database")
        else:
            self.base_data_dir = Path(base_data_dir)
        
        print(f"Initialized HourlyDataManager with base directory: {self.base_data_dir}")
    
    def extract_client_name(self, client_id):
        if '_' in client_id:
            return client_id.split('_')[0]
        else:
            return client_id
    
    def save_predictions(self, df_predictions, timestamp=None):
        """Save predictions data for all clients in the dataframe."""
        if df_predictions.empty or 'Client_ID' not in df_predictions.columns:
            print("No predictions data to save")
            return True
        
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('Europe/Berlin'))
        elif timestamp.tzinfo is None:
            timestamp = pytz.timezone('Europe/Berlin').localize(timestamp)
        else:
            timestamp = timestamp.astimezone(pytz.timezone('Europe/Berlin'))
        
        year = timestamp.strftime('%Y')
        month_num = timestamp.strftime('%m')
        month_name = calendar.month_name[timestamp.month]
        month_folder = f"{month_num}_{month_name}"
        date_str = timestamp.strftime('%Y-%m-%d')
        
        success_count = 0
        total_clients = df_predictions['Client_ID'].nunique()
        
        for client_id in df_predictions['Client_ID'].unique():
            try:
                client_name = self.extract_client_name(client_id)

                file_path = (self.base_data_dir / year / month_folder / client_name / 
                           "Shock data predicted" / client_id / f"{client_id}_{date_str}.parquet")
                
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                client_data = df_predictions[df_predictions['Client_ID'] == client_id].copy()
                
                # client_data['processed_at'] = datetime.now(pytz.utc).isoformat()
                # client_data['data_date'] = date_str
                
                if 'Time' in client_data.columns:
                    client_data['Time'] = pd.to_datetime(client_data['Time'])
                
                if file_path.exists():
                    existing_df = pd.read_parquet(file_path)
                    combined_df = pd.concat([existing_df, client_data], ignore_index=True)
                    
                    if 'Time' in combined_df.columns:
                        combined_df = combined_df.drop_duplicates(subset=['Time', 'Client_ID'], keep='last')
                        combined_df = combined_df.sort_values('Time').reset_index(drop=True)
                    
                    client_data = combined_df
                    print(f"Appended data for {client_id}. Total rows: {len(client_data)}")
                else:
                    print(f"Created new file for {client_id}")
                
                client_data.to_parquet(file_path, index=False, compression='snappy')
                print(f"Saved {len(client_data)} rows for {client_id}")
                success_count += 1
                
            except Exception as e:
                print(f"Error saving data for {client_id}: {e}")
        
        print(f"Successfully saved data for {success_count}/{total_clients} clients")
        return success_count == total_clients
    
    def read_data(self, client_id, date_str):
        """Read saved data for a specific client and date."""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.strftime('%Y')
            month_num = date_obj.strftime('%m')
            month_name = calendar.month_name[date_obj.month]
            month_folder = f"{month_num}_{month_name}"
            
            client_name = self.extract_client_name(client_id)
            
            file_path = (self.base_data_dir / year / month_folder / client_name / 
                        "Shock data predicted" / client_id / f"{client_id}_{date_str}.parquet")
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                print(f"Loaded {len(df)} rows for {client_id} on {date_str}")
                return df
            else:
                print(f"No file found for {client_id} on {date_str}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading data for {client_id} on {date_str}: {e}")
            return pd.DataFrame()