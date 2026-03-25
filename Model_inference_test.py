import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from ShockInference import ShockInference
from utilities import mode_postprocessing, analyze_loading_events, correct_short_road_segments, remove_false_positive_loading
from pathlib import Path
from tqdm import tqdm

# -----------------------
# Initialize models ONCE
# -----------------------
np.random.seed(42)

# Stage 1 model (Mode)
stage_1_dir = "saved_models/stage_1_mode/v2"
mode_window_size = 120
mode_step_size = 26
predictor_mode = ShockInference(stage_1_dir, mode_window_size, mode_step_size, 'Mode')
print("Stage_1 model successfully initiated")

# Stage 2 model (Pattern)
stage_2_dir = "saved_models/stage_2_pattern/v2"
pattern_window_size = 500
pattern_step_size = 26
predictor_pattern = ShockInference(stage_2_dir, pattern_window_size, pattern_step_size, 'Pattern')
print("Stage_2 model successfully initiated")

# -----------------------
# Configuration
# -----------------------
# INPUT_FILE = r"D:\Logs\Shock_model_log\input_data.csv"   # <-- Change to your file path
# OUTPUT_DIR = r"D:\\Prediction Plots\\Lindt"


def load_data(file_path):
    """Load shock data from a CSV, Parquet, or Excel file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = path.suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.parquet':
        df = pd.read_parquet(file_path)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .parquet, or .xlsx")

    print(f"Loaded {len(df)} rows from {path.name}")
    print(f"Columns: {list(df.columns)}")

    required = ['Time', 'accel_x', 'accel_y', 'accel_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if 'Client_ID' not in df.columns:
        df['Client_ID'] = path.stem
        print(f"No 'Client_ID' column found — assigned '{path.stem}' to all rows")

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.sort_values('Time').reset_index(drop=True)

    return df


def save_predictions_to_csv(df_predictions, loading_events, output_dir):
    """Save predictions and loading events to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    if not df_predictions.empty:
        predictions_file = output_path / f"predictions_{timestamp_str}.csv"
        df_predictions.to_csv(predictions_file, index=False)
        print(f"Saved predictions to: {predictions_file}")
        print(f"  Rows: {len(df_predictions)}")
    else:
        print("No predictions to save")

    if not loading_events.empty:
        loading_file = output_path / f"loading_events_{timestamp_str}.csv"
        loading_events.to_csv(loading_file, index=False)
        print(f"Saved loading events to: {loading_file}")
        print(f"  Rows: {len(loading_events)}")
    else:
        print("No loading events to save")




def plot_predictions(df_pred, output_dir, client_id=None):
    """Plot predicted shock data as an interactive HTML file with color-coded lines."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter client
    if client_id:
        df_plot = df_pred[df_pred['Client_ID'] == client_id].copy()
    else:
        df_plot = df_pred.copy()

    if df_plot.empty:
        print("No data to plot")
        return

    # Labels
    label = (
        client_id
        or df_plot['Client_ID'].iloc[0]
        if 'Client_ID' in df_plot.columns else "unknown"
    )
    date_str = pd.to_datetime(df_plot['Time'].iloc[0]).strftime('%Y-%m-%d')

    # Prediction column
    color_col = None
    if 'pattern' in df_plot.columns:
        color_col = 'pattern'
    elif 'predictor_mode' in df_plot.columns:
        color_col = 'predictor_mode'

    # Color map
    color_map = {
        'parking': '#1f77b4',
        'road': '#ff7f0e',
        'loading': '#2ca02c',
        'unloading': '#d62728',
        'unknown': '#999999',
    }

    # Time formatting
    df_plot['Time'] = pd.to_datetime(df_plot['Time'])
    time_list = df_plot['Time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').tolist()

    # ----------- BUILD COLORED TRACES (PYTHON SIDE) -----------
    traces = []

    if color_col and color_col in df_plot.columns:
        labels = df_plot[color_col].fillna('unknown').tolist()

        for accel_col in ['accel_x', 'accel_y', 'accel_z']:
            if accel_col not in df_plot.columns:
                continue

            values = df_plot[accel_col].tolist()

            seg_start = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[seg_start]:
                    seg_label = labels[seg_start]

                    traces.append({
                        'x': time_list[seg_start:i],
                        'y': [
                            float(v) if pd.notna(v) else None
                            for v in values[seg_start:i]
                        ],
                        'name': f"{accel_col} — {seg_label}",
                        'color': color_map.get(seg_label, '#cccccc')
                    })

                    seg_start = i

            # Last segment
            seg_label = labels[seg_start]
            traces.append({
                'x': time_list[seg_start:],
                'y': [
                    float(v) if pd.notna(v) else None
                    for v in values[seg_start:]
                ],
                'name': f"{accel_col} — {seg_label}",
                'color': color_map.get(seg_label, '#cccccc')
            })

    # ----------- PACKAGE DATA -----------
    plot_data = {
        'traces': traces,
        'color_map': color_map,
        'label': label,
        'date': date_str,
    }

    json_str = json.dumps(plot_data, separators=(',', ':'))

    # ----------- HTML TEMPLATE -----------
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Predictions — {label} — {date_str}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #fff; height: 100vh; overflow: hidden; }}
        #plotDiv {{ width: 100%; height: 100vh; }}
        .loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #666; font-size: 18px; }}
    </style>
</head>
<body>
    <div id="plotDiv"></div>
    <div id="status" class="loading">Loading plot...</div>

    <script>
        const plotData = {json_str};

        document.addEventListener('DOMContentLoaded', function() {{
            const data = plotData;
            const traces = [];

            // Build traces from Python-prepared segments
            data.traces.forEach(t => {{
                traces.push({{
                    x: t.x,
                    y: t.y,
                    mode: 'lines',
                    name: t.name,
                    line: {{ color: t.color, width: 1 }},
                    showlegend: false
                }});
            }});

            // Legend (prediction classes)
            const legendLabels = [...new Set(data.traces.map(t => t.name.split(' — ')[1]))];

            legendLabels.forEach(lbl => {{
                traces.push({{
                    x: [null],
                    y: [null],
                    mode: 'lines',
                    name: lbl + ' (pred)',
                    line: {{ color: data.color_map[lbl] || '#cccccc', width: 8 }},
                    showlegend: true
                }});
            }});

            const layout = {{
                title: {{
                    text: 'Predictions — ' + data.label + ' — ' + data.date,
                    x: 0.5,
                    y: 0.98,
                    font: {{ size: 16 }}
                }},
                xaxis: {{
                    title: 'Time',
                    type: 'date',
                    tickformat: '%H:%M:%S'
                }},
                yaxis: {{
                    title: 'Acceleration'
                }},
                margin: {{ t: 40, r: 20, b: 40, l: 60 }},
                hovermode: 'closest',
                showlegend: true,
                autosize: true,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white'
            }};

            const config = {{
                displayModeBar: true,
                displaylogo: false,
                responsive: true,
                scrollZoom: true,
                doubleClick: 'reset'
            }};

            Plotly.newPlot('plotDiv', traces, layout, config);
            window.addEventListener('resize', () => Plotly.Plots.resize('plotDiv'));
            document.getElementById('status').style.display = 'none';
        }});
    </script>
</body>
</html>"""

    # ----------- SAVE FILE -----------
    filename = f"predictions_plot_{label}_{date_str}.html"
    file_path = output_path / filename

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Saved plot to: {file_path} (~{len(html_content)/1024:.1f} KB)")


def run_inference(df, predictor_mode, predictor_pattern):
    """Run Stage 1 and Stage 2 inference per client with tqdm progress."""
    client_ids = df['Client_ID'].unique()
    print(f"Found {len(client_ids)} client(s) in data")

    all_predictions = []
    all_loading_events = []

    for client_id in tqdm(client_ids, desc="Overall progress", unit="client"):
        df_client = df[df['Client_ID'] == client_id].copy().reset_index(drop=True)

        if df_client.empty:
            continue

        # -------- Stage 1: Mode inference --------
        tqdm.write(f"[{client_id}] Stage 1 — Mode inference ({len(df_client)} rows)...")
        predictions, confidences, avg_probs = predictor_mode.predict_timestamp_level_multi_scale(df_client)
        df_pred = mode_postprocessing(df_client, predictor_mode, predictions, avg_probs)
        df_pred['Client_ID'] = client_id
        tqdm.write(f"[{client_id}] Stage 1 done")

        # -------- Stage 2: Pattern inference on long parking --------
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
                tqdm.write(f"[{client_id}] Stage 2 — Pattern inference ({len(df_parking)} parking rows)...")
                predictions1, confidences1, avg_probs1 = predictor_pattern.predict_timestamp_level_multi_scale(
                    df_parking)
                predictions1 = np.asarray(predictions1).ravel()

                if len(predictions1) != len(df_parking):
                    raise ValueError(
                        f"Predictions length ({len(predictions1)}) != parking rows ({len(df_parking)})"
                    )

                df_pred.loc[long_parking_mask, 'pattern'] = pd.Series(predictions1, index=df_parking.index)
                df_pred.loc[df_pred["pattern"].isna() | (df_pred["pattern"] == "unknown"), "pattern"] = df_pred.get(
                    "predictor_mode", "unknown")

                # ============================================================
                # CORRECTION STEP 1: Fix short road blips between loading
                # (road segments <= 60s next to loading/unloading → relabel)
                # ============================================================
                tqdm.write(f"[{client_id}] Correcting short road segments...")
                df_pred = correct_short_road_segments(
                    df_pred,
                    pattern_col='pattern',
                    time_col='Time',
                    max_duration_sec=60,
                )

                # ============================================================
                # CORRECTION STEP 2: Remove false-positive loading labels
                # (loading segments with flat/no-spike signal → parking)
                # ============================================================
                tqdm.write(f"[{client_id}] Removing false-positive loading segments...")
                df_pred = remove_false_positive_loading(
                    df_pred,
                    pattern_col='pattern',
                    time_col='Time',
                    accel_cols=('accel_x', 'accel_y', 'accel_z'),
                    energy_method='std',  # 'std' or 'ptp'
                    threshold=None,  # None = auto-compute from parking baseline
                    auto_threshold_factor=2.0,  # tune this: higher = more lenient
                    relabel_to='parking',
                )

                loading_info = analyze_loading_events(df_pred)
                tqdm.write(f"[{client_id}] Stage 2 done — {len(loading_info)} loading events found")
            else:
                tqdm.write(f"[{client_id}] No long parking segments, skipping Stage 2")
                df_pred['pattern'] = df_pred.get('predictor_mode', 'unknown')
        else:
            tqdm.write(f"[{client_id}] No parking segments, skipping Stage 2")
            df_pred['pattern'] = df_pred.get('predictor_mode', 'unknown')

        all_predictions.append(df_pred)
        if not loading_info.empty:
            loading_info['Client_ID'] = client_id
            all_loading_events.append(loading_info)

    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_loading_events = pd.concat(all_loading_events, ignore_index=True) if all_loading_events else pd.DataFrame()

    return final_predictions, final_loading_events


if __name__ == '__main__':


    client = 'Lindt'
    client_no = 'Lindt_7' 
    file_source = 'Shock data'
    file_name = f'{client_no}_2026-03-18.parquet'
    folder_name = (f"C:\\Users\\HarshavardhanMoravap\\KONVOI GmbH\\Communication site - Product Development\\Data\\"
                f"\\Local_database\\2026\\03_March\\{client}\\{file_source}\\{client_no}")
    INPUT_FILE = os.path.join(folder_name,file_name)
    print(f"{file_name} file exists...")
    print("=" * 50)

    OUTPUT_DIR = r"D:\\Prediction Plots\\Lindt"
    print("SHOCK INFERENCE PIPELINE")
    print("=" * 50)
    print(f"Input file : {INPUT_FILE}")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 50)

    # Load data
    df = load_data(INPUT_FILE)

    # Run inference
    predictions, loading_events = run_inference(df, predictor_mode, predictor_pattern)

    # Save CSV results
    print("=" * 50)
    print("SAVING OUTPUT FILES")
    print("=" * 50)
    save_predictions_to_csv(predictions, loading_events, OUTPUT_DIR)

    # Generate plots per client
    if not predictions.empty:
        print("=" * 50)
        print("GENERATING PLOTS")
        print("=" * 50)
        for cid in predictions['Client_ID'].unique():
            plot_predictions(predictions, OUTPUT_DIR, client_id=cid)

    # Summary
    print("=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total predictions: {len(predictions)}")
    print(f"Total loading events: {len(loading_events)}")

    if not loading_events.empty:
        print("Loading events by client:")
        for client_name, count in loading_events['Client_ID'].value_counts().items():
            print(f"  {client_name}: {count} events")
        print(f"Total pallets loaded: {loading_events['pallet_count'].sum()}")

    print("=" * 50)
    print("Pipeline completed successfully!")