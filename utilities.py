import numpy as np
import pandas as pd


class RealTimeSmoother:
    """
    Optimized version — eliminates per-row sklearn label_encoder calls
    by working entirely with integer indices internally.
    """

    def __init__(self, label_encoder, min_run_by_label=None, margin=0.0, cooldown=0, patience=3):
        self.label_encoder = label_encoder
        self.min_run_by_label = min_run_by_label or {}
        self.patience = patience
        self.cooldown = cooldown

        n_classes = len(label_encoder.classes_)
        if isinstance(margin, dict):
            self.margin_arr = np.array([
                margin.get(label_encoder.classes_[i], 0.0)
                for i in range(n_classes)
            ], dtype=np.float64)
        else:
            self.margin_arr = np.full(n_classes, float(margin), dtype=np.float64)

        self.current_idx = -1
        self.counter = 0

    def update_idx(self, probs):
        """Fast update using only integer indices — no sklearn calls."""
        top_idx = int(probs.argmax())

        if self.current_idx < 0:
            self.current_idx = top_idx
            self.counter = 1
            return top_idx

        if top_idx == self.current_idx:
            self.counter = 0
            return self.current_idx

        margin_ok = (probs[top_idx] - probs[self.current_idx]) >= self.margin_arr[self.current_idx]

        if margin_ok:
            self.counter += 1
            if self.counter >= self.patience:
                self.current_idx = top_idx
                self.counter = 0
                return top_idx
        else:
            self.counter = 0

        return self.current_idx


def mode_postprocessing(df, predictor_mode, predictions, avg_probs):
    """Optimized post-processing with vectorized instability + fast smoother."""
    labels = predictor_mode.label_encoder.classes_

    label_confidence = {label: float(np.mean(avg_probs[:, i])) for i, label in enumerate(labels)}

    pred_indices = np.argmax(avg_probs, axis=1)

    # --- Instability score: vectorized run-length encoding ---
    changes = np.concatenate(([True], pred_indices[1:] != pred_indices[:-1]))
    run_ids = np.cumsum(changes)
    run_lengths = np.bincount(run_ids)
    run_label_idx = pred_indices[changes]

    instability_score = {}
    for i, label in enumerate(labels):
        mask = run_label_idx == i
        runs_for_label = run_lengths[1:][mask]
        instability_score[label] = int(np.sum((runs_for_label > 0) & (runs_for_label < 15)))

    base_margin = 0.25
    max_extra_margin = 0.5
    dynamic_margin = {}
    for label in labels:
        instability_factor = instability_score[label] / max(
            1, instability_score[label] + label_confidence[label] * 100
        )
        dynamic_margin[label] = min(base_margin + instability_factor * max_extra_margin, 0.75)

    smoother = RealTimeSmoother(
        predictor_mode.label_encoder,
        min_run_by_label={label: 10 for label in labels},
        margin=dynamic_margin,
        cooldown=10,
        patience=3,
    )

    n = len(avg_probs)
    stable_indices = np.empty(n, dtype=np.int64)
    for i in range(n):
        stable_indices[i] = smoother.update_idx(avg_probs[i])

    smoothed_predictions = predictor_mode.label_encoder.inverse_transform(stable_indices)
    df['predictor_mode'] = smoothed_predictions
    return df


def remove_false_positive_loading(df, pattern_col='pattern', time_col='Time',
                                  accel_cols=('accel_x', 'accel_y', 'accel_z'),
                                  energy_method='std',
                                  threshold=None,
                                  auto_threshold_factor=2.0,
                                  relabel_to='parking'):
    """
    Remove false-positive 'loading'/'unloading' segments that have no actual
    physical activity — the accelerometer signal is flat (stationary), but the
    model wrongly predicted loading.

    How it works:
      1. Compute a per-row acceleration magnitude:  sqrt(x² + y² + z²)
      2. Build run-length encoding of the pattern column.
      3. For each 'loading' or 'unloading' segment, compute an energy metric
         (std or peak-to-peak of the magnitude within that segment).
      4. Compare against a threshold. If below → relabel to 'parking'.

    Threshold selection:
      - If `threshold` is given explicitly, use that value.
      - Otherwise, auto-compute: take the energy metric of all 'parking' segments
        and set threshold = mean_parking_energy + auto_threshold_factor * std.
        This means: if a "loading" segment has energy indistinguishable from
        parking, it's a false positive.

    Args:
        df:                     DataFrame with pattern, time, and accel columns.
        pattern_col:            Column with pattern labels.
        time_col:               Datetime column.
        accel_cols:             Tuple of acceleration column names (x, y, z).
        energy_method:          'std' (standard deviation) or 'ptp' (peak-to-peak).
                                'std' is more robust to single outlier samples.
        threshold:              Explicit energy threshold. If None, auto-computed
                                from parking segments.
        auto_threshold_factor:  Number of std deviations above parking mean to
                                set the auto threshold (default 2.0).
        relabel_to:             Label for false positives (default 'parking').

    Returns:
        df with corrected pattern column (modified in place).
    """
    if pattern_col not in df.columns or len(df) == 0:
        return df

    # Verify accel columns exist
    available_accel = [c for c in accel_cols if c in df.columns]
    if not available_accel:
        print(f"  Warning: no acceleration columns found ({accel_cols}), skipping FP removal")
        return df

    # --- Compute acceleration magnitude ---
    accel_values = np.column_stack([df[c].values.astype(np.float64) for c in available_accel])
    magnitude = np.sqrt(np.sum(accel_values ** 2, axis=1))

    # --- Run-length encoding ---
    patterns = df[pattern_col].values.copy()
    is_change = np.concatenate(([True], patterns[1:] != patterns[:-1]))
    run_ids = np.cumsum(is_change)
    n_runs = run_ids[-1]

    run_start_idx = np.searchsorted(run_ids, np.arange(1, n_runs + 1), side='left')
    run_end_idx = np.searchsorted(run_ids, np.arange(1, n_runs + 1), side='right') - 1
    run_labels = patterns[run_start_idx]

    # --- Compute energy per segment ---
    def segment_energy(start, end):
        seg = magnitude[start:end + 1]
        if len(seg) < 2:
            return 0.0
        if energy_method == 'ptp':
            return float(np.ptp(seg))
        else:  # 'std'
            return float(np.std(seg))

    run_energies = np.array([
        segment_energy(run_start_idx[r], run_end_idx[r])
        for r in range(n_runs)
    ])

    # --- Auto-compute threshold from parking segments if not given ---
    if threshold is None:
        parking_mask = np.array([l == 'parking' for l in run_labels])
        if parking_mask.any():
            parking_energies = run_energies[parking_mask]
            threshold = float(np.mean(parking_energies) + auto_threshold_factor * np.std(parking_energies))
            print(f"  Auto-threshold for loading FP removal: {threshold:.6f} "
                  f"(parking mean={np.mean(parking_energies):.6f}, "
                  f"std={np.std(parking_energies):.6f}, factor={auto_threshold_factor})")
        else:
            # No parking segments to calibrate against — use a conservative fallback
            threshold = float(np.median(run_energies) * 0.5)
            print(f"  No parking segments for calibration, using fallback threshold: {threshold:.6f}")

    # --- Filter false-positive loading/unloading ---
    loading_labels = {'loading', 'unloading'}
    pattern_col_idx = df.columns.get_loc(pattern_col)
    corrected_count = 0

    for r in range(n_runs):
        if run_labels[r] not in loading_labels:
            continue

        if run_energies[r] < threshold:
            start = run_start_idx[r]
            end = run_end_idx[r] + 1
            df.iloc[start:end, pattern_col_idx] = relabel_to
            corrected_count += 1

    if corrected_count > 0:
        print(f"  Removed {corrected_count} false-positive loading segment(s) "
              f"(energy < {threshold:.6f}) → '{relabel_to}'")
    else:
        print(f"  No false-positive loading segments found (threshold={threshold:.6f})")

    return df

def correct_short_road_segments(df, pattern_col='pattern', time_col='Time',
                                max_duration_sec=60, relabel_to='loading'):
    """
    Relabel short 'road' segments that sit next to loading/unloading.

    After Stage 2 prediction, the model sometimes briefly predicts 'road' for
    30-60 seconds in the middle of a loading/unloading session. This is physically
    impossible — a truck cannot leave and return in under a minute. This function
    finds those spurious road blips and corrects them.

    Rules applied:
      1. Build a run-length encoding of the pattern column.
      2. For every 'road' run whose duration <= max_duration_sec:
         - If the segment immediately BEFORE is loading/unloading → relabel
         - If the segment immediately AFTER  is loading/unloading → relabel
      3. The relabel target is the neighbor's label (loading or unloading),
         so the corrected segment merges naturally with its context.
         If both neighbors qualify, the PREVIOUS neighbor's label wins
         (the truck was already doing that activity).

    Args:
        df:               DataFrame sorted by time, with pattern_col and time_col.
        pattern_col:      Column with pattern labels ('road', 'loading', etc.).
        time_col:         Datetime column.
        max_duration_sec: Road segments shorter than this get corrected (default 60s).
        relabel_to:       Fallback label if no neighbor context (default 'loading').
                          In practice, the neighbor's actual label is used.

    Returns:
        df with corrected pattern column (modified in place).
    """
    if pattern_col not in df.columns or len(df) == 0:
        return df

    df[time_col] = pd.to_datetime(df[time_col])
    patterns = df[pattern_col].values.copy()
    times = df[time_col].values

    # --- Run-length encoding (vectorized) ---
    is_change = np.concatenate(([True], patterns[1:] != patterns[:-1]))
    run_ids = np.cumsum(is_change)
    n_runs = run_ids[-1]

    # Start/end index of each run
    run_start_idx = np.searchsorted(run_ids, np.arange(1, n_runs + 1), side='left')
    run_end_idx = np.searchsorted(run_ids, np.arange(1, n_runs + 1), side='right') - 1

    run_labels = patterns[run_start_idx]
    run_durations_sec = (
        (times[run_end_idx] - times[run_start_idx]) / np.timedelta64(1, 's')
    )

    loading_labels = {'loading', 'unloading'}
    corrected_count = 0
    pattern_col_idx = df.columns.get_loc(pattern_col)

    for r in range(n_runs):
        if run_labels[r] != 'road':
            continue
        if run_durations_sec[r] > max_duration_sec:
            continue

        # Check neighbors
        prev_label = run_labels[r - 1] if r > 0 else None
        next_label = run_labels[r + 1] if r < n_runs - 1 else None

        prev_is_loading = prev_label in loading_labels
        next_is_loading = next_label in loading_labels

        if not (prev_is_loading or next_is_loading):
            continue

        # Pick the best relabel: prefer previous neighbor's label
        if prev_is_loading:
            new_label = prev_label
        else:
            new_label = next_label

        # Apply correction
        start = run_start_idx[r]
        end = run_end_idx[r] + 1
        df.iloc[start:end, pattern_col_idx] = new_label
        corrected_count += 1

    if corrected_count > 0:
        print(f"  Corrected {corrected_count} short road segment(s) "
              f"(<= {max_duration_sec}s adjacent to loading/unloading)")

    return df


def analyze_loading_events(df):
    """Analyze loading events ensuring Client_ID is preserved."""
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'])

    df = df.sort_values(['Time']).reset_index(drop=True)
    loading_sessions = find_loading_sessions(df)

    if loading_sessions:
        output_df = pd.DataFrame(loading_sessions)
        if 'Client_ID' in df.columns and not df.empty:
            output_df['Client_ID'] = df['Client_ID'].iloc[0]
        column_order = ['start_time', 'end_time', 'pattern', 'event', 'pallet_count']
        if 'Client_ID' in output_df.columns:
            column_order = ['Client_ID'] + column_order
        output_df = output_df[column_order]
    else:
        columns = ['start_time', 'end_time', 'pattern', 'event', 'pallet_count']
        if 'Client_ID' in df.columns:
            columns = ['Client_ID'] + columns
        output_df = pd.DataFrame(columns=columns)

    return output_df

def validate_loading_session_accel_z(df, session_start_time, session_end_time):
    """
    Validate if a loading session is genuine based on accel_z values.
    Returns True if ANY accel_z value during the session is < 9.
    """
    session_data = df[(df['Time'] >= session_start_time) & (df['Time'] <= session_end_time)]
    
    if session_data.empty or 'accel_z' not in session_data.columns:
        return False
    
    has_low_accel_z = (session_data['accel_z'] < 9).any()
    return has_low_accel_z


def find_pattern_transitions(df):
    """
    Find all pattern transitions in the dataframe.
    Returns list of transitions with details about what changed.
    """
    patterns = df['pattern'].values
    times = df['Time'].values
    
    transitions = []
    for i in range(1, len(patterns)):
        if patterns[i] != patterns[i-1]:
            transitions.append({
                'index': i,
                'time': times[i],
                'from_pattern': patterns[i-1],
                'to_pattern': patterns[i]
            })
    
    return transitions


def find_loading_sessions(df):
    """
    Enhanced loading session detection with strict pattern transition rules.
    
    Rules:
    1. Loading session must transition from 'parking' to 'loading'
    2. Duration ≥ 15 seconds
    3. Any accel_z < 9 during session
    4. If 'road' appears between loading sessions:
       - Sessions before road with <3 pallets are discarded (likely misclassification)
       - Sessions after road are treated as new sessions (like >1hr gap)
    """
    patterns = df['pattern'].values
    times = df['Time']
    
    # Find all loading segments with their preceding patterns
    loading_segments = []
    i = 0
    while i < len(patterns):
        current_pattern = patterns[i]
        prev_pattern = patterns[i-1] if i > 0 else None
        
        if current_pattern == 'loading' and prev_pattern != 'loading':
            segment_start_time = times.iloc[i]
            segment_start_idx = i
            
            # Find end of loading segment
            while (i + 1 < len(patterns) and patterns[i + 1] == 'loading'):
                i += 1
            
            segment_end_time = times.iloc[i]
            segment_duration = (segment_end_time - segment_start_time).total_seconds()
            
            loading_segments.append({
                'start_time': segment_start_time,
                'end_time': segment_end_time,
                'duration': segment_duration,
                'start_idx': segment_start_idx,
                'end_idx': i,
                'preceding_pattern': prev_pattern
            })
        
        i += 1
    
    if not loading_segments:
        return []
    
    # Apply Rule 1: Only keep segments that transition from parking to loading
    valid_segments = []
    for segment in loading_segments:
        if segment['preceding_pattern'] == 'parking' and segment['duration'] >= 15:
            valid_segments.append(segment)
        else:
            if segment['preceding_pattern'] != 'parking':
                print(f"Loading segment rejected: transition from '{segment['preceding_pattern']}' to loading "
                      f"at {segment['start_time']} (must transition from parking)")
            elif segment['duration'] < 15:
                print(f"Loading segment rejected: duration {segment['duration']:.1f}s < 15s "
                      f"at {segment['start_time']}")
    
    if not valid_segments:
        return []
    
    # Group segments into sessions, considering 'road' interruptions
    sessions = []
    current_session = None
    
    for i, segment in enumerate(valid_segments):
        if current_session is None:
            # Start first session
            current_session = {
                'start_time': segment['start_time'],
                'pattern': 'loading',
                'event': 4,
                'pallet_count': 1,
                'segments': [segment]
            }
        else:
            # Check what happened between last segment and current segment
            last_segment_end = current_session['segments'][-1]['end_time']
            time_gap = (segment['start_time'] - last_segment_end).total_seconds()
            
            # Check if there's any 'road' pattern between segments
            between_mask = (df['Time'] > last_segment_end) & (df['Time'] < segment['start_time'])
            between_data = df[between_mask]
            has_road_between = (between_data['pattern'] == 'road').any() if not between_data.empty else False
            
            if has_road_between or time_gap > 3600:
                # Finalize current session and apply road-related rules
                current_session['end_time'] = current_session['segments'][-1]['end_time']
                
                if has_road_between and current_session['pallet_count'] < 3:
                    # Rule 4a: Discard session before road if <3 pallets
                    print(f"Loading session rejected: <3 pallets ({current_session['pallet_count']}) "
                          f"before road interruption at {current_session['start_time']}")
                else:
                    # Keep the session
                    sessions.append(current_session)
                
                # Start new session after road/time gap
                current_session = {
                    'start_time': segment['start_time'],
                    'pattern': 'loading',
                    'event': 4,
                    'pallet_count': 1,
                    'segments': [segment]
                }
            else:
                # Continue current session (no road, within time limit)
                current_session['pallet_count'] += 1
                current_session['segments'].append(segment)
    
    # Don't forget the last session
    if current_session is not None:
        current_session['end_time'] = current_session['segments'][-1]['end_time']
        sessions.append(current_session)
    
    # Apply accel_z validation to each session
    validated_sessions = []
    for session in sessions:
        is_valid_accel_z = validate_loading_session_accel_z(
            df, session['start_time'], session['end_time']
        )
        
        if is_valid_accel_z:
            # Remove segments from output and preserve Client_ID
            if 'segments' in session:
                del session['segments']
            
            # Ensure Client_ID is preserved if available in original dataframe
            if 'Client_ID' in df.columns and not df.empty:
                session['Client_ID'] = df['Client_ID'].iloc[0]
            
            validated_sessions.append(session)
        else:
            print(f"Loading session rejected due to accel_z validation: "
                  f"{session['start_time']} to {session['end_time']} "
                  f"(no accel_z < 9 found)")
    
    return validated_sessions