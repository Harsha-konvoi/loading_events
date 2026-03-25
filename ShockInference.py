import os
import numpy as np
from tqdm import tqdm
import torch
import joblib
from model_architecture import PositionalEncoding, ImprovedTransformerEncoder, EnhancedLabelingStrategies

class ShockInference:
    def __init__(self,model_path,window_size,step_size,model_type):
        self.model_path = model_path
        self.window_size = window_size
        self.step_size = step_size
        self.model_type = model_type
        self.window_extraction = EnhancedLabelingStrategies(window_size=self.window_size,step_size=self.step_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.label_encoder = joblib.load(os.path.join(self.model_path,f"{self.model_type}_transformer_label_encoder.pkl"))
        self.normalization_params = joblib.load(os.path.join(self.model_path,f"{self.model_type}_transformer_normalization_params.pkl"))
        checkpoint = torch.load(os.path.join(self.model_path,f"{self.model_type}_transformer_best_model.pth"),map_location = self.device, weights_only=False)
        self.model = ImprovedTransformerEncoder(input_dim=3, hidden_dim=128, num_classes=len(self.label_encoder.classes_)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        #print(f"{self.model_type} model is loaded ...........")


    def predict_timestamp_level_multi_scale(self, df):
        """
        Predicts timestamp-level trailer modes using multi-scale voting windows.
        """
        if self.step_size is None or self.step_size == 0:
            self.step_size = self.window_size
    
        df = df.copy()
        N = len(df)
        feats = df[['accel_x', 'accel_y', 'accel_z']].values.astype(np.float32)
    
        # Extract windows
        sequences, indices = [], []
        for start in range(0, N - self.window_size + 1, self.step_size):
            end = start + self.window_size
            sequences.append(feats[start:end])
            indices.append((start, end))
        if len(sequences) == 0:
            raise ValueError("No valid sequences extracted from data.")
    
        # Normalize
        def _normalize(x):
            return (x - self.normalization_params['mean']) / self.normalization_params['std']
        sequences = [_normalize(seq) for seq in sequences]
    
        # Initialize arrays to collect per-timestamp scores
        num_classes = len(self.label_encoder.classes_)
        all_probs = np.zeros((len(df), num_classes))
        vote_counts = np.zeros(len(df))
    
        probs_per_window = []
        with torch.no_grad():
            for i in range(0, len(sequences), 64):
                batch = np.stack(sequences[i:i+64], axis=0)  # (B, T, 3)
                x = torch.tensor(batch, dtype=torch.float32, device=self.device)
                out = self.model(x)                           # (B, T, C)
                p = torch.softmax(out, dim=-1).cpu().numpy()
                probs_per_window.extend([p[j] for j in range(p.shape[0])])
    
        # Aggregate probs (fixed_120 mode)
        all_probs = np.zeros((N, num_classes), dtype=np.float32)
        vote_counts = np.zeros(N, dtype=np.float32)
    
        for (start, end), p in zip(indices, probs_per_window):
            all_probs[start:end] += p
            vote_counts[start:end] += 1.0
    
        vote_counts[vote_counts == 0] = 1.0
        avg_probs = all_probs / vote_counts[:, None]
    
        # Predictions & confidences
        pred_idx = avg_probs.argmax(axis=1)
        predictions = self.label_encoder.inverse_transform(pred_idx)
        confidences = avg_probs.max(axis=1)
    
        return predictions, confidences, avg_probs


    def predict_timestamp_level_pattern(self, df):
        """
        Predicts timestamp-level trailer modes using multi-scale voting windows.
        """
        df = df.copy()
        #df['mode'] = 'unknown'  # dummy label required by the strategy
        N = len(df)
        feats = df[['accel_x', 'accel_y', 'accel_z']].values.astype(np.float32)
        
        sequences, indices = [], []
        for start in range(0, N - self.window_size + 1, self.step_size):
            end = start + self.window_size
            sequences.append(feats[start:end])
            indices.append((start, end))
        if len(sequences) == 0:
            raise ValueError("No valid sequences extracted from data.")
    
        def _normalize(x):
            return (x - self.normalization_params['mean']) / self.normalization_params['std']
        sequences = [_normalize(seq) for seq in sequences]
    
        # Initialize arrays to collect per-timestamp scores
        num_classes = len(self.label_encoder.classes_)
        all_probs = np.zeros((len(df), num_classes))
        vote_counts = np.zeros(len(df))
    
        # Batched inference with tqdm for progress monitoring
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), 64), desc="Running Inference"):
                batch_seqs = sequences[i:i+64]
                max_len = max(seq.shape[0] for seq in batch_seqs)
        
                # Pad sequences to max length in batch
                padded_batch = np.array([
                    np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), mode='constant')
                    for seq in batch_seqs
                ])
        
                batch_tensor = torch.tensor(padded_batch, dtype=torch.float32).to(self.device)
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=2).cpu().numpy()
        
                for j, (start, end) in enumerate(indices[i:i+64]):
                    length = end - start
                    all_probs[start:end] += probs[j, :length]
                    vote_counts[start:end] += 1
    
        # Avoid division by zero
        vote_counts[vote_counts == 0] = 1
        avg_probs = all_probs / vote_counts[:, None]
    
        # Final predictions
        pred_indices = np.argmax(avg_probs, axis=1)
        predictions = self.label_encoder.inverse_transform(pred_indices)
        confidences = np.max(avg_probs, axis=1)
    
        return predictions, confidences

    def extract_windows_multi_scale(self, df):
        """
        Use multi-scale voting strategy to extract windows.
        """
        df = df.copy()
        df['mode'] = 'unknown'  # Dummy label needed by the strategy
        sequences, _, indices = self.window_extraction.strategy_4_multi_scale_voting(df)
        return sequences, indices  # <-- Don't wrap in np.array()
