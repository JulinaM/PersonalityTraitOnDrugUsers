# import pandas as pd
# import numpy as np
# import torch
# import os
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
# import gc
# import psutil
# import warnings
# import sys

# # --- Conditional GPU Monitoring ---
# # Attempt to import and initialize pynvml, but don't fail if it's not present.
# try:
#     from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
#     nvmlInit()
#     GPU_AVAILABLE = True
# except (ImportError, Exception):
#     GPU_AVAILABLE = False
#     print("Warning: pynvml library not found or failed to initialize. GPU monitoring will be disabled.")


# # --- Ensure custom modules can be found ---
# # This path is from your sample code. Please ensure it is correct in your environment.
# sys.path.insert(0, '/data/jmharja/projects/PersonaClassifier/')
# try:
#     from utils.Models import BiLSTMClassifier
# except ImportError:
#     print("FATAL ERROR: Could not import BiLSTMClassifier.")
#     print("Please ensure the path '/data/jmharja/projects/PersonaClassifier/' is correct and the module exists.")
#     sys.exit(1)

# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')

# # --- Configuration ---
# BASE_PATH = "/data2/julina/scripts/tweets/2020/03/"
# ANALYSIS_DIR = os.path.join(BASE_PATH, "SU_and_NON_SU_analysis/")
# INPUT_FILE = os.path.join(ANALYSIS_DIR, "all_users_classified_combined.csv")
# OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "all_users_classified_with_personality_V3.csv") # Changed to V3

# # Model checkpoint path from your sample code
# CHECKPOINT = '/data/jmharja/projects/PersonaClassifier/checkpoint/v4_bilstm_roberta-2025-01-22_11-14-52_final_eval_the_best_S1/models/'

# # Personality traits to predict
# TARGET_COLUMNS = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']

# # Define the chunk size for processing the large CSV
# CHUNK_SIZE = 100000  # Process 100,000 rows at a time. Adjust based on your RAM.

# def print_memory_stats():
#     """Prints detailed CPU and (if available) GPU memory statistics."""
#     process = psutil.Process(os.getpid())
#     print(f"\nCPU Memory - Used: {process.memory_info().rss / 1024 ** 2:.2f} MB | Available: {psutil.virtual_memory().available / 1024 ** 2:.2f} MB")

#     if GPU_AVAILABLE:
#         try:
#             handle = nvmlDeviceGetHandleByIndex(0)
#             info = nvmlDeviceGetMemoryInfo(handle)
#             print(f"GPU Memory - Used: {info.used / 1024 ** 2:.2f} MB | Free: {info.free / 1024 ** 2:.2f} MB | Total: {info.total / 1024 ** 2:.2f} MB")
#         except Exception as e:
#             print(f"Could not retrieve GPU memory info: {e}")


# class MemorySafeEmbedder:
#     """Generates embeddings with careful memory management."""
#     def __init__(self, model_name='roberta-base'):
#         self.model_name = model_name
#         self.tokenizer = None
#         self.model = None
#         self.device = self._get_safe_device()
#         print(f"Embedder initialized on device: {self.device}")

#     def _get_safe_device(self):
#         """Determines the safest available device (GPU with enough memory or CPU)."""
#         if torch.cuda.is_available() and GPU_AVAILABLE:
#             handle = nvmlDeviceGetHandleByIndex(0)
#             info = nvmlDeviceGetMemoryInfo(handle)
#             # Check for at least 2GB of free VRAM
#             if info.free > 2 * 1024 ** 3:
#                 print("Sufficient GPU VRAM detected. Using CUDA.")
#                 return torch.device('cuda:0')
#         print("Using CPU for embeddings. This may be slow.")
#         return torch.device('cpu')

#     def load_model(self):
#         """Loads the tokenizer and model onto the selected device."""
#         if self.model is None:
#             print(f"Loading '{self.model_name}' model onto {self.device}...")
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
#             self.model.eval()
#             print("Model loaded successfully.")

#     def unload_model(self):
#         """Removes the model from memory to free up resources."""
#         if self.model is not None:
#             print("\nUnloading embedding model to free memory...")
#             del self.model
#             del self.tokenizer
#             self.model = None
#             self.tokenizer = None
#             if self.device.type == 'cuda':
#                 torch.cuda.empty_cache()
#             gc.collect()
#             print("Model unloaded.")

#     def generate_embeddings(self, texts, batch_size=64):
#         """Generates embeddings for a list of texts. Assumes model is already loaded."""
#         if self.model is None:
#             raise RuntimeError("Embedding model is not loaded. Call `load_model()` first.")

#         all_embeddings = []
#         for i in tqdm(range(0, len(texts), batch_size), desc="      Embedding batch"):
#             batch_texts = texts[i:i + batch_size].tolist()
#             with torch.no_grad():
#                 inputs = self.tokenizer(
#                     batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
#                 ).to(self.device)
#                 outputs = self.model(**inputs)
#                 cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#                 all_embeddings.append(cls_embeddings)
        
#         # Clear memory
#         del inputs, outputs, cls_embeddings
#         if self.device.type == 'cuda':
#             torch.cuda.empty_cache()
            
#         # Handle case where no embeddings were generated
#         if not all_embeddings:
#             return np.array([]).reshape(0, 768) # Return empty 2D array with correct second dimension
            
#         return np.vstack(all_embeddings)


# class PersonalityPredictorCPU:
#     """Loads pre-trained BiLSTM models and predicts on embeddings using the CPU."""
#     def __init__(self, checkpoint_path, target_cols):
#         self.device = torch.device('cpu')
#         self.checkpoint_path = checkpoint_path
#         self.target_cols = target_cols
#         self.models = {}
#         print(f"Predictor initialized on device: {self.device}")

#     def load_all_models(self):
#         """Pre-loads all personality models into memory."""
#         print("Loading all personality prediction models...")
#         for target_col in tqdm(self.target_cols, desc="Loading predictors"):
#             self._load_single_model(target_col)
#         print("All prediction models loaded.")

#     def _load_single_model(self, target_col):
#         """Loads a specific model from the checkpoint path."""
#         if target_col not in self.models:
#             model_path = os.path.join(self.checkpoint_path, f"BiLSTMClassifier_{target_col}.json")
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"FATAL: Model checkpoint not found at {model_path}")

#             model = BiLSTMClassifier(input_dim=768, hidden_dim=256, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.0001)
#             model.load_state_dict(torch.load(model_path, map_location=self.device))
#             model.to(self.device)
#             model.eval()
#             self.models[target_col] = model

#     def predict_single_target(self, X, target_col, batch_size=1024):
#         """Predicts probabilities for a single personality trait."""
#         if X.shape[0] == 0:
#             return np.array([]) # Return empty array if input is empty

#         model = self.models.get(target_col)
#         if model is None:
#             raise RuntimeError(f"Prediction model for {target_col} is not loaded.")

#         all_probas = []
#         with torch.no_grad():
#             for i in tqdm(range(0, len(X), batch_size), desc=f"      Predicting {target_col}", leave=False):
#                 batch = torch.tensor(X[i:i+batch_size]).float().to(self.device)
#                 logits = model(batch)
#                 probas = torch.sigmoid(logits).cpu().numpy()
#                 all_probas.extend(probas.flatten()) # Use extend and flatten
#         return np.array(all_probas)


# def main():
#     """Main function to orchestrate the prediction process using chunking."""
#     print("--- Starting Personality Prediction Script (V3 - Robust Chunking) ---")

#     if not os.path.exists(INPUT_FILE) or not os.path.exists(CHECKPOINT):
#         print("FATAL ERROR: Input file or checkpoint directory not found. Please verify paths.")
#         return

#     # --- Initialize components ---
#     embedder = MemorySafeEmbedder()
#     predictor = PersonalityPredictorCPU(CHECKPOINT, TARGET_COLUMNS)

#     try:
#         # --- Load all models into memory ONCE ---
#         embedder.load_model()
#         predictor.load_all_models()
#         print_memory_stats()
        
#         is_first_chunk = True
#         total_rows_processed = 0

#         # --- Create a file reader iterator that processes the CSV in chunks ---
#         # Using 'warn' will print a warning for bad lines but not stop the process.
#         # This helps diagnose data quality issues without halting the script.
#         df_iterator = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, on_bad_lines='warn', engine='python')

#         for i, df_chunk in enumerate(df_iterator):
#             print(f"\n--- Processing Chunk {i+1} ---")
            
#             # 1. Clean data in the chunk
#             original_count = len(df_chunk)
#             df_chunk.dropna(subset=['text'], inplace=True)
#             df_chunk['text'] = df_chunk['text'].astype(str)
            
#             if df_chunk.empty:
#                 print("Chunk is empty after cleaning, skipping.")
#                 continue
            
#             print(f"Processing {len(df_chunk)} of {original_count} rows in this chunk.")

#             # 2. Generate Embeddings for the current chunk
#             X_embeddings = embedder.generate_embeddings(df_chunk['text'])
            
#             # 3. Make Predictions for all targets
#             for target_col in TARGET_COLUMNS:
#                 probas = predictor.predict_single_target(X_embeddings, target_col)
#                 df_chunk[target_col] = probas
                
#             # 4. Save the processed chunk to the output file
#             if is_first_chunk:
#                 # For the first chunk, create the file and write the header
#                 df_chunk.to_csv(OUTPUT_FILE, index=False, mode='w')
#                 is_first_chunk = False
#             else:
#                 # For all subsequent chunks, append to the file without the header
#                 df_chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

#             total_rows_processed += len(df_chunk)
#             print(f"Chunk {i+1} finished. Total rows processed so far: {total_rows_processed}")
#             print_memory_stats()
#             gc.collect()

#     except Exception as e:
#         print(f"\n--- AN ERROR OCCURRED ---")
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         print("The script has been halted. Partially processed data may be available in the output file.")
    
#     finally:
#         # --- Unload the embedding model to free up VRAM/RAM ---
#         embedder.unload_model()

#     print("\n--- Script Finished ---")
#     print(f"Final output with {total_rows_processed} rows saved to {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc
import psutil
import warnings
import sys

# --- Conditional GPU Monitoring ---
# Attempt to import and initialize pynvml, but don't fail if it's not present.
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False
    print("Warning: pynvml library not found or failed to initialize. GPU monitoring will be disabled.")


# --- Ensure custom modules can be found ---
# This path is from your sample code. Please ensure it is correct in your environment.
sys.path.insert(0, '/data/jmharja/projects/PersonaClassifier/')
try:
    from utils.Models import BiLSTMClassifier
except ImportError:
    print("FATAL ERROR: Could not import BiLSTMClassifier.")
    print("Please ensure the path '/data/jmharja/projects/PersonaClassifier/' is correct and the module exists.")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
BASE_PATH = "/data2/julina/scripts/tweets/2020/03/"
ANALYSIS_DIR = os.path.join(BASE_PATH, "SU_and_NON_SU_analysis/")
INPUT_FILE = os.path.join(ANALYSIS_DIR, "all_users_classified_combined.csv")
OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "all_users_classified_with_personality_V3.csv") # Changed to V3

# Model checkpoint path from your sample code
CHECKPOINT = '/data/jmharja/projects/PersonaClassifier/checkpoint/v4_bilstm_roberta-2025-01-22_11-14-52_final_eval_the_best_S1/models/'

# Personality traits to predict
TARGET_COLUMNS = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']

# Define the chunk size for processing the large CSV
CHUNK_SIZE = 100000  # Process 100,000 rows at a time. Adjust based on your RAM.

def print_memory_stats():
    """Prints detailed CPU and (if available) GPU memory statistics."""
    process = psutil.Process(os.getpid())
    print(f"\nCPU Memory - Used: {process.memory_info().rss / 1024 ** 2:.2f} MB | Available: {psutil.virtual_memory().available / 1024 ** 2:.2f} MB")

    if GPU_AVAILABLE:
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU Memory - Used: {info.used / 1024 ** 2:.2f} MB | Free: {info.free / 1024 ** 2:.2f} MB | Total: {info.total / 1024 ** 2:.2f} MB")
        except Exception as e:
            print(f"Could not retrieve GPU memory info: {e}")


class MemorySafeEmbedder:
    """Generates embeddings with careful memory management."""
    def __init__(self, model_name='roberta-base'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_safe_device()
        print(f"Embedder initialized on device: {self.device}")

    def _get_safe_device(self):
        """Determines the safest available device (GPU with enough memory or CPU)."""
        if torch.cuda.is_available() and GPU_AVAILABLE:
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            # Check for at least 2GB of free VRAM
            if info.free > 2 * 1024 ** 3:
                print("Sufficient GPU VRAM detected. Using CUDA.")
                return torch.device('cuda:0')
        print("Using CPU for embeddings. This may be slow.")
        return torch.device('cpu')

    def load_model(self):
        """Loads the tokenizer and model onto the selected device."""
        if self.model is None:
            print(f"Loading '{self.model_name}' model onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("Model loaded successfully.")

    def unload_model(self):
        """Removes the model from memory to free up resources."""
        if self.model is not None:
            print("\nUnloading embedding model to free memory...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            print("Model unloaded.")

    def generate_embeddings(self, texts, batch_size=64):
        """Generates embeddings for a list of texts. Assumes model is already loaded."""
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded. Call `load_model()` first.")

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="      Embedding batch"):
            batch_texts = texts[i:i + batch_size].tolist()
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
                ).to(self.device)
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
        
        # Clear memory
        del inputs, outputs, cls_embeddings
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Handle case where no embeddings were generated
        if not all_embeddings:
            return np.array([]).reshape(0, 768) # Return empty 2D array with correct second dimension
            
        return np.vstack(all_embeddings)


class PersonalityPredictorCPU:
    """Loads pre-trained BiLSTM models and predicts on embeddings using the CPU."""
    def __init__(self, checkpoint_path, target_cols):
        self.device = torch.device('cpu')
        self.checkpoint_path = checkpoint_path
        self.target_cols = target_cols
        self.models = {}
        print(f"Predictor initialized on device: {self.device}")

    def load_all_models(self):
        """Pre-loads all personality models into memory."""
        print("Loading all personality prediction models...")
        for target_col in tqdm(self.target_cols, desc="Loading predictors"):
            self._load_single_model(target_col)
        print("All prediction models loaded.")

    def _load_single_model(self, target_col):
        """Loads a specific model from the checkpoint path."""
        if target_col not in self.models:
            model_path = os.path.join(self.checkpoint_path, f"BiLSTMClassifier_{target_col}.json")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"FATAL: Model checkpoint not found at {model_path}")

            model = BiLSTMClassifier(input_dim=768, hidden_dim=256, output_dim=1, num_layers=2, bidirectional=True, do_attention=True, dropout_rate=0.0001)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[target_col] = model

    def predict_single_target(self, X, target_col, batch_size=1024):
        """Predicts probabilities for a single personality trait."""
        if X.shape[0] == 0:
            return np.array([]) # Return empty array if input is empty

        model = self.models.get(target_col)
        if model is None:
            raise RuntimeError(f"Prediction model for {target_col} is not loaded.")

        all_probas = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X), batch_size), desc=f"      Predicting {target_col}", leave=False):
                batch = torch.tensor(X[i:i+batch_size]).float().to(self.device)
                logits = model(batch)
                probas = torch.sigmoid(logits).cpu().numpy()
                all_probas.extend(probas.flatten()) # Use extend and flatten
        return np.array(all_probas)


def main():
    """Main function to orchestrate the prediction process using chunking."""
    print("--- Starting Personality Prediction Script (V3 - Robust Chunking) ---")

    if not os.path.exists(INPUT_FILE) or not os.path.exists(CHECKPOINT):
        print("FATAL ERROR: Input file or checkpoint directory not found. Please verify paths.")
        return

    # --- Initialize components ---
    embedder = MemorySafeEmbedder()
    predictor = PersonalityPredictorCPU(CHECKPOINT, TARGET_COLUMNS)

    try:
        # --- Load all models into memory ONCE ---
        embedder.load_model()
        predictor.load_all_models()
        print_memory_stats()
        
        is_first_chunk = True
        total_rows_processed = 0

        # --- Create a file reader iterator that processes the CSV in chunks ---
        # Using 'warn' will print a warning for bad lines but not stop the process.
        # This helps diagnose data quality issues without halting the script.
        df_iterator = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, on_bad_lines='warn', engine='python')

        for i, df_chunk in enumerate(df_iterator):
            print(f"\n--- Processing Chunk {i+1} ---")
            
            # 1. Clean data in the chunk
            original_count = len(df_chunk)
            df_chunk.dropna(subset=['text'], inplace=True)
            df_chunk['text'] = df_chunk['text'].astype(str)
            
            if df_chunk.empty:
                print("Chunk is empty after cleaning, skipping.")
                continue
            
            print(f"Processing {len(df_chunk)} of {original_count} rows in this chunk.")

            # 2. Generate Embeddings for the current chunk
            X_embeddings = embedder.generate_embeddings(df_chunk['text'])
            
            # 3. Make Predictions for all targets
            for target_col in TARGET_COLUMNS:
                probas = predictor.predict_single_target(X_embeddings, target_col)
                df_chunk[target_col] = probas
                
            # 4. Save the processed chunk to the output file
            if is_first_chunk:
                # For the first chunk, create the file and write the header
                df_chunk.to_csv(OUTPUT_FILE, index=False, mode='w')
                is_first_chunk = False
            else:
                # For all subsequent chunks, append to the file without the header
                df_chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

            total_rows_processed += len(df_chunk)
            print(f"Chunk {i+1} finished. Total rows processed so far: {total_rows_processed}")
            print_memory_stats()
            gc.collect()

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("The script has been halted. Partially processed data may be available in the output file.")
    
    finally:
        # --- Unload the embedding model to free up VRAM/RAM ---
        embedder.unload_model()

    print("\n--- Script Finished ---")
    print(f"Final output with {total_rows_processed} rows saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
