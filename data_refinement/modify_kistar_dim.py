import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def process_action_dimension(action_array):
    """
    Removes specified indices from a 35-dimensional action array.
    Indices to remove: 3, 4 (head joints), and 34 (last hand joint).
    """
    # Ensure it's a numpy array for np.delete
    if not isinstance(action_array, np.ndarray):
        action_array = np.array(action_array)

    if action_array.shape == (35,):
        # Indices to delete: 3, 4, 34
        hand = action_array[19:]
        new_hand = np.zeros(5,)
        new_hand[0] = hand[0]
        new_hand[1] = (hand[2] + hand[3]) / 2
        new_hand[2] = (hand[5] + hand[6] + hand[7]) / 3
        new_hand[3] = (hand[9] + hand[10] + hand[11]) / 3
        new_hand[4] = (hand[13] + hand[14] + hand[15]) / 3

        action_array = action_array[:19]
        action_array = np.concatenate([action_array, new_hand], axis=0)
        
        indices_to_delete = [3, 4]
        modified_action = np.delete(action_array, indices_to_delete)

        return modified_action
        
    else:
        # If shape is not (35,), return it as is and let the user know.
        print(f"Warning: Encountered action with unexpected shape {action_array.shape}. Not modifying.", file=sys.stderr)
        return action_array

def process_state_dimension(state_array):
    """
    Reduces hand dimension for a 35-dimensional state array, keeping head joints.
    """
    # Ensure it's a numpy array for processing
    if not isinstance(state_array, np.ndarray):
        state_array = np.array(state_array)

    if state_array.shape == (35,):
        # --- Start of new state processing logic ---

        hand = state_array[19:]

        new_hand = np.zeros(5,)
        new_hand[0] = hand[0]
        new_hand[1] = (hand[2] + hand[3]) / 2
        new_hand[2] = (hand[5] + hand[6] + hand[7]) / 3
        new_hand[3] = (hand[9] + hand[10] + hand[11]) / 3
        new_hand[4] = (hand[13] + hand[14] + hand[15]) / 3

        body = state_array[:19]

        modified_state = np.concatenate([body, new_hand], axis=0)
        
        indices_to_delete = [3, 4]
        final_state = np.delete(modified_state, indices_to_delete)
        
        return final_state
        # --- End of new state processing logic ---
    else:
        # If shape is not (35,), return it as is and let the user know.
        print(f"Warning: Encountered state with unexpected shape {state_array.shape}. Not modifying.", file=sys.stderr)
        return state_array

def main(directory_path):
    """
    Processes all .parquet files in a given directory to reduce action dimension.
    """
    p = Path(directory_path)
    if not p.is_dir():
        print(f"Error: Directory not found at '{directory_path}'", file=sys.stderr)
        sys.exit(1)

    parquet_files = sorted(list(p.glob("*.parquet")))
    
    if not parquet_files:
        print(f"No .parquet files found in '{directory_path}'", file=sys.stderr)
        return
        
    print(f"Found {len(parquet_files)} .parquet files to process in '{directory_path}'.")

    for file_path in tqdm(parquet_files, desc="Modifying dimensions"):
        try:
            df = pd.read_parquet(file_path)
            
            if 'action' in df.columns:
                df['action'] = df['action'].apply(process_action_dimension)
            else:
                print(f"Warning: 'action' column not found in {file_path}. Skipping.", file=sys.stderr)

            if 'observation.state' in df.columns:
                df['observation.state'] = df['observation.state'].apply(process_state_dimension)
            else:
                print(f"Warning: 'observation.state' column not found in {file_path}. Skipping.", file=sys.stderr)

            # Simple verification on the first row
            if not df.empty and 'action' in df.columns and df['action'].iloc[0].shape != (22,):
                 print(f"Warning: New action dimension is not 22 in {file_path}. Shape is {df['action'].iloc[0].shape}. Skipping overwrite.", file=sys.stderr)
                 continue

            if not df.empty and 'observation.state' in df.columns and df['observation.state'].iloc[0].shape != (22,):
                 print(f"Warning: New state dimension is not 22 in {file_path}. Shape is {df['observation.state'].iloc[0].shape}. Skipping overwrite.", file=sys.stderr)
                 continue

            # Overwrite the original file
            df.to_parquet(file_path, index=False)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        main(target_dir)
        print("\nProcessing complete.")
    else:
        print("Usage: python modify_kistar_dim.py <path_to_directory_with_parquet_files>", file=sys.stderr)
        sys.exit(1)
