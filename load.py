import subprocess

# Step 1: Read the input CSV file
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python run_all.py <input_file_path>")
    sys.exit(1)

file_path = sys.argv[1]
df = pd.read_csv(file_path)

output_file_path = '/home/doc-bd-a1/Housing.csv'
df.to_csv(output_file_path, index=False)

subprocess.run(["python", "dpre.py", output_file_path])
subprocess.run(["python", "eda.py", output_file_path])
subprocess.run(["python", "vis.py", output_file_path])
subprocess.run(["python", "model.py", output_file_path])