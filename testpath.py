import os

dataset_path = r"D:\dataset"  # Change if needed
print("Files in Healthy:", os.listdir(os.path.join(dataset_path, "Healthy"))[:5])
print("Files in Damaged:", os.listdir(os.path.join(dataset_path, "Damaged"))[:5])