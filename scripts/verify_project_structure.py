import os

error_counter: int = 0

# assert that data is in the correct folder
data_files = ['test_data.txt', 'train_pos_full.txt', 'train_pos.txt', 'train_neg_full.txt', 'train_neg.txt']
for file in data_files:
    if not(os.path.exists(os.path.join('data', file))):
        error_counter += 1
        print(f"File not found: data/{file} does not exist.")

if error_counter > 0:
    print(f"Found {error_counter} errors in total.")
    exit(1)