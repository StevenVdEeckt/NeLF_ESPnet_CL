import argparse
from typing import List

def sort_and_remove_duplicates(lines: List[str], remove_duplicates: bool) -> List[str]:
    if remove_duplicates:
        # Remove duplicates by converting the list to a set and back to a list
        # Note: This also sorts the lines alphabetically based on the first column
        return sorted(list(set(lines)), key=lambda x: x.split()[0])
    else:
        # Sort without removing duplicates
        return sorted(lines, key=lambda x: x.split()[0])

def sort_file(file_path: str, remove_duplicates: bool):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        sorted_lines = sort_and_remove_duplicates(lines, remove_duplicates)
        
        with open(file_path, 'w') as file:
            file.writelines(sorted_lines)
        
        print(f"Processed {file_path}")
    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping...")

def sort_files_in_dir(data_dir: str, files_to_sort: List[str], remove_duplicates: bool):
    for filename in files_to_sort:
        file_path = f"{data_dir}/{filename}"
        sort_file(file_path, remove_duplicates)

def main():
    parser = argparse.ArgumentParser(description="Sort specified files in a directory alphabetically and optionally remove duplicates.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the files to be sorted.")
    parser.add_argument("--remove_duplicates", type=int, choices=[0, 1], default=1, help="Remove duplicates if 1, keep if 0. Default is 1.")
    
    args = parser.parse_args()
    files_to_sort = ['wav.scp', 'text', 'text.ctc', 'text.prev', 'spk2utt', 'utt2spk', 'segments']
    
    remove_duplicates = args.remove_duplicates == 1
    
    sort_files_in_dir(args.data_dir, files_to_sort, remove_duplicates)

if __name__ == "__main__":
    main()

