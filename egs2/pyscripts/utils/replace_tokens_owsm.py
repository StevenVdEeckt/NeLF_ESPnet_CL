import argparse
import os

# Define the dictionary for token replacement
token_mapping = {
    "<nl>": "<nld>",
    "<unk>": "X",
    # Add more tokens as required
}

def replace_tokens_in_file(file_path, token_mapping):
    """
    Reads a file, replaces tokens based on the provided mapping, and writes the changes back to the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    for old_token, new_token in token_mapping.items():
        content = content.replace(old_token, new_token)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def main(directory):
    """
    Main function that finds and processes files starting with 'text' in the given directory.
    """
    for filename in os.listdir(directory):
        if filename.startswith("text"):
            file_path = os.path.join(directory, filename)
            replace_tokens_in_file(file_path, token_mapping)
            print(f"Processed {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files starting with 'text' and replace specified tokens.")
    parser.add_argument("directory", type=str, help="Directory to search for files")
    
    args = parser.parse_args()
    main(args.directory)

