import argparse
import os
import re
from deepmultilingualpunctuation import PunctuationModel

def print_tokens_and_positions(sentence):
    # Regex to identify words and special tokens
    token_pattern = re.compile(r"(<[^>]+>|\w+)")
    tokens = token_pattern.findall(sentence)
    # Tuple of tuples for all tokens with their positions
    tokens_with_positions = tuple((token, idx) for idx, token in enumerate(tokens))
    print(tokens_with_positions)
    # Filtering and printing only the special tokens with their positions
    special_tokens_with_positions = tuple((token, idx) for idx, token in enumerate(tokens) if token.startswith('<') and token.endswith('>'))
    print(special_tokens_with_positions)


def remove_special_tokens(text):
    # Splitting text into words and tokens
    #words_and_tokens = re.split('(\s+|(<[^>]+>)+)', text) 
    words_and_tokens = re.split('(\s+|<[^>]+>(?:<[^>]+>)*)', text)
    words_and_tokens = [item for item in words_and_tokens if item is not None and item.strip() != '']
    # Extracting tokens with their positions in terms of word index
    tokens_with_positions = [(i, w) for i, w in enumerate(words_and_tokens) if w.startswith('<') and w.endswith('>')]
    # Reconstructing text without tokens
    text_without_tokens = ' '.join([w for w in words_and_tokens if not (w.startswith('<') and w.endswith('>'))]).strip()
    return text_without_tokens, tokens_with_positions

def insert_special_tokens(processed_text, tokens_with_positions):
    # Splitting processed text into words
    words = processed_text.split()
    # Reinserting tokens based on their word positions
    for pos, token in tokens_with_positions:
        if pos < len(words):
            words.insert(pos, token)
        else:
            words.append(token)
    return ' '.join(words)

# Function to process text: add punctuation and capitalize sentences
def process_text(text, model):
    text_without_tokens, tokens_with_positions = remove_special_tokens(text)
    # Replace specific standalone words
    replacements = {' ggg ': ' <nospeech> ', ' X ': ' <unk> ', ' xxx ': ' <unk> '}
    for word, replacement in replacements.items():
        text_without_tokens = re.sub(r'\b{}\b'.format(re.escape(word.strip())), replacement.strip(), text_without_tokens)
    if text_without_tokens.strip() and model is not None:
       punctuated_text = model.restore_punctuation(text_without_tokens.strip())
       sentences = punctuated_text.split('. ')
       capitalized_sentences = [sentence.capitalize() for sentence in sentences]
       processed_text = '. '.join(capitalized_sentences).strip()
    else:
        processed_text = text_without_tokens
    # Insert tokens back into their original positions
    processed_text_with_tokens = insert_special_tokens(processed_text, tokens_with_positions)
    return processed_text_with_tokens

def process_file(input_file_path, model):
    backup_file_path = f"{input_file_path}.no_punct"
    # Read the input file, process the lines, and prepare the output
    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(backup_file_path, "w", encoding="utf-8") as backupfile:
        lines = infile.readlines()
        processed_lines = []

        for i, line in enumerate(lines, 0):
            backupfile.write(line)  # Write the original line to the backup file
            utterance_id, text = line.split(' ', 1)
            processed_text = process_text(text, model)
            processed_lines.append(f"{utterance_id} {processed_text}\n")
            print("(%d / %d) 1. %s %s" % (i, len(lines), utterance_id, text))
            print("(%d / %d) 2. %s %s" % (i, len(lines), utterance_id, processed_text))
            print("...")
    
    # Write the processed lines to the output file
    with open(input_file_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(processed_lines)

def main(directory, add_punctuation):
    # Initialize the punctuation model
    if add_punctuation:
        model = PunctuationModel(model="oliverguhr/fullstop-dutch-sonar-punctuation-prediction")
    else:
        model = None
    # Files to process
    files_to_process = ["text", "text.ctc", "text.prev"]
    
    for file_name in files_to_process:
        input_file_path = os.path.join(directory, file_name)
        if os.path.exists(input_file_path):
            process_file(input_file_path, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files to add punctuation, capitalize sentences, and handle special tokens.")
    parser.add_argument("directory", type=str, help="Directory containing the text files to process.")
    parser.add_argument("--add_punctuation", type=int, default=1, help="To add punctuation or not, default=True. Set 0 for False, 1 for True.")
    args = parser.parse_args()
    main(args.directory, args.add_punctuation == 1)

