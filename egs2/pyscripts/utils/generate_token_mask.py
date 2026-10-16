import argparse
import os
import sentencepiece as spm
import numpy as np
import logging

def main():
    # Setup logging
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process ASR dataset directories and generate a token mask.")
    parser.add_argument('directories', type=str, help='Comma-separated list of directories containing the text files.')
    parser.add_argument('--vocab', type=str, default="data/clm_token_list/bpe_unigram20000/bpe.model", help='Path to the SentencePiece model file.')
    parser.add_argument('--mask_name', type=str, help='Name to store the mask of used tokens.', default=None)
    
    args = parser.parse_args()
    directories = args.directories.split(',')
    model_file = args.vocab
    
    # Generate default mask name if not provided
    if not args.mask_name:
        model_dir, model_name = os.path.split(model_file)
        task = model_dir.split('/')[1].replace('_token_list', '')
        vocab_name = os.path.basename(model_dir)
        args.mask_name = f"mask_{task}_{vocab_name}"
    
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    vocab_size = sp.GetPieceSize()
    total_tokens = 0
    
    # Process each directory
    for directory in directories:
        mask = np.zeros(vocab_size, dtype=bool)
        mask[0:2] = True
        logging.info(f"Processing directory: {directory}")
        text_file_path = os.path.join(directory, "text")
        
        if not os.path.exists(text_file_path):
            logging.warning(f"Text file not found in directory: {directory}")
            continue
        
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            for line_count, line in enumerate(text_file, start=1):
                _, utterance = line.split(' ', 1)
                tokens = sp.EncodeAsIds(utterance.strip())
                mask[tokens] = True
                
        logging.info(f"Processed {line_count} lines in {directory}.")
        
        # Calculate token presence statistics
        num_present_tokens = np.sum(mask)
        total_tokens += num_present_tokens
        percentage = (num_present_tokens / vocab_size) * 100
        logging.info(f"Tokens present in {directory}: {num_present_tokens} ({percentage:.2f}%)")
        
        # Save the mask for the current directory
        mask_file_path = os.path.join(directory, args.mask_name)
        np.save(mask_file_path, mask)
        logging.info(f"Mask saved to {mask_file_path}")

    # total tokens present in all datasets = total_tokens
    # if total_tokens > vocab_size; then there is overlap; else None
    if total_tokens > vocab_size:
        message = "there is overlap"
    else:
        message = "there is *no* overlap"
    percentage = (total_tokens / vocab_size) * 100
    logging.info(f"Total tokens present ({message}): {total_tokens} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()

