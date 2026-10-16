import argparse
import yaml

def read_yaml_and_extract_tokens(input_file_path):
    with open(input_file_path, 'r') as file:
        data = yaml.safe_load(file)
        token_list = data.get('token_list', [])
        print(f"Read {len(token_list)} tokens from {input_file_path}")
        return token_list

def write_tokens_to_file(tokens, output_file_path):
    with open(output_file_path, 'w') as file:
        for token in tokens:
            file.write(f"{token}\n")
    print(f"Written {len(tokens)} tokens to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Process token list from YAML configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to the input YAML configuration file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output token list text file')
    
    args = parser.parse_args()
    
    # Extract token list from the YAML file
    tokens = read_yaml_and_extract_tokens(args.config)
    
    # Write the tokens to the specified output file
    write_tokens_to_file(tokens, args.output)

if __name__ == '__main__':
    main()

