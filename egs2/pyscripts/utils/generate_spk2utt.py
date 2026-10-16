import argparse
from collections import defaultdict

def generate_spk2utt(data_dir: str):
    utt2spk_path = f"{data_dir}/utt2spk"
    spk2utt_path = f"{data_dir}/spk2utt"

    # Read utt2spk and build spk2utt mapping
    spk2utt = defaultdict(list)
    with open(utt2spk_path, 'r') as file:
        for line in file:
            try: utt_id, spk_id = line.strip().split()
            except: print(utt2spk_path, line.strip().split())
            #utt_id, spk_id = line.strip().split()
            spk2utt[spk_id].append(utt_id)
    
    # Write spk2utt mapping to file
    with open(spk2utt_path, 'w') as file:
        for spk_id, utt_ids in spk2utt.items():
            line = f"{spk_id} {' '.join(utt_ids)}\n"
            file.write(line)

def main():
    parser = argparse.ArgumentParser(description="Generate spk2utt from utt2spk.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the utt2spk file.")
    
    args = parser.parse_args()
    generate_spk2utt(args.data_dir)

if __name__ == "__main__":
    main()

