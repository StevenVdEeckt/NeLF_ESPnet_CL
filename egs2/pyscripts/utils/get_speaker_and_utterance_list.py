import logging
import argparse


def main():

    parser = argparse.ArgumentParser(description='Remove punctuations...')
    parser.add_argument('data', default=str, help="Path to the data")

    args = parser.parse_args()

    # Prepare log file
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

    speakers = []
    utterances = []

    logging.info("Reading %s/spk2utt..." + args.data)
    with open(args.data + "/spk2utt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line.strip("\n")
            line = line.split(" ")
            speakers.append(line[0])
            utterances += line[1:]
            line = f.readline()

    logging.info("Number of speakes: %d" % len(speakers))
    logging.info("Number of utterances: %d" % len(utterances))

    logging.info("Writing %s/list_of_speakers.txt..." % args.data)
    with open(args.data + "/list_of_speakers.txt", "w", encoding="utf-8") as f:
        for speaker in speakers:
            f.write(speaker + "\n")

    logging.info("Writing %s/list_of_utterances.txt..." % args.data)
    with open(args.data + "/list_of_utterances.txt", "w", encoding="utf-8") as f:
        for utt in utterances:
            f.write(utt + "\n")

    logging.info("Done!")



# RUN MAIN FILE
if __name__ == "__main__":
    main()