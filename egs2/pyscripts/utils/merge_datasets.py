import argparse
import os
import logging
import numpy as np
import shutil


def main():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    parser = argparse.ArgumentParser(description='Merging datasets')
    parser.add_argument('data_dirs', type=str, help='Comma-separated list of data dirs to merge')
    parser.add_argument('new_data_dir', type=str, help='Data dir of the new dataset')

    args = parser.parse_args()

    # set the params
    data_dirs = args.data_dirs.split(",")
    assert len(data_dirs) >= 1, "Expecting more than one data dir, but found %d" % len(data_dirs)
    logging.info("Merging the following datasets: %s" % str(data_dirs))
    new_data_dir = args.new_data_dir
    
    # files to merge
    files_ = ['text','utt2spk', 'wav.scp', 'spk2utt', 'text.ctc', 'text.prev']
    copy_files_ = ['feats_type']
    #files_ = ['text']
    #copy_files_ = []

    files = []
    copy_files = []

    # check if all files are present
    for  data_dir in data_dirs:
        for f in files_:
            if not os.path.isfile("%s/%s" % (data_dir, f)):
                logging.info('File %s/%s does not exist' % (data_dir, f))
            else:    
                files.append(f)
        for f in copy_files_:
            if not os.path.isfile("%s/%s" % (data_dir, f)):
                logging.info('File %s/%s does not exist' % (data_dir, f))
            else:    
                copy_files.append(f)

    # make the new datadir
    if not os.path.isdir(new_data_dir):
        os.mkdir(new_data_dir)

    # for each file, merge all datasets and save at new location
    for f in files:
        logging.info("Merge file.. %s" % f)
        lines = []
        for data_dir in data_dirs:
            logging.info("... reading file: %s/%s" % (data_dir, f))
            with open("%s/%s" % (data_dir, f), "r", encoding='utf-8') as nf:
                line = nf.readline()
                while line:
                    lines.append(line)
                    line = nf.readline()
        logging.info("File %s/%s will have %d lines!" % (new_data_dir, f, len(lines)))
        lines = list(set(lines))
        lines.sort()
        logging.info("Sorted the lines!") 
        logging.info("Writing the lines...")
        with open("%s/%s" % (new_data_dir, f), 'w', encoding='utf-8') as nf:
            for line in lines:
                nf.write(line)
        logging.info("Done with file %s!" % f)

    for f in copy_files:
        logging.info("Copying file.. %s" %  f)
        shutil.copyfile("%s/%s" % (data_dirs[0], f), "%s/%s" % (new_data_dir, f))

    logging.info("Done!")



# RUN MAIN FILE
if __name__ == "__main__":
    main()

