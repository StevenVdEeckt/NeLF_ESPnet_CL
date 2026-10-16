import torch
import numpy as np
import logging
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Average checkpoints...')
    parser.add_argument('dir', type=str, help='Path to the model directory')
    parser.add_argument('measure', type=str, help="Measure based on which best epochs were selected..")
   
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

    epochs = []
    # iterate over files *epoch.pth in dir
    for epoch in os.listdir(args.dir):
        if 'epoch.pth' in epoch:
           f = os.path.join(args.dir, epoch)
           epochs.append(f)

    N = len(epochs)
    assert N > 1, 'Found only %d epochs, expected at least 2 to average over' % N 

    logging.info("Averaging over %d checkpoints: %s" % (N, str(epochs)))

    model = None
    for epoch in epochs:
        logging.info("... loading epoch: %s" % str(epoch))
        if model is None:
            model = torch.load(epoch, map_location='cpu')
        else:
            model = {n: model[n] + p for n, p in torch.load(epoch, map_location='cpu').items()}

    model = {n: p / N for n, p in model.items()}

    save_name = 'valid.%s.ave_%dbest.pth' % (args.measure, N)

    logging.info("Saving the model at: %s" % save_name)
    torch.save(model, "%s/%s" % (args.dir, save_name))

    save_name_sl = 'valid.%s.ave.pth' % args.measure

    logging.info("Creating symlink to: %s" % save_name_sl)
    os.symlink("%s" % save_name, "%s/%s" % (args.dir, 'temp_symlink'))
    os.rename("%s/%s" % (args.dir, 'temp_symlink'), "%s/%s" % (args.dir, save_name_sl))

    logging.info("Done!")


# RUN MAIN FILE
if __name__ == "__main__":
    main()

