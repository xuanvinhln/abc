from Corpus import Corpus
import LdaEstimator
import sys

def main():
    if (len(sys.argv) != 6):
       print "usage: python main.py <init_alpha> <modeldir_name> <num_topic> <data_file> <random/load>"
       sys.exit(1)

    init_alpha = float(sys.argv[1])
    directory = sys.argv[2]
    num_topics = int(sys.argv[3])
    data_file = sys.argv[4]
    start_type = sys.argv[5]

    # read_data
    corpus = Corpus()
    corpus.read_data(data_file)

    # Run LDA
    LdaEstimator.run_EM(init_alpha, directory, num_topics, corpus, start_type)

if (__name__ == "__main__"):
    main()