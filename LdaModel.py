import numpy as np
import numpy.random as rnd
from scipy import log

class LdaSufficientStats(object):
    def __init__(self, n_topics, size_vocab):
        self.alpha_ss = 0.0
        self.class_words = np.zeros((n_topics, size_vocab))
        self.class_total = np.zeros(n_topics)
        self.num_docs = 0

    # def corpus_initialize_ss(self, corpus):
    #     for k in xrange(self.num_topics):
    #         for i in xrange(self.NUM_INIT):
    #             d = round(rnd.random() * corpus.n_docs)
    #             print "Initialized with document {0}" % d
    #             doc = corpus.docs[d]
    #             for n in xrange(doc.length):
    #                 self.sufficient_statistics.class_words[k, doc.words[n]] += doc.word_counts[n]
    #     self.sufficient_statistics.class_words += 1
    #     self.sufficient_statistics.class_total = self.sufficient_statistics.class_words.sum(axis=1)
    
    def random_initialize_ss(self, n_topics, size_vocab):
        self.class_words = np.zeros((n_topics, size_vocab))
        self.class_total = np.zeros(n_topics)

        for k in xrange(n_topics):
           for n in xrange(size_vocab):
               self.class_words[k,n] += 1.0/size_vocab + rnd.random()
        self.class_total = self.class_words.sum(axis=1)

    def zero_initialize_ss(self, n_topics, size_vocab):
        self.alpha_ss = 0.0
        self.class_words = np.zeros((n_topics, size_vocab))
        self.class_total = np.zeros(n_topics)
        self.num_docs = 0

class LdaModel(object):
    def __init__ (self, n_topics=0, size_vocab=0, corpus=None):
        self.alpha = 1.0
        # self.NUM_INIT = 1
        self.log_prob_w = np.zeros((n_topics, size_vocab))
        self.num_topics = n_topics
        self.size_vocab = size_vocab

    def maximum_likelihood(self, ss):
        # cap nhat Beta
        for k in xrange(self.num_topics):
            for w in xrange(self.size_vocab):
                if (ss.class_words[k,w] > 0):
                    self.log_prob_w[k,w] = log(ss.class_words[k,w])\
                                           - log(ss.class_total[k])
                else:
                    self.log_prob_w[k,w] = -100
        
    def save(self, filename):
        f_name = filename+".beta"
        f = open(f_name,'w')
        for i in xrange(self.num_topics):
            f.write(" ".join("{0:5.10f}".format(b) for b in self.log_prob_w[i]))
            f.write("\n")
        f.close()

        f_name = filename+".other"
        f = open(f_name,'w')
        f.write("num topics: {0}\n".format(self.num_topics))
        f.write("vocab size: {0}\n".format(self.size_vocab))
        f.write("alpha: {0:5.10f}\n".format(self.alpha))
        f.close()

    def load(self, filename):
        print "\nLoad model"

        f_name = filename+".other"
        print "loading "+f_name
        f = open(f_name,'r')
        n_topics = int(f.readline())
        size_vocab = int(f.readline())
        alpha = float(f.readline())
        f.close()
        print (n_topics, size_vocab, alpha )

        self.alpha = alpha
        self.log_prob_w = np.zeros((n_topics, size_vocab))
        self.num_topics = n_topics
        self.size_vocab = size_vocab

        f_name = filename+".beta"
        print "loading "+f_name
        self.log_prob_w = np.genfromtxt(f_name, delimiter=" ")