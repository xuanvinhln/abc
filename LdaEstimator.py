from LdaInference import *
from LdaModel import *
from scipy.special import psi
import numpy as np

def docEStep(doc, gamma, phi, model, ss):

    # posterior inference
    likelihood = run_inference(doc, model, gamma, phi)

    # update sufficient statistics
    # Danh cho cap nhat Beta sau nay
    for n in xrange(doc.length):
        ss.class_words[:,doc.words[n]] += doc.counts[n]*phi[n]
        ss.class_total += doc.counts[n]*phi[n]
    ss.num_docs += 1

    return likelihood

def write_word_assignment(file, doc, phi, model):
    file.write("{:03d}".format(doc.length))
    for n in xrange(doc.length):
        file.write(" {:04d}:{:02d}".format(doc.words[n], np.argmax(phi[n])))
    file.write("\n")

def save_gamma(fileName, gamma, num_docs, num_topics):
    f = open(fileName,'w')
    for d in xrange(num_docs):
        f.write(" ".join("{:5.10f}".format(g) for g in gamma[d]))
        f.write("\n")
    f.close()

def run_EM(init_alpha, directory, num_topics, corpus, startType=None):

    max_iter = 100
    em_converged = 1e-4

    # initialize model

    # if startType=="seeded":
    #     model = LdaModel(num_topics, corpus.size_vocab, "seeded", corpus)
    if startType == "random":
        model = LdaModel(num_topics, corpus.size_vocab)
        ss = LdaSufficientStats(num_topics, corpus.size_vocab)
        ss.random_initialize_ss(num_topics, corpus.size_vocab)
        model.maximum_likelihood(ss)
        model.alpha = init_alpha
    else:
        model = LdaModel() 
        model.load(startType)
        ss = LdaSufficientStats(model.num_topics, model.size_vocab)

    filename = directory+"/000"
    model.save(filename)

    var_gamma = np.zeros((corpus.num_docs, model.num_topics))

    # Run Expectation Maximization
    likelihood_old = 0.0
    converged = 1.0
    filename = directory+"/likelihood.dat"
    likelihood_file = open(filename,'w')

    i = 0
    while (i<=max_iter):
        i += 1
        print "\n **** EM iteration {0} **** \n".format(i)
        likelihood = 0.0
        ss.zero_initialize_ss(model.num_topics, model.size_vocab)

        # E-step
        for d in xrange(corpus.num_docs):
            # moi van ban co 1 matran phi[Nd, K]
            phi = np.zeros((corpus.docs[d].length, model.num_topics))
            if d%100 == 0:
                print "document {0}".format(d)
            likelihood += docEStep(corpus.docs[d]\
                        , var_gamma[d]\
                        , phi\
                        , model\
                        , ss)

        # M-step (Update Beta)
        model.maximum_likelihood(ss)

        # Check converged
        converged = (likelihood_old - likelihood) / likelihood_old
        print "likelihood_old={:10.10f}".format(likelihood_old)
        print "converged={:5.5f}".format(converged)

        if converged<0:
            max_iter = max_iter*2
        likelihood_old = likelihood

        # output model and likelihood
        likelihood_file.write("{:10.10f}\t{:5.5e} \n".format(likelihood, converged))
        if i%5 == 0:
            filename = directory+"/{:03d}".format(i)
            model.save(filename)
            filename = directory+"/{:03d}.gamma".format(i)
            save_gamma(filename, var_gamma, corpus.num_docs, model.num_topics)

    # output the final model
    filename = directory+"/final"
    model.save(filename)
    filename = directory+"/final.gamma"
    save_gamma(filename, var_gamma, corpus.num_docs, model.num_topics)
    likelihood_file.close()

    # output the word assignments (for visualization)
    filename = directory+"/word-assignments.dat"
    w_asgn_file = open(filename, 'w')
    for d in xrange(corpus.num_docs):
        if (d%100 == 0):
            print "final e step document ",d
        likelihood += run_inference(corpus.docs[d], model, var_gamma[d], phi)   
        write_word_assignment(w_asgn_file, corpus.docs[d], phi, model)
        print "document ",d
    w_asgn_file.close()
