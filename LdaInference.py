import math
import numpy as np
from scipy.special import psi, gammaln
from scipy import log

# Compute log likelihood
def compute_likelihood(doc, model, phi, var_gamma):

    dig = psi(var_gamma)
    var_gamma_sum = var_gamma.sum()
    digsum = psi(var_gamma_sum)

    # Tinh tong 1
    likelihood = gammaln( model.alpha * model.num_topics )\
                - model.num_topics * gammaln( model.alpha )\
                - gammaln( var_gamma_sum )
    # Tinh tong 2
    likelihood +=( (model.alpha-1) * (dig - digsum)\
                + gammaln(var_gamma)\
                - (var_gamma-1) * (dig-digsum) ).sum()
    # Tinh tong 3
    for n in xrange(doc.length):
        likelihood +=( doc.counts[n]\
                    *(phi[n]* ((dig-digsum) - log(phi[n])\
                    + model.log_prob_w[:,doc.words[n]])) ).sum()

    return likelihood

# variational inference
def run_inference(doc, model, var_gamma, phi):
    var_max_iter = 20
    var_converged = 1e-6

    converged = 1.0
    likelihood_old = 0.0

    # compute posterior dirichlet
    # Khoi tao gia tri cho gamma va phi
    # Ma tran phi[Nd, K]
    phi.fill(1.0/model.num_topics)
    # Mang gamma[K]
    var_gamma.fill(model.alpha + doc.total/float(model.num_topics))
    digamma_gam = psi(var_gamma)
            
    # Cap nhat gamma va phi
    var_iter = 0
    while ((converged > var_converged) and (var_iter < var_max_iter or var_max_iter == -1)):
        var_iter += 1
        for n in xrange(doc.length):
            oldphi = phi[n].copy()
            phi[n] = digamma_gam + model.log_prob_w[:,doc.words[n]]
            phisum = log( np.exp(phi[n]) .sum() )

            # Chuan hoa tong bang 1 cho phi va phi=beta[k,wn]*exp()
            phi[n] = np.exp(phi[n] - phisum)
            # cap nhat gamma, tru lan luot di tung cai phi cu, cong them phi moi
            var_gamma += doc.counts[n] * (phi[n] - oldphi)
            digamma_gam = psi(var_gamma)
            
        likelihood = compute_likelihood(doc, model, phi, var_gamma)
        assert (not math.isnan(likelihood))
        converged = (likelihood_old - likelihood)/likelihood_old
        likelihood_old = likelihood

        # print "[LDA INF] iter={:2d}\t{:8.5f}\t{:1.3e}".format(var_iter, likelihood, converged)
    return likelihood
