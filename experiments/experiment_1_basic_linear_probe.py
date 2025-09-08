'''
In this experiment we test:
Can we train a linear probe to predict the correct target class over multiple classes?
Then, can we train a simple attention probe to beat it (random initialisation)?

i.e. use logistic regression on the final token, varying layers to see if we can get out the correct choice (expect this to be a subset of the times that the model knows the correct choice)

similar to what was done in 
https://arxiv.org/pdf/2502.16681
'''

f'''
Plan:
Simple probe:
* use boilerplate linear probe code over -1th token, varying layers. Different model for each dataset, different n_classes.
* Vary hyperparams and optimise over validation (lr, n_epochs, layer)
* pick set of hyperparams for test

Attention probe:
* rather than pred = f(resid_-1^l), we have pred = f_QKV(resid_[0:]^l) where Q K and V are all learnt (Q applies to final token, K to all, V projects to dim n_classes, is what was previously the linear probe)
* vary hyperparams and optimise over validation (lr, n_epochs, layer, d_head, n_heads)

Evaluation:
* we are training on the cross entropy loss. We are measuring the accuracy on the test set
'''