import numpy as np
import sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train0, y_train1):
    # y_train0 = 2 * y_train0 - 1
    # y_train1 = 2 * y_train1 - 1
    # y_test0 = 2 * y_test0 - 1
    # y_test1 = 2 * y_test1 - 1
    
	#for response 0
    X_mapped = my_map(X_train)
    svm0 = LinearSVC(loss='squared_hinge', C=10, max_iter=10000)
    svm0.fit(X_mapped, y_train0)
    
	#for response 1
    X_mapped = my_map(X_train)
    svm1 = LinearSVC(loss='squared_hinge', C=10, max_iter=10000)
    svm1.fit(X_mapped, y_train1)
    
	# Extract weights and biases
    w0 = svm0.coef_.flatten()
    b0 = svm0.intercept_[0]

    w1 = svm1.coef_.flatten()
    b1 = svm1.intercept_[0]

    return w0, b0, w1, b1


################################
# Non Editable Region Starting #
################################
def my_map(X):
    n_samples = X.shape[0]
    feat = np.zeros((n_samples, 63))
    feat[:, :32] = X
    D = 1 - 2 * X
    for i in range(31):
        product = np.prod(D[:, i:], axis=1)
        feat[:, 32 + i] = product
    return feat
