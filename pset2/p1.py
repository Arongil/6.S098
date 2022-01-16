# cost vector is random, normally distributed with mean as vector c_0 and covariance as vector Sigma.

# a) How to minimize expected cost E(c^T x) subject to Ax <= b?
# To minimize expected cost, we would treat the mean vector c_0 as weights on the coordinates of x, i.e. we would just minimize c_0^T x. Variance doesn't matter if we are minimizing only expected value.

# b) How to minimize risk-sensitive cost, which is a linear combination of expected value and variance?
# To minimize risk-sensitive cost, we would add together many objectives:
#  - c_0^T x for E(c^T x)
#  - something to do with the covariance matrix... that's the key
# Answer: quadratic forms! Min x^T \Sigma x. This quadratic form is convex if \Sigma is positive semi-definite.
#         If we assume the entries of c are independent, then convexity becomes automatic. Otherwise we must check.

# As the rest of this problem is theoretical and requires no code, I have typed it up only in the LaTeX doc.
