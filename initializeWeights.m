function weights = initializeWeights(sj_in, sj_out)
% INITIALIZEWEIGHTS will randomly initialize weights and return a matrix
% of random real values.
INIT_EPSILON = 0.1;
weights = rand(sj_out, sj_in+1) * (2 * INIT_EPSILON) - INIT_EPSILON;
end