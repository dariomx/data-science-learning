function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

MAX_C = 100;
MAX_SIGMA = 100;
MULSTEP = 10
min_C = 0;
min_sigma = 0;
min_error = flintmax('single');
C = 0.001;
while (C < MAX_C)
    sigma = 0.001;
    while (sigma < MAX_SIGMA)
        kernel = @(x1, x2) gaussianKernel(x1, x2, sigma);
        model= svmTrain(X, y, C, kernel);
        pred = svmPredict(model, Xval);
        error = mean(double(pred ~= yval));
        if (error < min_error)
            min_error = error;
            min_C = C;
            min_sigma = sigma;
        end
        sigma *= MULSTEP;
    end
    C *= MULSTEP;
end

C = min_C;
sigma = min_sigma;

% =========================================================================

end
