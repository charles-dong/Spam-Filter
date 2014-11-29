function [label, posterior] = q4_rf_predict(treeset, X)
% Predicts the labels and computes the class posterior probabilities of the test
% examples in X

% INPUT
% treeset   : [k x 1] cell array, each object is a tree formatted as
%              described in q4_dt_train 
%  X        : [m x n] matrix, where each row is an n-dimensional input example

% OUTPUT
%  label    : [m x 1] vector, the predicted labels of the test examples *based on the class majority scheme*
%  posterior: [m x 1] vector, the *average* class posterior probabilities of the test examples

k = size(treeset,1);
m = size(X, 1);
labels = zeros(m, k); %holds return values of each tree
posteriors = zeros(m, k); %holds return values of each tree
label = zeros(m, 1);
posterior = zeros(m, 1);

%display(posterior);
% get all of the labels and posteriors into m x k matrices labels and
% posteriors
for i = 1 : k
    currTree = treeset{i};
    [currLabels, currPosteriors] = q4_dt_predict(currTree, X);
    labels(:, i) = currLabels;
    posteriors(:, i) = currPosteriors;
end

%for each sample, sum the rows of labels and compare for the majority label 
    %and average the rows of posteriors for the average class posterior
for j = 1 : m
    %calc majority class label
    numberOfOnes = sum(labels(j, :));
    if(numberOfOnes > (k - numberOfOnes))
        %more 1s than 0s
        label(j) = 1;
    else
        label(j) = 0;
    end
    
    %calc avg posterior
    sumOfPosteriors = sum(posteriors(j, :));
    posterior(j) = sumOfPosteriors./k;
 
end



end
