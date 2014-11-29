function treeset = q4_rf_train(X, Y, C, k, F)
% Train a random forest using training data (X, Y)

% INPUT
%  X        : [m x n] matrix, where each row is an n-dimensional input example
%  Y        : [m x 1] vector, where the i-th element is the label for the i-th example
%  C        : [1 x 1] scalar, value in (0,1), parameter used for stopping condition (see the homework for further details)
%  k        : [1 x 1] scalar, number of trees to be learned
%  F        : [1 x 1] scalar, size of the random subset of features, used at every split

% OUTPUT
%  treeset : [1 x k] cell array, each object is a tree formatted as
%                    described in q4_dt_train

rand('twister', 0);

[N, d] = size(X); 
assert(length(Y) == N);
feat_idx = 1:d;
c= N*C;
treeset = cell(k,1);

for i=1:k
    tree = zeros(round(3*1/C), 3); % estimated size
    [tree, slot] = q4_rf_train_recursive(X, Y, feat_idx, tree, 1, c, F);
    tree = tree(1:(slot-1), :);
    treeset{i} = tree;
end;

end
