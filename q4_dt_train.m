function tree = q4_dt_train(X, Y, C)
% Train a decision tree using training data (X, Y) and parameter C

% INPUT
%  X        : [m x n] matrix, where each row is an n-dimensional input example
%  Y        : [m x 1] vector, where the i-th element is the label for the i-th example
%  C        : [1 x 1] scalar, value in (0,1), parameter used for one of the
%             stopping condition (see the homework for further details)

% OUTPUT
%  tree : [L x 3] matrix, the learned tree. L is the number of nodes in the
%                 tree. Each row represents one particular node and its three 
%                 values have the following meaning:
%
%                 tree(i,1) is an integer value between 0 and n, 
%                 0 means that this node is a leaf node, otherwise tree(i,1) 
%                 is the feature index used at this node i
%                 
%                 In the case of a leaf node, tree(i,2) will store the class
%                 label for this leaf and tree(i,3) stores its posterior

%                 In the case of a non-leaf node, tree(i,2) and tree(i,3) are
%                 the node ids (row number) of its children. Note that the
%                 branch to follow is determined by the splitting feature:
%                 if the feature defined by tree(i,1) is 1 -> go to tree(i,2)
%                 otherwise -> go to tree(i,3)

[N, d] = size(X); 
assert(length(Y) == N);
feat_idx = 1:d;
c = N*C;
tree = zeros(round(3*1/C), 3); % estimated size 

% run the DT_recursive algorithm
[tree, slot] = q4_dt_train_recursive(X, Y, feat_idx, tree, 1, c);
tree = tree(1:(slot-1), :);

end

