function [label, posterior] = q4_dt_predict(tree, X)
% Predicts the labels and computes the class posterior probabilities of the test
% examples in X

% INPUT
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
%  X        : [m x n] matrix, where each row is an n-dimensional input example

% OUTPUT
%  label    : [m x 1] vector, the predicted labels of the test examples 
%  posterior: [m x 1] vector, the class posterior probabilities of the test examples

m = size(X, 1);
label = zeros(m, 1);
posterior = zeros(m, 1);
for i = 1 : m %iterate through each example
    nodeIndex = 1;
    
    %while we're not at a leaf..
    while (tree(nodeIndex,1) ~= 0)
        
        
        %update nodeIndex appropriately
            %if feature of X in question is 1...
        if (X(i, tree(nodeIndex, 1)) == 1)
            nodeIndex = tree(nodeIndex, 2);
            %if feature of X in question is 0...
        else
            nodeIndex = tree(nodeIndex, 3);
        end
            
        
        
    end
    
    %we've reached a leaf
    label(i) = tree(nodeIndex, 2);
    posterior(i) = tree(nodeIndex, 3);
    
end





end
