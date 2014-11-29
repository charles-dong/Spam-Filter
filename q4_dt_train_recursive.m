function [tree, next_slot] = q4_dt_train_recursive(X, Y, feat_idx, curr_tree, slot, c)
% Learns recursively a decision tree using the subset (X,Y) of training examples falling in the current node.
% This function is used by q4_dt_train

% INPUT
%  X        : [m x n] matrix, where each row is an n-dimensional input example
%  Y        : [m x 1] vector, where the i-th element is the label for the i-th example
%  feat_idx : [1 x L] vector, indices of features to be considered
%  curr_tree: [max_num_slots x 3] matrix, current training tree, which can be expanded
%  slot     : [1 x 1] scalar, current filling slot in the tree
%  c        : [1 x 1] scalar, stop learning if the number of examples in the current node is smaller
%                     than or equal to c

% OUTPUT
%  tree     : [max_num_slots x 3] the learned/updated tree
%  next_slot: [1 x 1] the next slot, if continue to learn

tree = curr_tree;
m = size(X, 1);
n = size(X, 2);
L = size(feat_idx, 2);

%IF Terminate_Condition_is_TRUE ==> change to your stopping condition
if ( c >= m )
    % INSERT YOUR CODE HERE
    % you should call q4_leaf_info here
    % set the values for tree(slot,1:3)
    [label, posterior] = q4_leaf_info(Y);
    tree(slot, 1) = 0; %this is a leaf node
    tree(slot, 2) = label;
    tree(slot, 3) = posterior;
    
    next_slot = slot + 1;
else
    % INSERT YOUR CODE HERE
    % find feature index that makes the best split and assign it to feat_selected
    feat_selected = q4_split(X, Y, feat_idx);
    
    
    if ( feat_selected==0 )%no feasible splits
        % INSERT YOUR CODE HERE
        % you should call q4_leaf_info here
        % set the values for tree(slot,1:3)
        [label, posterior] = q4_leaf_info(Y);
        tree(slot, 1) = 0; %this is a leaf node
        tree(slot, 2) = label;
        tree(slot, 3) = posterior;

        next_slot = slot + 1;   
    else
        
        % Create a new split
        tree(slot, 1) = feat_selected;
        feat_idx = feat_idx(feat_idx ~= feat_selected);
        tree(slot, 2) = slot+1;

        % use the feat_selected to split the data
        left_examples = find(X(:, feat_selected) == 1);
        right_examples = find(X(:, feat_selected) == 0);

        % recursively train left sub-tree
        [tree, next_slot] = q4_dt_train_recursive(X(left_examples,:), Y(left_examples), ...
            feat_idx, tree, slot+1, c);
        tree(slot, 3) = next_slot;

        % recursively train right sub-tree
        [tree, next_slot] = q4_dt_train_recursive(X(right_examples, :), Y(right_examples), ...
            feat_idx, tree, next_slot, c);
    end
end

end

