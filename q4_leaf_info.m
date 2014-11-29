function [label, posterior] = q4_leaf_info(Y)
% Compute the majority class label and the posterior at the leaf node given
% the labels Y of the training examples at this node

% INPUT
%  Y        : [m x 1] vector, labels of the training examples at this node

% OUTPUT
%  label    : [1 x 1] scalar, majority label at this leaf node
%  posterior: [1 x 1] scalar, posterior probability of Y being 1 at this
%             leaf node. This is given by the fraction of examples having label 1 in Y


m = size(Y, 1);
label1 = 0;

for i = 1 : m
    if ( Y(i) == 1 )
        label1 = label1 + 1;
    end 
end

posterior = label1/m;
if ( label1 >= (m - label1) )
   label = 1;
else
    label = 0;
end


end
