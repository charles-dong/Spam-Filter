function feat_selected = q4_rf_split(X, Y, feat_idx, F)
% Find the split that maximizes the information gain for the subset 
% (X, Y) of the training set from a random subset of F features

% INPUT

%  X        : [m x n] matrix, where each row is an n-dimensional input example
%  Y        : [m x 1] vector, where the i-th element is the label for the i-th example
%  feat_idx : [1 x L] vector, indices of features to be potentially considered
%  F        : [1 x 1] scalar, size of the random subset of features to be considered

% OUTPUT
%  feat_selected : [1 x 1] scalar, the feature that maximizes the information gain for (X, Y) 
%                  (this should be one of the numbers stored in feat_idx and an integer between 1 and n). 
%                  Note this value must be set to 0 if there is no feasible split


% INSERT YOUR CODE HERE:
% compute the feasible feature indices
% store them in feasible_idx

%filter out degenerate features
L = size(feat_idx, 2);
bad_feat = [];
for k = 1 : L 
        partition1 = find(X(:, feat_idx(k)) == 1);
        partition0 = find(X(:, feat_idx(k)) == 0);
        if (size(partition1, 1) == 0 )
            bad_feat = [bad_feat feat_idx(k)];
        end
        if (size(partition0, 1) == 0 )
            bad_feat = [bad_feat feat_idx(k)];
        end
end
sizebad_feat = size(bad_feat, 2);
if (size(sizebad_feat,1) == 0 )
    feasible_idx = feat_idx;
else
    for l = 1 : sizebad_feat
        feat_idx = feat_idx(feat_idx ~= bad_feat(l));
    end
    feasible_idx = feat_idx;
end


%if no feasible features, return 0
if (length(feasible_idx) == 0)
    feat_selected = 0;
    
%else find best feature
else

% USE THIS AS RANDOM SELECTION OF FEATURE SUBSET
% LEAVE UNCHANGED
% -----------------------------------------------
    indperm = randperm(length(feasible_idx));
    if length(feasible_idx)>F
        feasible_idx = feasible_idx(indperm(1:F));
    end
% -----------------------------------------------

% INSERT YOUR CODE HERE:
% choose within this subset the one with the best gain
    feat_selected = q4_split(X, Y, feasible_idx);

    
    %L = length(feasible_idx);
    %infoGains = [];
    
    %creates vector of info_gains corresponding to each feat_idx
    %for i = 1 : L
    %    infoGains = [infoGains q4_info_gain(X, Y, feasible_idx(i))];
    %end

    %[SORTED, INDICES] = sort(infoGains, 'descend');

    %INDICES(1) is the index of the largest element in infoGains
    %feat_selected = feasible_idx(INDICES(1));




end


end
