function feat_selected = q4_split(X, Y, feat_idx)
% Find the split that maximizes the information gain for the subset 
% (X, Y) of the training set

% INPUT
%  X      : [m x n] matrix, where each row is an n-dimensional input example
%  Y      : [m x 1] vector, where the i-th element is the label for the i-th example
%  feat_idx : [1 x L] vector, indices of features to be considered

% OUTPUT
%  feat_selected : [1 x 1] scalar, the feature that maximizes the information gain for (X, Y) 
%                  (this should be one of the numbers stored in feat_idx and an integer between 1 and n)



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
else
    for l = 1 : sizebad_feat
        feat_idx = feat_idx(feat_idx ~= bad_feat(l));
    end
end



if ( size(feat_idx, 1) == 0 )
   feat_selected = 0; 
else
    L = size(feat_idx, 2);
    infoGains = [];
    
    %creates vector of info_gains corresponding to each feat_idx
    for i = 1 : L
        infoGains = [infoGains q4_info_gain(X, Y, feat_idx(i))];
    end

    [SORTED, INDICES] = sort(infoGains, 'descend');

    %INDICES(1) is the index of the largest element in infoGains
    feat_selected = feat_idx(INDICES(1));
end


end
