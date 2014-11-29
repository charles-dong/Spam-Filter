function info_gain = q4_info_gain(X, Y, idx)
% Compute the information gain on (X, Y) using feature idx as a split

% INPUT
%  X      : [m x n] matrix, where each row is an n-dimensional input example
%  Y      : [m x 1] vector, where the i-th element is the label for the i-th example
%  idx    : [1 x 1] scalar, feature index to be used as the split test

% OUTPUT
%  info_gain: [1 x 1] scalar, information gain on (X, Y) using the split
%                     test of feature idx.
%                     *** NOTE *** omit the first term of the
%                     information gain because it is a constant for a
%                     given set (X,Y)

m = size(X, 1);
samplesof1s = [];%indices of samples where x_idx = 1

%get number of 1 and 0 labels
numberof1s = 0;
for i = 1 : m
    if ( X(i, idx) == 1 )
        numberof1s = numberof1s + 1;
        samplesof1s = [samplesof1s i];
    end
end
numberof0s = m - numberof1s;

toIncludeInS = ismember(1:m, samplesof1s);
toIncludeNotS = ~ismember(1:m, samplesof1s);

subsetS = Y(toIncludeInS, :);
subsetNotS = Y(toIncludeNotS);

info_gain = 0 - (numberof1s/m).*q4_entropy(subsetS) - (numberof0s/m).*q4_entropy(subsetNotS);




end
