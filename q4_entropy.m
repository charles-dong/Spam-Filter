function entropy = q4_entropy(Y)
% Compute the entropy for the label set Y

% INPUT
%  Y     : [m x 1] vector, the labels of the subset of examples for which entropy must
%          be computed

% OUTPUT
%  entropy : entropy of label set Y

m = size(Y, 1);
label1 = 0;
dontcalc = 0;

%how many are of class 1?
for i = 1 : m
    if( Y(i) == 1 )
       label1 = label1 + 1;
    end
end

%how many of class 0?
label0 = m - label1;

%account for 0 probabilities
if (label0 == 0)
    dontcalc = 1;
end
if (label1 == 0)
    dontcalc = 1;
end

%dont calc or calc entropy, depending on 0 probs
if (dontcalc == 1)
    entropy = 0;
else
    entropy = (-1).*( (label1./m).*log(label1./m) + (label0./m).*log(label0./m) );
end


end
