function pred_Y = q4_nb_predict(X, phi_y0, phi_y1, phi_prior)
% Predicts the labels of examples X, given the trained model

% INPUT
%  X         : [m x n] matrix containing m examples, each corresponding to an n-dimensional row of the matrix
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)

% OUTPUT
%  pred_Y    : [m x 1] vector, predicted labels for the m examples in X

% HINTS
%  1. for an example pred_y = argmax_k p(y=k) \prod_{j=1}^n p(x_j|y=k) 
%  2. use the log function to avoid numerical problems:
%       pred_y = argmax_k { \log{p(y=k)} + \sum_{j=1}^n \log{p(x_j|y=k)} }

m = size(X, 1);
n = size(X, 2);
pred_Y = zeros(m, 1);

for i = 1 : m
    %reset label values for this particular sample i
    label1 = log(phi_prior);
    label0 = log(1 - phi_prior);
    
    %calculate label probabilities (relative to each other)
    for j = 1 : n
        
        if ( X(i, j) == 1)
            label1 = label1 + log(phi_y1(j));
            label0 = label0 + log(phi_y0(j));
        
        else %X(i, j) == 0
            label1 = label1 + log(1 - phi_y1(j));
            label0 = label0 + log(1 - phi_y0(j));
            
        end
    end

    %assign appropriate label
    if (label1 > label0)
        pred_Y(i) = 1;
    else
        pred_Y(i) = 0;
    end

    
end


end