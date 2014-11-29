function [phi_y0, phi_y1, phi_prior] = q4_nb_train(Xtrain, Ytrain)
% Train a Naive Bayes model using Laplacian smoothing

% INPUT
%  Xtrain    : [m x n] matrix, where each row is an n-dimensional input *training* example
%  Ytrain    : [m x 1] vector, where the i-th element is the label for the i-th *training* example

% OUTPUT
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)


m = size(Xtrain, 1);
n = size(Xtrain, 2);

phi_prior = 2; %add 2 examples for y=1 for laplacian smoothing
phi_y0 = zeros(n, 1);
phi_y1 = zeros(n, 1);
for i = 1 : m
    if( Ytrain(i) == 1 )
        phi_prior = phi_prior + 1;
        
        for j = 1 : n
            if ( Xtrain(i, j) == 1 )
                phi_y1(j) = phi_y1(j) + 1;
            end
        end
        
    else
        %Ytrain(i) == 0
        for j = 1 : n
            if ( Xtrain(i, j) == 1 )
                phi_y0(j) = phi_y0(j) + 1;
            end
        end
    end
end

phi_y1 = (1 + phi_y1)./phi_prior;
phi_y0 = (1 + phi_y0)./(m + 4 - phi_prior);

phi_prior = phi_prior/(m+4); %add 2 examples for each of y=1/0 for laplacian smoothing




end
