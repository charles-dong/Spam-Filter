function word_idx = q4_top_words(phi_y0, phi_y1, phi_prior, k)
% For each class, finds the words that are most indicative of a message
% belonging to that class

% INPUT
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)
%  k         : [1 x 1] scalar, the number of words to output

% OUTPUT
%  word_idx  : [2 x k] matrix, the first row contains the indices of the k most indicative 
%              words for class y=0, the the second row the ones for y=1

word_idx = zeros(2, k);
n = size(phi_y0, 1);

for j = 1 : n
    
    evidence = phi_y1(j).*phi_prior + phi_y0(j).*(1-phi_prior);
    phi_y1(j) = phi_y1(j)./evidence;
    phi_y0(j) = phi_y0(j)./evidence;

end


[SORTED, INDICES1] = sort(phi_y1, 'descend');
[SORTED2, INDICES0] = sort(phi_y0, 'descend');

for i = 1 : k
    
    word_idx(1, i) = INDICES0(i);
    word_idx(2, i) = INDICES1(i);
 
end



end
