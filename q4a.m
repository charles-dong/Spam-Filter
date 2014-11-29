function q4a()
% This script requires the following functions to be implemented:
% q2_error
% q4_nb_train
% q4_nb_predict

assert(checking('q4a')==0);

% load the spam dataset
S = load('spamdata.mat');

X = S.trainsetX;
Y = S.trainsetY;
Xt = S.testsetX;
Yt = S.testsetY;

clear S;

% train a Niave Bayes model
[phi_y0, phi_y1, phi_prior] = q4_nb_train(X, Y);

pred_Y = q4_nb_predict(X, phi_y0, phi_y1, phi_prior);
train_error = q2_error(Y, pred_Y);

pred_Y = q4_nb_predict(Xt, phi_y0, phi_y1, phi_prior);
test_error = q2_error(Yt, pred_Y);

fprintf('Training error: %.2f%%\n', 100*train_error);
fprintf('Test error: %.2f%%\n', 100*test_error);



