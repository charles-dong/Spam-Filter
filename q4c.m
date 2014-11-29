function q4c()
% This script requires the following functions to be implemented:
% q2_error
% q4_entropy
% q4_info_gain
% q4_leaf_info
% q4_split
% q4_dt_train_recursive
% q4_dt_predict

assert(checking('q4c')==0);

S = load('spamdata.mat');

X = S.trainsetX;
Y = S.trainsetY;
Xt = S.testsetX;
Yt = S.testsetY;

clear S;

% different values of C
Cs = [0.005 0.01 0.05 0.1 ];

train_error = zeros(1, size(Cs,2));
test_error = zeros(1, size(Cs,2));
for i=1:length(Cs)
    tree = q4_dt_train(X, Y, Cs(i));
    [pred_Y, ~] = q4_dt_predict(tree, X);
    train_error(i) = q2_error(Y, pred_Y);
    [pred_Y, ~] = q4_dt_predict(tree, Xt);
    test_error(i) = q2_error(Yt, pred_Y);
end;

% plot data
plot(log(Cs), train_error);
hold on;
plot(log(Cs), test_error, '--r');
legend('Misclassification Rate on the Training Set', 'Misclassification Rate on the Test Set');
set(gca,'XTick', log(Cs))
set(gca,'XTickLabel',mat2cell(Cs, 1, length(Cs)))
xlabel('C (in log scale)');
ylabel('misclassification rate');
grid on;

saveas(gcf, 'q4c.fig');

