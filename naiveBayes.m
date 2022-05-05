clear;
load ('spambase1.mat');
X=spambase1;
labels = cell(4601,1);
for i=1:1812
    labels(i,1)={'spam'};
end
for i=1813:length(labels)
    labels(i,1)={'non-spam'};
end
Y=labels;
rng('default')

cv =cvpartition(labels,'HoldOut',0.20);
trainInds = training(cv);
testInds = test(cv);
XTrain = X(trainInds,:);
YTrain = Y(trainInds);
XTest = X(testInds,:);
YTest = Y(testInds);

Model=fitcnb(XTrain,YTrain,'ClassNames',{'spam','non-spam'});

idx = randsample(sum(testInds),100);
label = predict(Model,XTest); 


table(YTest(idx),label(idx),'VariableNames',{'Actual Value', 'Predicted Value'})
cm = confusionchart(YTest,label);
cm.ColumnSummary = 'column-normalized';