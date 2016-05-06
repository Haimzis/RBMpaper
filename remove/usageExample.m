close all; clc; clear;
%% read data

datasetName = '';% path to dataset file.
% the dataset needs to be a .mat file, with a binary matrix f of size d x n 
% and (optionally, a binary label vector y). 
iDivideData = 0;
data = readData2a(datasetName, iDivideData);

%% setup
rbmInput.restart=1;

% the following are configurable hyperparameters for RBM

rbmInput.reg_type = 'l2';
rbmInput.weightPenalty = 1e-2;%1e-2 % also can try 0, will give deeper net, but better accuracy 
weightPenaltyOrg = rbmInput.weightPenalty;
rbmInput.epsilonw      = 5e-2;%5e-2 % Learning rate for weights 
rbmInput.epsilonvb     = 5e-2;%5e-2 % Learning rate for biases of visible units 
rbmInput.epsilonhb     = 5e-2;%5e-2 % Learning rate for biases of hidden units 
rbmInput.CD=10;   
rbmInput.initialmomentum  = 0;%0;
rbmInput.finalmomentum    = 0.9;%0.9;
rbmInput.maxEpoch = 150; %100
rbmInput.decayLrAfter = 120; % 95
rbmInput.decayMomentumAfter = 90; % when to switch from initial to final momentum
rbmInput.iIncreaseCD = 0;
% monitor free energy and likelihood change (on validation set) with time
rbmInput.iMonitor = 1;


%% train
sizes = [];
rbmInput.data = data;
rbmInput.numhid = size(data.allDataTable,2); % non-configurable for RBM1
stack = cell(1,1);
layerCounter = 1;
addLayers = 1;
while addLayers
    % train RBM
    A = rbmInput.data.allDataTable;
    y = rbmInput.data.labels;
    c1 = corr(A(y==0,:));
    figure
    imagesc(c1)
    h = colorbar;
    caxis([-.2,1])
    %title ('conditional correlation matrix, class = 0')
    set(gca, 'fontsize', 15)
    set(gca,'xtick',0:((length(c1)>5)+1):length(c1));
    set(gca,'ytick',0:((length(c1)>5)+1):length(c1));
    
    rbmInput.weightPenalty = weightPenaltyOrg;
    rbmOutput = rbmV2(rbmInput);
    % collect params
    stack{layerCounter}.vishid = rbmOutput.vishid;
    stack{layerCounter}.hidbiases = rbmOutput.hidbiases;
    stack{layerCounter}.visbiases = rbmOutput.visbiases;

    % SVD to determine number of hidden nodes
    [U,D,V]  = svd (stack{layerCounter}.vishid);
    cumsum(diag(D)) /sum( diag(D))
    numhid = min(find(cumsum(diag(D))/sum(diag(D))>0.95));
    fprintf ('need %1.0f hidden units\n', numhid);
    disp 'paused, press any key to continue'
    %pause;

    % Re-train RBM
    sizes = [sizes, numhid];
    rbmInput.numhid = numhid;
    rbmInput.weightPenalty = 0;%rbmInput.weightPenalty/10;
    rbmOutput = rbmV2(rbmInput);
    % collect params
    stack{layerCounter}.vishid = rbmOutput.vishid;
    stack{layerCounter}.hidbiases = rbmOutput.hidbiases;
    stack{layerCounter}.visbiases = rbmOutput.visbiases;
    figure
    imagesc(stack{layerCounter}.vishid)
    colorbar;
    %title (strcat('weight matrix of RBM ', num2str(layerCounter)));
    xlabel(' hidden units')
    ylabel('visible units')
    set(gca,'ytick',0:((size(stack{layerCounter}.vishid,1)>5)+1):size(stack{layerCounter}.vishid,1));
    set(gca,'xtick',0:((size(stack{layerCounter}.vishid,2)>5)+1):size(stack{layerCounter}.vishid,2));
    set(gca, 'fontsize', 15)
    % setup for next RBM
    rbmInput.data = obtainHiddenRep(rbmInput, rbmOutput);

    % stopping criterion
    if numhid ==1
        addLayers = 0;
    end
    layerCounter = layerCounter + 1;
end

numLayers = size(stack,2);
fprintf ('trained a deep net with %1.0f layers, of sizes:\n', numLayers);
disp(sizes)
%% obtain posterior probabilities
% deterministic
mode = 'deterministic';
posteriorProbsDet = forward (stack, data.allDataTable, mode);


% stochastic
mode = 'stochastic';
nit = 100;
posteriorProbsStoch = forward (stack, data.allDataTable, mode, nit);

%% predict labels
labels = data.labels';

% deterministic mode:
predictedLabels = round(posteriorProbsDet);
% check if predictedLables need to be flipped
m = mean(predictedLabels == data.allDataTable(:,1));
if (m<0.5)
    predictedLabels = 1-predictedLabels;
end
acc = mean(labels==predictedLabels);
inds1 = labels==1;
inds0 = labels==0;
sensitivity = mean(predictedLabels(inds1));
specificity = 1-mean(predictedLabels(inds0));

balAcc_rbmDet = (sensitivity + specificity)/2;
disp 'Deterministic mode:'
fprintf (1,'sensitivity: %0.3f%%\n',100*sensitivity);
fprintf (1,'specificity: %0.3f%%\n',100*specificity);
fprintf (1,'accuracy: %0.3f%%\n',100*acc);
fprintf (1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmDet);

%stochastic mode:
predictedLabels = round(posteriorProbsStoch);
% check if predictedLables need to be flipped
m = mean(predictedLabels == data.allDataTable(:,1));
if (m<0.5)
    predictedLabels = 1-predictedLabels;
end
acc = mean(labels==predictedLabels);
inds1 = labels==1;
inds0 = labels==0;
sensitivity = mean(predictedLabels(inds1));
specificity = 1-mean(predictedLabels(inds0));

balAcc_rbmStoch = (sensitivity + specificity)/2;
disp 'Stochastic mode:'
fprintf (1,'sensitivity: %0.3f%%\n',100*sensitivity);
fprintf (1,'specificity: %0.3f%%\n',100*specificity);
fprintf (1,'accuracy: %0.3f%%\n',100*acc);
fprintf (1,'balanced accuracy: %0.3f%%\n',100*balAcc_rbmStoch);
