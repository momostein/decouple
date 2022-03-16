%% Try decoupling on a neural network
% 
% Digit recognition
%
% Adapted from https://nl.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html

%% Load the digit dataset

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Display some of the images
figure(1);
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% Count the amount of samples with each label
labelCount = countEachLabel(imds)

% Size of each image
img_size = size(readimage(imds, 1))

%% Separate Training and Validation Sets

numTrainFiles = 750
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Define network architecture

layers = [
    imageInputLayer([28 28 1],"Name","imageinput")

    convolution2dLayer([3 3],8,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    
    convolution2dLayer([3 3],16,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    
    convolution2dLayer([3 3],32,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    
    fullyConnectedLayer(64,"Name","fc_1")
    reluLayer("Name","relu_4")
    
    % This is the layer we'll try to replace
    fullyConnectedLayer(32,"Name","fc_replace")
    reluLayer("Name","relu_replace")
    
    fullyConnectedLayer(10,"Name","fc_3")
    softmaxLayer("Name","softmax")
    
    classificationLayer("Name","classoutput")];

%% Train the network
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

%% Calculate accuracy

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%% Try to evaluate a single image

layer_inputs = squeeze(activations(net, imdsTrain, "relu_4"));
layer_outputs = squeeze(activations(net,  imdsTrain, "relu_replace"));

