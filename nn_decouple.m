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
	% Image inputs (28x28 monochrome)
    imageInputLayer([28 28 1],"Name","imageinput")
	
	% Convolutional layers
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
    
	% Fully connected layers (classical neural networks)
    fullyConnectedLayer(64,"Name","fc_before_replace")
    reluLayer("Name","relu_before_replace")
    
    % This is the layer we'll try to replace
    fullyConnectedLayer(32,"Name","fc_replace")
% 	tanhLayer("Name","act_replace")
    reluLayer("Name","act_replace")
    
    fullyConnectedLayer(10,"Name","fc_after_replace")
    softmaxLayer("Name","softmax_after_replace")
    
	% Final output layer
    classificationLayer("Name","classoutput")];

%% Train the network

% Define options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(imdsTrain,layers,options);

% Calculate accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%% Calculate input and output activations of the given layers

layer_inputs = activations(net, imdsTrain, "relu_before_replace");
layer_outputs = activations(net,  imdsTrain, "act_replace");

size(layer_inputs)

%% Extract the fully connected layer and its weights and biases

fully_con_layer = net.Layers(15)

weight_mat = fully_con_layer.Weights;
bias_vec = fully_con_layer.Bias;

% Apply relu
test_out_mat = extractdata(relu(dlarray(weight_mat*squeeze(layer_inputs) + bias_vec)));

% Add extra dimensions
clear test_out
test_out(1, 1, :, :) = test_out_mat;

size(test_out)

% Calculate error between the original activations and the manually
% calculated activations
norm(squeeze(layer_outputs - test_out))

%% Assemble the last output layers into a new pre-trained subnetwork

% Make a new layer array with a new input layer
% (has to be shaped as a 1x1x32 imageInputLayer)
last_layers = cat(1, [imageInputLayer([1 1 32],"Name","imageinput","Normalization","none")] ,net.Layers(17:19))

% Assemble them into a new network without re-training them
last_layers_net = assembleNetwork(last_layers);

%% Test prediction of the new subnetwork

% Caculate the outputs from the original and manually calculated
% activations
YPred_2_orig = classify(last_layers_net,activations(net,  imdsTrain, "act_replace"));
YPred_2_extracted = classify(last_layers_net,test_out);

YValidation_2 = imdsTrain.Labels;

% Calculate accuracy of both predictions
accuracy_2_orig = sum(YPred_2_orig == YValidation_2)/numel(YValidation_2)
accuracy_2_extracted = sum(YPred_2_extracted == YValidation_2)/numel(YValidation_2)


%% Calculate the jacobian matrix of the target layer in each point

% Calculate the outputs of the fully connected layer
layer_out = weight_mat*squeeze(layer_inputs) + bias_vec;

% Calculate the derivatives of the activation layer
activation_derivative = fully_connected_layer_out > 0; % ReLU function
% activation_derivative = 1 - tanh(fully_connected_layer_out)^2; % tanh function

% Tensorize and multiply each activation by the weight --> jacobian tensor
activation_derivative_tensor(:, 1, :) =  activation_derivative;
jacobian_tensor = weight_mat .* test_foo;


%% Calculate CPD of the jacobian_tensor

R = 16;

U = cpd(jacobian_tensor, R)

W = U{1};
V = U{2};
H = U{3};

%% Try to make sense of this thing

X_combined = (V' * squeeze(layer_inputs))';

g_i = 6

figure(2)
scatter(X_combined( :, 17-g_i), H(:, g_i))


