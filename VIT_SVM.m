clear; clc;

net = visionTransformer('base-16-imagenet-384');
inputSize = net.Layers(1).InputSize;

imds = imageDatastore('Rice_Image_Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsSubset = splitEachLabel(imds,400,'randomized');
[imdsTrain,imdsValidation] = splitEachLabel(imdsSubset,0.7,'randomized');

numTrain = numel(imdsTrain.Labels);
numVal   = numel(imdsValidation.Labels);

featureDim = 1000;   

trainFeatures = zeros(numTrain,featureDim);
valFeatures   = zeros(numVal,featureDim);

trainLabels = imdsTrain.Labels;
valLabels   = imdsValidation.Labels;

for i = 1:numTrain
    img = im2single(imresize(readimage(imdsTrain,i),inputSize(1:2)));
    trainFeatures(i,:) = extractdata(predict(net,dlarray(img,'SSC')));
end

for i = 1:numVal
    img = im2single(imresize(readimage(imdsValidation,i),inputSize(1:2)));
    valFeatures(i,:) = extractdata(predict(net,dlarray(img,'SSC')));
end

trainFeatures = normalize(trainFeatures);
valFeatures   = normalize(valFeatures);

model = fitcecoc(trainFeatures,trainLabels);

YPred = predict(model,valFeatures);

accuracy = mean(YPred == valLabels)

figure
confusionchart(valLabels,YPred)
title(['ViT + SVM | Accuracy: ' num2str(accuracy*100, '%.2f') '%'])