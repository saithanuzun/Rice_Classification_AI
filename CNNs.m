PretrainedCNNs = { 
    'alexnet', ...
    'googlenet', ...
    'resnet50',... 
    'vgg16',...
}; 

imds = imageDatastore('Rice_Image_Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsSubset = splitEachLabel(imds,100,'randomized');
[imdsTrain,imdsValidation] = splitEachLabel(imdsSubset,0.7,'randomized');

 
for n = 1:numel(PretrainedCNNs)

    netName = PretrainedCNNs{n};

    net = eval(netName);
    inputSize = net.Layers(1).InputSize;

    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandXTranslation', [-30 30], ...
        'RandYTranslation', [-30 30]);

    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain,'DataAugmentation', imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ...
        'MaxEpochs', 10, ...
        'InitialLearnRate', 0.0001, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augimdsValidation, ...
        'ValidationFrequency', 50, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    numClasses = numel(categories(imdsTrain.Labels));

    switch netName

        case {'alexnet','vgg16'}

            layersTransfer = net.Layers(1:end-3);

            layers = [
                layersTransfer
                fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
                softmaxLayer
                classificationLayer];

            netTransfer = trainNetwork(augimdsTrain, layers, options);


        case 'googlenet'

            lgraph = layerGraph(net);

            newFCLayer = fullyConnectedLayer(numClasses,'Name','loss3-classifier',...
                'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);

            lgraph = replaceLayer(lgraph,'loss3-classifier',newFCLayer);

            newClassLayer = classificationLayer('Name','output');
            lgraph = replaceLayer(lgraph,'output',newClassLayer);

            netTransfer = trainNetwork(augimdsTrain,lgraph,options);


        case 'resnet50'

            lgraph = layerGraph(net);

            newFCLayer = fullyConnectedLayer(numClasses,'Name','fc1000',...
                'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);

            lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);

            newClassLayer = classificationLayer('Name','ClassificationLayer_fc1000');
            lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);

            netTransfer = trainNetwork(augimdsTrain,lgraph,options);

    end


    YPred = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;

    accuracy = mean(YPred == YValidation);

    figure
    confusionchart(YValidation,YPred)
    title(['Confusion Matrix - ' netName ' | Accuracy: ' num2str(accuracy*100, '%.2f') '%'])



end
