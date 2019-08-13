% Main scrpit for DCIASGSA

%% Loading data, variables and objects
%=============================================
%=============DATA============================    
numFolds = 5;

% Result object
results(numFolds) = Results;


% DBs without Shuttle
db = {'anneal', 'arrhythmia', 'autos', 'balanceScale', 'contraceptive', ...
    'flare', 'gene_splice', 'glass', 'horse', 'landsat', 'led7digit', ...
    'lymphography', 'nursery', 'page-blocks', 'penbased', 'post-operative', ...
    'satimage', 'segment', 'thyroid', 'vehicle', 'vowel', 'wine', 'wineqr', ...
    'yeast', 'zoo', 'amcs_abalone', 'amcs_carEval',  'amcs_dermatology', ...
    'amcs_ecoli', 'amcs_gene_splice', 'amcs_hayes-roth', 'amcs_newThyroid', ...
    'amcs_soybean', 'amcs_wineqw', 'keel_cleveland', 'keel_hayes-roth', ...
    'keel_page-blocks', 'keel_wineqw', 'uci_abalone', 'uci_carEval', ...
    'uci_cleveland', 'uci_dermatology', 'uci_ecoli', 'uci_hayes-roth', ...
    'uci_newThyroid', 'uci_soybean'};


% MATLAB workspace Path
mainPath = 'matlab path\';
% Algorithm Path/Folder
algPath = 'algorithm name\';
% Classifier Folder
%   This path may change, following the current classifier
clfPath = '1NN\';
% % Version Folder
version = 'm1\';
% Results Folder
resultsFolder = 'results\';

% Executing DCIA for each DB
for dataSet = 1:length(db)
    % Pre-process==========================================================
    %   Getting Data Sets from a specified fold All DBs, which is inside
    %   Matlab workspace
    [dataF, dataTNum, numCls] = getDataSets(db{dataSet}, strcat(mainPath, 'All DBs\'));
    
    % Separating data with k-fold
    foldIdx = dobscv(numCls, dataF, dataTNum, numFolds);
    
    % Printing the name of the current data set
    disp(db{dataSet});
    
    % For each fold
    for k=1:numFolds
        % Printing current iteration
        str1 = 'ITERAÇÃO'; str2 = ' '; str3 = num2str(k);
        disp([str1 str2 str3]);
        
        % Test Set
        testIdx = foldIdx(k).indices;
        testSet = dataF(testIdx, :); testTargets = dataTNum(testIdx);
        
        % Train Set
        trainIdx = [];
        for i=1:numFolds
            if i ~= k
                trainIdx = [trainIdx; foldIdx(i).indices];
            end
        end
        trainSet = dataF(trainIdx, :); trainTargets = dataTNum(trainIdx);
        
        % UNDERSAMPLING----------------------------------------------------
        [trainSet, trainTargets] = undersampling(trainSet, trainTargets, numCls, numFolds);
        %------------------------------------------------------------------
        % OVERSAMPLING-----------------------------------------------------
        [trainSet, trainTargets] = oversampling(trainSet, trainTargets, numCls, numFolds, numDim);
        %------------------------------------------------------------------
        
        % Normalizing features
        [trainSet, testSet] = normalization(trainSet, testSet);
        
        prototypes = DCIASGSA(trainSet, trainTargets, numDim, numCls);
        
        % Training and testing classifier with generated prototypes
        knn = fitcknn(prototypes(:, 1:end-1), prototypes(:, end));
        output = predict(knn, testSet); clear knn
        
        % Calculating performance with several metrics
        %   AUCarea
        aucarea = calculateAUCarea(output, testTargets);
        %   MAUC
        %       In some cases the number of classes will be lower, as a
        %       class may not be present in the test set.
        nClass = size(unique(testTargets), 1);
        mauc = calculateMAUC(output, testTargets, nClass);
        %   Confusion Matrix
        confusionMatrix = generateConfusionMatrix(output, testTargets, numCls);
        %   OTP : Overall True Positive, or sum of all True Positives
        %   Sensitivities: True Positives for each class
        [otp, sensitivities] = calculateTP(numCls, confusionMatrix);
        %   Kappa
        kappa = calculateKappa(confusionMatrix, numCls, otp, size(testSet,1));
        %   CBA
        cba = calculateCBA(nClass, confusionMatrix, size(unique(testTargets), 1));
        %   GMean : root of products
        %   GMOVO : Average G-Mean
        [gmean, gmovo] = calculateGM(numCls, sensitivities, size(unique(testTargets), 1));
        %   F-Measure
        fm = calculateFM(numCls, confusionMatrix, sensitivities);
        %   Overall Accuracy
        acc = otp/size(testSet, 1);
        
        % Filling results object
        results(k).MAUC = mauc; results(k).AUCA = aucarea;
        results(k).GMean = gmean; results(k).GMOVO = gmovo;
        results(k).Sensitivities = sensitivities;
        results(k).Kappa = kappa; results(k).CBA = cba;
        results(k).FM = fm; results(k).Accuracy = acc;
        results(k).Dim = numDim; results(k).NumProt = size(prototypes, 1);        
    end
    
    delete(gcp('nocreate'))
    
    % Saving Result object
    fullPath = strcat(mainPath, algPath, clfPath, version, resultsFolder);
    
    fileName = strcat(db{dataSet}, '_', num2str(numFolds), 'folds');
    
    save(strcat(fullPath, fileName), 'results')
end

function [trainSet, trainTargets] = undersampling(trainSet, trainTargets, numCls, numFolds)
% Clean Tomek Links instances based on their class probability.
% Algorithm:
%   - Find all instances that form Tomek links
%       - Erase these instances from data set if they do not belong to
%       small class

    percCls = zeros(numCls, 1);
    for c=1:numCls
        percCls(c) = size(find(trainTargets == c), 1)/size(trainTargets,1);
    end
        
    tlList = [];
    for i=1:size(trainSet, 1)
        if ~ismember(i, tlList)
            dist = distance(transpose(trainSet(i, :)), transpose(trainSet));
            dist(i) = NaN; % Distance to itself
            nearest = find(dist == min(dist));

            if size(nearest, 2) == 1
                dist2 = distance(transpose(trainSet(nearest, :)), transpose(trainSet));
                dist2(nearest) = NaN; % Distance to itself
                nearest2 = find(dist2 == min(dist2));

                if size(nearest2, 2) == 1 && nearest2 == i
                    if trainTargets(i) ~= trainTargets(nearest)
                        tlList = [tlList nearest nearest2];
                    end
                end
            end
        end
    end

    if ~isempty(tlList)
        toErase = [];
        for i=1:size(tlList, 2)
            % Very small classes will not be deleted
            if size(trainTargets(tlList(i)), 1) > numFolds
                if rand() <= percCls(trainTargets(tlList(i)))
                    toErase = [toErase ; tlList(i)];
                end
            end
        end
        trainSet(toErase, :) = []; trainTargets(toErase, :) = [];
    end
end

function [trainSet, trainTargets] = oversampling(trainSet, trainTargets, numCls, numFolds, numDim)
% Add synthetic instances to all classes with less than 2k instances, with
% k being the number of folds
    addSet = []; addTargets = [];
    for i=1:numCls
        numInst = size(find(trainTargets == i), 1);
        if numInst < (2*numFolds)
            instances = trainSet(trainTargets == i, :);

            if ~isempty(instances)
                if numInst > 1
                    % One new synthetic instance between each pair of
                    % instances
                    for n=1:size(instances,1)-1
                        for m=n+1:size(instances,1)
                            newInst = zeros(1, numDim);
                            for d=1:numDim
                                newInst(1, d) = ...
                                    mean([instances(n, d) instances(m, d)]);
                            end
                            addSet = [addSet; newInst];
                            addTargets = [addTargets; i];
                        end
                    end
                else
                    % In this case there will be 2 * numDim neighbors, 
                    % two for each dimension, + and - half of its mean 
                    % value
                    for d=1:numDim
                        meanDim = mean(trainSet(:, d));

                        newInsts = zeros(2, numDim);
                        newInsts(1,:) = instances; 
                        newInsts(2,:) = instances;

                        newInsts(1, d) = newInsts(1, d)+(0.5*meanDim);
                        newInsts(2, d) = newInsts(2, d)-(0.5*meanDim);

                        addSet = [addSet; newInsts];
                        addTargets = [addTargets; i; i];
                    end   
                end
            end
        end
    end
    trainSet = [trainSet; addSet]; trainTargets = [trainTargets; addTargets];
end