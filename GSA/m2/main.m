% Main VDBC Script to initialize VDBC execution

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
% Version Folder
version = 'm2\';
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
        
        % Normalizing features
        [trainSet, testSet] = normalization(trainSet, testSet);
        
        prototypes = DCIAGSA(trainSet, trainTargets, numDim, numCls);
        
         % Training classifier with generated prototypes
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
    
    % Saving Result object
    fullPath = strcat(mainPath, algPath, clfPath, version, resultsFolder);
    
    fileName = strcat(db{dataSet}, '_', num2str(numFolds), 'folds');
    
    save(strcat(fullPath, fileName), 'results')
end