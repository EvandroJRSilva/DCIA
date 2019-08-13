function foldIdx = kfoldIndices(numCls, dataTNum, numFolds)
% Function for creating indices for the k folds    
    foldIdx(numFolds).indices = [];
    clsInd(numCls).indices = [];
    clsFoldInd(numCls).indices = [];
    % Getting the indices of instances from each class and then the inside
    % class indices for folds
    for i=1:numCls
        clsInd(i).indices = find(dataTNum == i);
        clsFoldInd(i).indices = crossvalind('Kfold', size(clsInd(i).indices, 1), numFolds);
    end
    
    for i=1:numFolds
        for j=1:numCls
            foldIdx(i).indices = [foldIdx(i).indices; clsInd(j).indices(clsFoldInd(j).indices == i)];
        end
    end
end