function [dataF, dataTNum, numCls] = getDataSets(db, dbFolder)
% Function to put together the entire data from datasets provided from AMCS
% source code
    
    path = ...
        strcat(dbFolder, db,'.mat');
    
    % It loads data a struct with feature and label fields
    base = load(path);
    
    dataF = base.data.feature;
    dataTNum = base.data.label;
    numCls = size(unique(base.data.label),1);
    
end