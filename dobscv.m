function foldIdx = dobscv(numCls, dataF, dataTNum, numFolds)
% DOB-SCV
% X. Zeng, T.R. Martinez, Distribution-balanced stratified cross validation 
% for accuracy estimation, Journal of Experimental and Theoretical 
% Artificial Intelligence 12 (1) (2000) 1–12.

    % Instance indices of each class
    class(numCls).indices = [];
    % Fold indices for each instance
    class(numCls).foldIndices = [];
    % Indices of instances for each fold
    foldIdx(numFolds).indices = [];
    
    
    for i=1:numCls
        class(i).indices = find(dataTNum == i); 
        auxFold = zeros(size(class(i).indices,1), 1);
        %Separating instances into k folds with DOB-SCV
        while any(auxFold == 0)
            currentIdx = randi(size(class(i).indices,1)); 
            % If the current instance does not belong to any fold, it will
            % belong to fold 1 and its nearest neighbors that also do not
            % belong to any fold will be set to folds 2, 3, and so on
            if auxFold(currentIdx) == 0
                % Finding current index on dataF
                current = class(i).indices(currentIdx);
                % Finding distance from current instance to all others of
                % the same class
                dist = distance(transpose(dataF(current, :)), ...
                    transpose(dataF(class(i).indices, :)));
                % Finding sorted indices
                [~, sortId] = sort(dist);
                % As it does not belong to any fold, it will be set to fold
                % one
                auxFold(currentIdx) = 1;
                fold = 2; % auxiliar for the remainder instances
                % If the number of unassingned instances is bigger than
                % number of folds, there will be another turn of fold
                % assignment. Otherwise this will be the last turn.
                if size(find(~auxFold),1) >= (numFolds - 1)
                    for j=2:size(sortId,2)
                        if fold > numFolds
                            break
                        elseif auxFold(sortId(j)) == 0
                            auxFold(sortId(j)) = fold;
                            fold = fold + 1;
                        end
                    end
                else
                    % Guarantees assignment to folds 2 - 9
                    limit = size(find(~auxFold),1) + 1;
                    for j=2:size(sortId,2)
                        if fold > limit
                            break
                        elseif auxFold(sortId(j)) == 0
                            auxFold(sortId(j)) = fold;
                            fold = fold + 1;
                        end
                    end
                end
            end
        end
        
        class(i).foldIndices = auxFold;
        % Special case for small classes with repeated doubled instances.
        % Forcing different folds
        if size(find(dataTNum == i), 1) <= numFolds
            class(i).foldIndices = [];
            for s=1:size(find(dataTNum == i), 1)
                class(i).foldIndices = [class(i).foldIndices; s];
            end
        end
    end
    
    for i=1:numFolds
        for j=1:numCls
            foldIdx(i).indices = [foldIdx(i).indices; class(j).indices(class(j).foldIndices == i)];
        end
    end
end