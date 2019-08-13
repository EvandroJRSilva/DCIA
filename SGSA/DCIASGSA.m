function prototypes = DCIASGSA(trainSet, trainTargets, numDim, numCls)
% DCIASGSA (Dynamic Centroid Insertion and Adjustment with Simple Gravity
% Search Algorithm) is a Prototype Generation (PG) algorithm based on
% IPADE-ID. It begins with each class centroid. After adjustment phase it
% tries to dynamically insert new prototypes for each class and adjust
% them. The SGSA is a simpler version of GSA, first proposed in this
% research.
%
% Inputs
%   - trainSet : trxd matrix, in which tr is the total number of training
%   samples and d is the data set dimension;
%   - trainTargets : trx1 matrix with an integer class label for each
%   training sample;
%   - numDim : value of d, i.e., the data set dimension;
%   - numCls : the number of classes present in the data set.
%
% Outputs
%   - Generated Prototypes
%    
%    
% The algorithm consists of three steps: (1) Initialization, (2) Adjustment
% and (3) Insertion. Initialization step begins with the very centroids of
% each class, then these centroids are adjusted. Afterwards there is the
% attempt of new prototypes insertion. Steps 2 and 3 are repeated until
% finish criterion is found.

    % Creating first set of solutions
    N = 50;
    population(N) = Solution;
    
    for i=1:N
        if i == 1
            population(i).Prototypes = createPrototypes(trainSet, trainTargets, numDim, 'centroid', numCls);
        else
            population(i).Prototypes = createPrototypes(trainSet, trainTargets, numDim, 'prototypes', numCls);
        end
        
        % Default is k = 1
        knn = fitcknn(population(i).Prototypes(:, 1:end-1), population(i).Prototypes(:, end));
        population(i).Output = predict(knn, trainSet); clear knn
        
        population(i).MAUC = calculateMAUC(population(i).Output, trainTargets, numCls);
    end
    
    % =====================================================================
    % Adjusting prototypes with SGSA ======================================
    iter = 1;
    
    while iter <= 50
        % In the first iteration output and MAUC are already calculated.
        % From second iteration and forth its update after [possible]
        % insertion of new prototypes is carried on here
        if iter > 1
            for i=1:N
                % Default is k = 1
                knn = fitcknn(population(i).Prototypes(:, 1:end-1), population(i).Prototypes(:, end));
                population(i).Output = predict(knn, trainSet); clear knn
                
                population(i).MAUC = calculateMAUC(population(i).Output, trainTargets, numCls);
            end
        end
        
        [~, bestId] = max([population.MAUC]);
        
        % Updating solutions
        newPopulation = population;
        
        for i=1:N
            if i ~= bestId
                dist = solutionDistance(population(i).Prototypes(:, 1:end-1),...
                            population(bestId).Prototypes(:, 1:end-1));
                if dist == 0
                    pull = 0;
                else
                    pull = population(bestId).MAUC/(dist^2);
                end
                % For each prototype its dimensions are updated. All
                % populations have the same amount of prototypes
                for j=1:size(population(1).Prototypes, 1)
                    for d=1:size(population(1).Prototypes,2)-1
                        if population(i).Prototypes(j, d) > ...
                            population(bestId).Prototypes(j, d)
                                population(i).Prototypes(j, d) = ...
                                    population(i).Prototypes(j, d) - pull;
                        else
                            population(i).Prototypes(j, d) = ...
                                population(i).Prototypes(j, d) + pull;
                        
                        end
                    end
                end
                % Updating
                % Default is k = 1
                knn = fitcknn(newPopulation(i).Prototypes(:, 1:end-1), newPopulation(i).Prototypes(:, end));
                newPopulation(i).Output = predict(knn, trainSet); clear knn
                
                newPopulation(i).MAUC = calculateMAUC(population(i).Output, trainTargets, numCls);
            else % Exploitation
                possibleBetter = population(bestId);
                for j=1:size(population(1).Prototypes, 1)
                    % With 50% of probability an exploitation will occur in
                    % this prototype
                    if rand() < 0.5
                        for d=1:size(population(1).Prototypes,2)-1
                            % With 50% of probability an exploitation will
                            % occur in this dimension
                            if rand() < 0.5
                                exploit = rand()/100;
                                    % Choosing randomly if the exploit
                                    % value will increase or decrease the
                                    % dimension value
                                if rand() < 0.5
                                    possibleBetter.Prototypes(j, d) = ...
                                        possibleBetter.Prototypes(j, d) + exploit;
                                else
                                    possibleBetter.Prototypes(j, d) = ...
                                        possibleBetter.Prototypes(j, d) - exploit;
                                end
                            end
                        end
                    end
                end
                
                knn = fitcknn(possibleBetter.Prototypes(:, 1:end-1), possibleBetter.Prototypes(:, end));
                possibleBetter.Output = predict(knn, trainSet); clear knn
                
                possibleBetter.MAUC = calculateMAUC(possibleBetter.Output, trainTargets, numCls);
                if possibleBetter.MAUC > population(bestId).MAUC
                    % Updating
                    newPopulation(bestId) = possibleBetter;
                end
            end
        end
        
        % Replacing current population
        population = newPopulation;
        clear newPopulation
        
        % Insertion of new Centroids
        [~, bestId] = max([population.MAUC]);
        newSolution = population(bestId);
            
        for c=1:numCls
            % For each class a new prototype is selected from training set
            % and added to a temporary prorotypes set, which is a copy of
            % the current prototypes set. If the performance increases
            % after insertion, prototypes set is updated.
            newPrototype = createPrototypes(trainSet, trainTargets, numDim, 'single', c);
                
            newSolution.Prototypes = [newSolution.Prototypes; zeros(1, numDim+1)];
            newSolution.Prototypes(end, :) = newPrototype;
            
            knn = fitcknn(newSolution.Prototypes(:, 1:end-1), newSolution.Prototypes(:, end));
            newSolution.Output = predict(knn, trainSet); clear knn
            
            newSolution.MAUC = calculateMAUC(newSolution.Output, trainTargets, numCls);
                
            if newSolution.MAUC > population(bestId).MAUC
                % As the new prototype resulted in improvement, all
                % solutions will have a new prototype from the same class
                population(bestId) = newSolution;
                for s=1:N
                    if s ~= bestId
                        newPrototype = createPrototypes(trainSet, trainTargets, numDim, 'single', c);
                        population(s).Prototypes = [population(s).Prototypes; zeros(1, numDim+1)];
                        population(s).Prototypes(end, :) = newPrototype;
                    end
                end
                
            else
                % If no improvement comes from a new prototype, it is
                % erased
                newSolution.Prototypes(end, :) = [];
            end
        end
        
        % Iteration
        iter = iter+1;
    end
    
    % Returning the best set of prototypes
    [~, bestId] = max([population.MAUC]);
    prototypes = population(bestId).Prototypes;
    
end


function prototype = createPrototypes(trainSet, trainTargets, numDim, varargin)
% Function to create prototypes. It may create only one or a set of
% prototypes. It is also necessary to choose the 'type' of prototype, i.e.,
% if it is going to be a class centroid or not. If it is not a centroid,
% the class lable must be passed.
%
% Varargin
%   'single' : if typed, it indicates the request of a single prototype.
%   Otherwise the return is a set of prototypes;
%   class : label of the single prototype to be created. Must follow
%   'single'.
%   'centroid' : if typed it must be the first solution of the first
%   iteration of GSA, meaning that all prototypes will be centroids
%   numCls : number of classes must follow 'centroid' type.
%   'prototypes' : if typed it must be another solutions of the first
%   iteration of GSA, meaning that there will be a single prototype for
%   each class, but not centroids.
% There must always be 2 inputs for varargin: 'single' and class label,
% 'centroid' followed by the number of classes or 'prototypes', also
% followed by the number of classes.

    switch varargin{1}
        case 'single'
            class = varargin{2};
            inds = find(trainTargets == class);
            selected = randi(size(inds, 1));
            prototype = zeros(1, numDim+1);
    
            prototype(1, 1:numDim) = trainSet(inds(selected), :);
            prototype(1, end) = class;
        case 'centroid'
            numCls = varargin{2};
            prototype = zeros(numCls, numDim+1);
            for c=1:numCls
               inds = find(trainTargets == c);
               for d=1:numDim
                   prototype(c, d) = mean(trainSet(inds, d));
               end
               prototype(c, end) = c;
            end
        case 'prototypes'
            numCls = varargin{2};
            prototype = zeros(numCls, numDim+1);
            for c=1:numCls
                inds = find(trainTargets == c);
                selected = randi(size(inds, 1));
                prototype(c, 1:numDim) = trainSet(inds(selected), :);
                prototype(c, end) = c;
            end
    end
end