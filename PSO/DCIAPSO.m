function prototypes = DCIAPSO(trainSet, trainTargets, numDim, numCls)
% DCIAPSO (Dynamic Centroid Insertion and Adjustment with Particle Swarm
% Optimization) is a Prototype Generation (PG) algorithm based on IPADE-ID.
% It begins with each class centroid. After adjustment phase it tries to
% dynamically insert new prototypes for each class and adjust them.
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
        % The first set of solutions is composed with each class centroid
        if i == 1
            population(i).Prototypes = createPrototypes(trainSet, trainTargets, numDim, 'centroid', numCls);
        else
            population(i).Prototypes = createPrototypes(trainSet, trainTargets, numDim, 'prototypes', numCls);
        end
        
        % Default is k = 1
        knn = fitcknn(population(i).Prototypes(:, 1:end-1), population(i).Prototypes(:, end));
        population(i).Output = predict(knn, trainSet); clear knn
        
        population(i).Fitness = calculateMAUC(population(i).Output, trainTargets, numCls);
        % In the beginning all particles are their own pBest. Their
        % velocity is a matrix of zeros. As the algorithm iniciate with one
        % prototype per class, the matrix size follows the number of
        % classes and obviously the number of dimensions.
        population(i).PBest = population(i).Prototypes;
        population(i).PBestFit = population(i).Fitness;
        population(i).Velocity = zeros(numCls, numDim);
    end
    
    % Finding first global best particle and setting constant value
    [~, bestId] = max([population.Fitness]);
    gBest = population(bestId).Prototypes; 
    gBestFit = population(bestId).Fitness;
    c = 2;
    
    %======================================================================
    % Adjusting prototypes with PSO =======================================
    iter = 1;
    
    while iter <= 50
        % It is assumed that in the beginning of each loop it is known all
        % particles' properties values. PSO moves particles and in the end 
        % of the loop all properties values are updated
        
        % For each particle
        for i=1:N
            population(i).Velocity = population(i).Velocity + ...
                    (c * rand() * (population(i).PBest(:, 1:end-1) - population(i).Prototypes(:, 1:end-1))) + ...
                    (c * rand() * (gBest(:, 1:end-1) - population(i).Prototypes(:, 1:end-1)));
            population(i).Prototypes(:, 1:end-1) = population(i).Prototypes(:, 1:end-1) + population(i).Velocity;
            
            % Default is k = 1
            knn = fitcknn(population(i).Prototypes(:, 1:end-1), population(i).Prototypes(:, end));
            population(i).Output = predict(knn, trainSet); clear knn
        
            population(i).Fitness = calculateMAUC(population(i).Output, trainTargets, numCls);
            
            [population(i).PBest, population(i).PBestFit] = ...
                updateBest(population(i), population(i).PBest, population(i).PBestFit);
        end
        
        % Finding best solution
        [~, bestId] = max([population.Fitness]);
        
        %------------------------------------------------------------------
        % Insertion of new prototypes
        newSolution = population(bestId);
            
        for c=1:numCls
            % For each class a new prototype is selected from training set
            % and added to a temporary prototypes set, which is a copy of
            % the current prototypes set. If the performance increases
            % after insertion, prototypes set is updated.
            newPrototype = createPrototypes(trainSet, trainTargets, numDim, 'single', c);
                
            newSolution.Prototypes = [newSolution.Prototypes; zeros(1, numDim+1)];
            newSolution.Prototypes(end, :) = newPrototype;
            
            % Default is k = 1
            knn = fitcknn(newSolution.Prototypes(:, 1:end-1), newSolution.Prototypes(:, end));
            newSolution.Output = predict(knn, trainSet); clear knn
            
            newSolution.Fitness = calculateMAUC(newSolution.Output, trainTargets, numCls);
                
            if newSolution.Fitness > population(bestId).Fitness
                % Updating best solution so far
                population(bestId) = newSolution;
                % As the new prototype resulted in improvement, all
                % solutions will have a new prototype from the same class                
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
        
        % If the number of prototypes is bigger, it means there were
        % insertion for sure. In this case some values must be updated.
        % Otherwise, only gBest go through a possible update.
        if size(population(1).Prototypes, 1) > size(population(1).PBest, 1)
            % Updating Output, Fitness and bests
            for i=1:N
                if i ~= bestId
                    % Default is k = 1
                    knn = fitcknn(population(i).Prototypes(:, 1:end-1), population(i).Prototypes(:, end));
                    population(i).Output = predict(knn, trainSet); clear knn
                    
                    population(i).Fitness = calculateMAUC(population(i).Output, trainTargets, numCls);
                    % As there is a "new" set of prototypes after the
                    % insertion, each particle becomes its pBest again
                    population(i).PBest = population(i).Prototypes;
                    population(i).PBestFit = population(i).Fitness;
                    % Reseting velocity matrix
                    population(i).Velocity = zeros(size(population(i).Prototypes, 1), numDim);
                else
                    % As there is a "new" set of prototypes after the
                    % insertion, each particle becomes its pBest again
                    population(i).PBest = population(i).Prototypes;
                    population(i).PBestFit = population(i).Fitness;
                    % Reseting velocity matrix
                    population(i).Velocity = zeros(size(population(i).Prototypes, 1), numDim);
                end
            end
            
            % Forcing Update gBest for the "new" set of prototypes
            [~, bestId] = max([population.Fitness]);
            [gBest, gBestFit] = updateBest(population(bestId), gBest, 0);
        else
            % Updating gBest with tue current set of prototypes
            [~, bestId] = max([population.Fitness]);
            [gBest, gBestFit] = updateBest(population(bestId), gBest, gBestFit);
        end
        
        % Replacing
        clear newSolution
        
        % Iteration
        iter = iter+1;
    end
    
    % Returning the best set of prototypes
    prototypes = gBest; 
end


function [prototypes, fitness] = updateBest(population, bestProt, bestFit)
% Function to update pBest and gBest
    
    if population.Fitness > bestFit
        prototypes = population.Prototypes;
        fitness = population.Fitness;
    else
        % Making sure there will be a return
        prototypes = bestProt;
        fitness = bestFit;
    end
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