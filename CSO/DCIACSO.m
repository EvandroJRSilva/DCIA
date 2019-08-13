function prototypes = DCIACSO(trainSet, trainTargets, numDim, numCls)
% DCIAPSO (Dynamic Centroid Insertion and Adjustment with Competitive Swarm
% Optimizer) is a Prototype Generation (PG) algorithm based on IPADE-ID.
% It begins with each class centroid. After adjustment phase it tries to
% dynamically insert new prototypes for each class and adjust them.
%
% Inputs
%   - trainSet : trxd matrix, in which tr is the total number of training
%   samples and d is the data set dimension;
%   - trainTargets : trx1 matrix with an integer class label for each
%   training sample;
%   - numDim : value of d, i.e., the data set dimension, or the number of 
%   attributes;
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
        
        population(i).Fitness = calculateMAUC(population(i).Output, trainTargets, numCls);
        % In the beginning their velocity is a matrix of zeros. As the 
        % algorithm iniciate with one prototype per class, the matrix size 
        % follows the number of classes and obviously the number of 
        % dimensions.
        population(i).Velocity = zeros(numCls, numDim);
    end
    
    %======================================================================
    % Adjusting prototypes with CSO =======================================
    iter = 1;
    
    % Algorithm parameter
    phi = 0.1;
    
    while iter <= 50
        % It is assumed that in the beginning of each loop it is known all
        % particles' properties values. CSO moves particles and in the end 
        % of the loop all properties values are updated
        
        
        % Calculating mean position of current population. It is actually a
        % matrix in which each line holds the mean values of all population
        % for that line
        meanProt = zeros(size(population(1).Prototypes, 1), numDim);
        
        values = zeros(1, N);
        
        for d=1:numDim
            for p=1:size(population(1).Prototypes, 1)
                for n=1:50
                    values(n) = population(n).Prototypes(p, d);
                end
                meanProt(p, d) = mean(values);
            end
        end
        
        newPopulation = [];
        
        while ~isempty(population)
            selected = randperm(size(population, 2), 2);
            
            if population(selected(1)).Fitness > population(selected(2)).Fitness
                winner = population(selected(1));
                loser = population(selected(2));
                % Erasing selected population
                population([selected(1) selected(2)]) = [];
            else
                winner = population(selected(2));
                loser = population(selected(1));
                % Erasing selected population
                population([selected(1) selected(2)]) = [];
            end
            
            % Updating loser stats
            loser.Velocity = (rand()*loser.Velocity) + ...
                (rand()*(winner.Prototypes(:, 1:end-1) - loser.Prototypes(:, 1:end-1))) + ...
                (phi*rand()*(meanProt - loser.Prototypes(:, 1:end-1)));
            loser.Prototypes(:, 1:end-1) = loser.Prototypes(:, 1:end-1) + loser.Velocity;
            
            % Default is k = 1
            knn = fitcknn(loser.Prototypes(:, 1:end-1), loser.Prototypes(:, end));
            loser.Output = predict(knn, trainSet); clear knn
            
            loser.Fitness = calculateMAUC(loser.Output, trainTargets, numCls);
            
            newPopulation = [newPopulation, winner, loser];
        end
        
        population = newPopulation;
        clear newPopulation
        
        
        % Finding best solution
        [~, bestId] = max([population.Fitness]);
        
        %------------------------------------------------------------------
        % Insertion of new prototypes
        newSolution = population(bestId);
        
        insertion = 0; % flag indicating insertion
            
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
                insertion = 1;
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
        
        % If there were prototypes insertion there must be values' update.
        if insertion
            for i=1:N
                if i ~= bestId
                    % Default is k = 1
                    knn = fitcknn(population(i).Prototypes(:, 1:end-1), population(i).Prototypes(:, end));
                    population(i).Output = predict(knn, trainSet); clear knn
                    
                    population(i).Fitness = calculateMAUC(population(i).Output, trainTargets, numCls);
                    % As there is a "new" set of prototypes the search goes
                    % back to the beginning. Velocity is then reset.
                    population(i).Velocity = zeros(size(population(i).Prototypes, 1), numDim);
                else
                    % As there is a "new" set of prototypes the search goes
                    % back to the beginning. Velocity is then reset.
                    population(i).Velocity = zeros(size(population(i).Prototypes, 1), numDim);
                end
            end
        end
        
        % Replacing
        clear newSolution
        
        % Iteration
        iter = iter+1;
    end
    
    % Finding best solution
    [~, bestId] = max([population.Fitness]);
    
    % Returning the best set of prototypes
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