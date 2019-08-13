function prototypes = DCIAGSA(trainSet, trainTargets, numDim, numCls)
% DCIAGSA (Dynamic Centroid Insertion and Adjustment with Gravity 
% Search Algorithm) is a Prototype Generation (PG) algorithm based on
% IPADE-ID. It begins with each class centroid. After adjustment phase it
% tries to dynamically insert new prototypes for each class and adjust
% them.
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
        population(i).Output = test(trainSet, population(i).Prototypes);
        population(i).MAUC = calculateMAUC(population(i).Output, trainTargets, numCls);
        population(i).Velocity = 0; population(i).Mass = 0;
        population(i).Force = 0; population(i).Acceleration = 0;
    end
    
    % =====================================================================
    % Adjusting prototypes with GSA =======================================
    G_ini = 1;
    T = 50;
    
    for i=1:T+1
        
        if i==1
            G = G_ini;
        else
            G = G_ini*(1 - (i/T));
            % In the first iteration, output and MAUC are already
            % calculated. From second iteration and forth, it will be
            % updated
            for j=1:N
                population(j).Output = test(trainSet, population(j).Prototypes);
                population(j).MAUC = calculateMAUC(population(j).Output, trainTargets, numCls);
            end
        end
        
        % Calculating Mass
        for j=1:N
            population(j).Mass = (population(j).MAUC - min([population.MAUC]))/...
                sum([population.MAUC] - min([population.MAUC]));
        end
        
        newPopulation = population;
        
        for j=1:N           
            % In the iteration T+1 there will be only output and MAUC
            % calculations over the T iteration update
            if i<= T
                for l=1:N
                    if l ~= j
                        dist = solutionDistance(population(j).Prototypes(:, 1:end-1),...
                            population(l).Prototypes(:, 1:end-1));
                        population(j).Force = population(j).Force + ...
                            (G *((population(j).Mass*population(l).Mass)/...
                            (dist^2)))*(population(l).Prototypes(:, 1:end-1) - ...
                            population(j).Prototypes(:, 1:end-1));
                    end
                end
                
                % Updating velocity and prototypes' positions
                newPopulation(j).Acceleration = population(j).Force./population(j).MAUC;
                newPopulation(j).Velocity = (rand.*population(j).Velocity) + population(j).Acceleration;
                newPopulation(j).Prototypes(:, 1:end-1) = ...
                    population(j).Prototypes(:, 1:end-1) + population(j).Velocity;
                % Reseting force and acceleration
                newPopulation(j).Force = 0; newPopulation(j).Acceleration = 0;
            end
        end
        
        % Replacing current population
        population = newPopulation;
        
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
            newSolution.Output = test(trainSet, newSolution.Prototypes);
            newSolution.MAUC = calculateMAUC(newSolution.Output, trainTargets, numCls);
                
            if newSolution.MAUC > population(bestId).MAUC
                % As the new prototype resulted in improvement, all
                % solutions will have a new prototype from the same class
                population(bestId) = newSolution;
                for s=1:N
                    population(s).Velocity = 0;
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