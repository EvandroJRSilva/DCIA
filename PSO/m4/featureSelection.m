function [trainSet, testSet, numDim] = featureSelection(trainSet, trainTargets, testSet)
% Feature Selection with Whale Optimization Algorithm (WOA). From:
%_________________________________________________________________________%
%  Whale Optimization Algorithm (WOA) source codes demo 1.0               %
%                                                                         %
%  Developed in MATLAB R2011b(7.13)                                       %
%                                                                         %
%  Author and programmer: Seyedali Mirjalili                              %
%                                                                         %
%         e-Mail: ali.mirjalili@gmail.com                                 %
%                 seyedali.mirjalili@griffithuni.edu.au                   %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                                                                         %
%   Main paper: S. Mirjalili, A. Lewis                                    %
%               The Whale Optimization Algorithm,                         %
%               Advances in Engineering Software , in press,              %
%               DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008   %
%_________________________________________________________________________%
%
% ***                                                 ***
% ***In this file WOA was adapted to work with DCIAPSO***
% ***                                                 ***

    numDim = size(trainSet, 2); % There is no need for it to be passed in the function call
    % Memory will store all searched solutions, and in the last column it
    % holds solution's fitness
    memory = zeros(1, numDim+1);
    % Swarm size
    N = 20;
    % Algorithm parameters
    t = 100;
    
    %Initializing population
    population(N).position = zeros(1, numDim);
    population(N).fitness = 0;
    
    for i=1:N
        for d=1:numDim
            population(i).position(d) = randi(2) - 1;
        end
        population(i).fitness = calculateFit(population(i).position, ...
            trainSet, trainTargets);
        if ~ismember(population(i).position, memory(:, 1:end-1), 'rows')
            memory = [memory; population(i).position population(i).fitness];
        end
    end
    
    % Finding the best solution
    [~, bestId] = max([population.fitness]);
    best = population(bestId).position; 
    bestFit = population(bestId).fitness;
    
    iter = 1;
    
    while iter < t
        % Calculating mutation rate
        r = 0.9 + ((-0.9*(iter-1))/(t - 1));
        %Updating parameters
        a = 2 - iter*(2/t);
        a2 = -1+iter*((-1)/t); % As in the original
        b=1;
        
        
        newPop = population;
        % For each solution
        for i=1:N
            % Updating parameters
%             C = 2*rand();
            A = (2*a*rand()) - a;
            l=(a2-1)*rand+1;
            p = rand();
            
            
            if p < 0.5
                if abs(A) < 1
                    % Mutation on best solution
                    X_mut = best;
                    for d=1:numDim
                        if rand() > r
                            if X_mut(d) == 1
                                X_mut(d) = 0;
                            else
                                X_mut(d) = 1;
                            end
                        end
                    end
                    % Crossover
                    for d=1:numDim
                        if rand() >= 0.5
                            newPop(i).position(d) = X_mut(d);
                        else
                            newPop(i).position(d) = population(i).position(d);
                        end
                    end
                    % Verifying if this new solution already exists on
                    % memory
                    [exist, id] = ismember(newPop(i).position, memory(:, 1:end-1), 'rows');
                    if exist
                        % If the solution was already searched, no fitness
                        % calculation is necessary
                        newPop(i).fitness = memory(id, end);
                    else
                        newPop(i).fitness = calculateFit(newPop(i).position, ...
                            trainSet, trainTargets);
                        % Updating memory with new entrance
                        memory = [memory; newPop(i).position newPop(i).fitness];
                    end
                else
                    % As in the original
                    %   Selecting random solution for mutation phase
                    rand_Id = floor(N*rand()+1);
                    X_mut = population(rand_Id).position;
                    
                    % Mutation
                    for d=1:numDim
                        if rand() > r
                            if X_mut(d) == 1
                                X_mut(d) = 0;
                            else
                                X_mut(d) = 1;
                            end
                        end
                    end
                    % Crossover
                    for d=1:numDim
                        if rand() >= 0.5
                            newPop(i).position(d) = X_mut(d);
                        else
                            newPop(i).position(d) = population(i).position(d);
                        end
                    end
                    % Verifying if this new solution already exists on
                    % memory
                    [exist, id] = ismember(newPop(i).position, memory(:, 1:end-1), 'rows');
                    if exist
                        % If the solution was already searched, no fitness
                        % calculation is necessary
                        newPop(i).fitness = memory(id, end);
                    else
                        newPop(i).fitness = calculateFit(newPop(i).position, ...
                            trainSet, trainTargets);
                        % Updating memory with new entrance
                        memory = [memory; newPop(i).position newPop(i).fitness];
                    end
                end
            else
                for d=1:numDim
                    distance2Leader=abs(best(d) - population(i).position(d));
                    newPop(i).position(d) = ...
                        distance2Leader*exp(b.*l).*cos(l.*2*pi)+best(d);
                end
                % Verifying if this new solution already exists on memory
                [exist, id] = ismember(newPop(i).position, memory(:, 1:end-1), 'rows');
                if exist
                    % If the solution was already searched, no fitness
                    % calculation is necessary
                    newPop(i).fitness = memory(id, end);
                else
                    newPop(i).fitness = calculateFit(newPop(i).position, ...
                        trainSet, trainTargets);
                    % Updating memory with new entrance
                    memory = [memory; newPop(i).position newPop(i).fitness];
                end
            end
        end
        
        population = newPop;
        clear newPop
        
        % Finding the best solution
        [~, newBestId] = max([population.fitness]);
        newBest = population(newBestId).position; 
        newBestFit = population(newBestId).fitness;
        if newBestFit > bestFit
            best = newBest;
            bestFit = newBestFit;
        end
        clear newBestId newBest newBestFit
        
        iter = iter+1;
    end
    
    % Returning best solution found
    trainSet = trainSet(:, find(best));
    testSet = testSet(:, find(best));
    numDim = size(trainSet, 2);
end

function fitness = calculateFit(solution, trainSet, trainTargets)

    alpha = 0.99; beta = 1 - alpha;
    selTrSet = trainSet(:, find(solution));
    output = zeros(size(trainSet, 1), 1);
    
    
    % Calculating distances and classifying
    for i=1:size(trainSet, 1)
        dist = distance(transpose(selTrSet(i, :)), transpose(selTrSet));
        dist(i) = NaN;
        
        [~, nearN] = sort(dist);
        output(i) = mode(trainTargets(nearN(1:5)));
    end
    
    % Calculating fitness
    nClass = size(unique(trainTargets), 1);
    mauc = calculateMAUC(output, targets, nClass);
    
    fitness = (alpha*mauc) + (beta*(size(find(solution),2)/size(trainSet,2)));
end