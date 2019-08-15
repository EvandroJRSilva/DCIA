function [trainSet, testSet, numDim] = featureSelection(trainSet, trainTargets, testSet)
% Feature Selection with Adapted SYMON

    % SYMON parameters-----------------------------------------------------
    trSize = size(trainSet, 1);
    HMCR = 0.65;
    NVAR = size(trainSet, 2);
    HMS = 200;
    PARmax = 0.9;
    PARmin = 0.5;
    MaxItr = 200;
    currentIteration  = 0;
    
    r = 1;
    coef = 1;
    d = floor(coef * NVAR / 5);
    if d == 0
        d = 1;
    end
    
    data = zeros(size(trainSet,1), size(trainSet,2)+1);
    data(:, 1:end-1) = trainSet; data(:, end) = trainTargets;
    
    Correlations = SU(data, NVAR+1);
    Correlations = transpose(Correlations);
    Correlations = sortrows(Correlations);
    %----------------------------------------------------------------------
    
    %Initializing population-----------------------------------------------
    [HM,F] = initializHS(HMS,NVAR,trainSet, trainTargets);
    
    % Search---------------------------------------------------------------
    while(currentIteration < MaxItr)
    
        PAR=(PARmax-PARmin)/(MaxItr)*currentIteration+PARmin;

        % improvise a new harmony vector
        for i =1:NVAR
            if( rand < HMCR ) % memory consideration
                index = randi([1 HMS],1,1);
                NCHV(i) = HM(index,i);
                pvbRan = rand;
                if( pvbRan > PAR) % pitch adjusting
                    if(NCHV(i) == 1)
                        NCHV(i) = 0;
                    else
                        NCHV(i) = 1;
                    end
                end
            else
                if(rand > 0.5)
                    NCHV(i)= 1;
                else
                    NCHV(i)= 0;
                end
            end
        end

        NCHV = LocalSearch(NCHV, d, r, NVAR, Correlations);
        newFitness = calculateFit(NCHV, trainSet, trainTargets);
%         newFitness = Fitness(NCHV,NVAR,Train,Test,tr_lbl,ts_lbl, trSize, tsSize);
        [HM,F] = UpdateHM(newFitness,F,HM,HMS,NCHV,NVAR );

        currentIteration = currentIteration + 1;
        if(currentIteration == MaxItr)
            break;
        end
    end
    % Finding the best solution
    [~, bestId] = max(F);
    best = HM(bestId);
  
    % Returning best solution found
    trainSet = trainSet(:, find(best));
    testSet = testSet(:, find(best));
    numDim = size(trainSet, 2);
end

function fitness = calculateFit(solution, trainSet, trainTargets)
    
    selTrSet = trainSet(:, find(solution));
    numCls = size(unique(trainTargets), 1);
    
    % In order to calculate distance only once
    allDist = distance(transpose(selTrSet), transpose(selTrSet));
    
    mean_inner = zeros(1, numCls); std_inner = zeros(1, numCls);
    mean_outer = zeros(1, numCls); std_outer = zeros(1, numCls);
    
    for c=1:numCls
        clsIdx = find(trainTargets == c);
        
        % Inner distances
        if size(clsIdx, 1) > 1
            numPair_one = nchoosek(size(clsIdx, 1), 2);
            dist_one = zeros(1, numPair_one);
            
            pair = 1;
            for f = 1:size(clsIdx, 1) - 1
                for s = f+1:size(clsIdx, 1)
                    dist_one(pair) = allDist(clsIdx(f), clsIdx(s));
                    pair = pair+1;
                end
            end
        else
            dist_one = 0;
        end
        
        mean_inner(1, c) = mean(dist_one); std_inner(1, c) = std(dist_one);
        
        % Outer distances
        trainIdx = 1:size(trainSet, 1);
        trainIdx(clsIdx) = [];
    
        [A,B] = meshgrid(clsIdx, trainIdx);
        p = cat(2,A',B');
        pairs = reshape(p,[],2);
    
        numPair_two = size(pairs, 1);
        dist_two = zeros(1, numPair_two);
        
        pair = 1;
        for p = 1:size(pairs, 1)
            dist_two(pair) = allDist(pairs(p, 1), pairs(p, 2));
            pair = pair+1;
        end

        mean_outer(1, c) = mean(dist_two); std_outer(1, c) = std(dist_two); 
    end
    
    % Pessimistic fitness
    fitness = sum((mean_outer - std_outer) - (mean_inner + std_inner));
end

function [Correlations] = SU(data,class_id)
% ORIGINAL
%   No modifications performed in this function
    y = class_id;

    for x = 1:(class_id - 1)
        S = 2*(entropy(data(:,x)) - conditionalEntropy(data(:,x),data(:,y)));
        M = entropy(data(:,x)) + entropy(data(:,y));
        Correlations(x, 1) = S / M;
        Correlations(x, 2) = x;
    end
    Correlations(x, 1) = Correlations(x, 1) / sum(Correlations(x, 1));
end

function z = entropy(x)
% ORIGINAL
%   No modifications performed in this function
%
% Compute entropy H(x) of a discrete variable x.
% Written by Mo Chen (mochen80@gmail.com).
    n = numel(x);
    x = reshape(x,1,n);
    [u,~,label] = unique(x);
    p = full(mean(sparse(1:n,label,1,n,numel(u),n),1));
    z = -dot(p,log2(p+eps));
end

function z = conditionalEntropy(x, y)
% ORIGINAL
%   No modifications performed in this function
%
% Compute conditional entropy H(x|y) of two discrete variables x and y.
% Written by Mo Chen (mochen80@gmail.com).
    assert(numel(x) == numel(y));
    n = numel(x);
    x = reshape(x,1,n);
    y = reshape(y,1,n);
    
    l = min(min(x),min(y));
    x = x-l+1;
    y = y-l+1;
    k = max(max(x),max(y));
    
    y = round(y);
    x = round(x);
    idx = 1:n;
    
    Mx = sparse(idx,x,1);%,n,k,n);
    My = sparse(idx,y,1);%,n,k,n);
    Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
    Hxy = -dot(Pxy,log2(Pxy+eps));

    Py = mean(My,1);
    Hy = -dot(Py,log2(Py+eps));

    % conditional entropy H(x|y)
    z = Hxy-Hy;
end

function [HM,F] = initializHS(HMS,NVAR,trainSet, trainTargets)
% ADAPTED
%   Multiclass fitness calculation for train set only, with MAUC metric. As
%   it does not follow the original fitness calculation for binary classes,
%   it is possible that the algorithm does not perform as well as reported
    for i=1:HMS
        for j=1:NVAR
            if(rand > 0.5)
                HM(i,j)= 1;
            else
                HM(i,j)= 0;
            end
        end
        if(OneCounter(HM(i,:),NVAR)>0)
%             F(i) = Fitness(HM(i,:),NVAR,trainSet,Test,trainTargets,ts_lbl,trSize,tsSize);
            F(i) = calculateFit(HM(i,:), trainSet, trainTargets);
        else
            F(i) = 0;
        end
    end
end

function Ones = OneCounter(NHV,NVAR)
% ORIGINAL
%   No modifications performed in this function
    Ones = 0;

    for l = 1: NVAR
        if (NHV(l) == 1)
            Ones = Ones + 1;
        end
    end
end

function NewSolution = LocalSearch(solution, d, r, NVAR,Correlations)
% ADAPTED
%   The algorithm could not select new feature if none were selected, and
%   could not remove a feature if all were selected. Examples: [1 1 1 1] or
%   [0 0 0 0] would return error. In these cases a random feature is
%   selected or removed
    Selected = OneCounter(solution,NVAR);
    
    if Selected == NVAR
        NewSolution = solution;
        NewSolution(randi(NVAR)) = 0;
    elseif Selected == 0
        NewSolution = solution;
        NewSolution(randi(NVAR)) = 1;
    else
        if(Selected == d)
            NewSolution = Ripple_Add(r, d, solution, Correlations);
            NewSolution = Ripple_Remove(r, d, NewSolution, Correlations);
        end

        if(Selected > d)
            NewSolution = Ripple_Remove(r, d, solution, Correlations);
        end

        if(Selected < d)
            NewSolution = Ripple_Add(r, d, solution, Correlations);
        end 
    end
end

function NewSolution = Ripple_Add(r, d, NHV, Correlations)
% ORIGINAL
%   No modifications performed in this function    
    Len = OneCounter(NHV,length(NHV));
    while(d ~= Len)
                
        Zeropos = ZeroPositionDetector(NHV,length(NHV));
        Onepos = PositionDetector(NHV,length(NHV));
        
        for i=1:length(Onepos)
            SelectedCores(i) = Correlations(Onepos(i));
        end
        
        for i=1:length(Zeropos)
            IgnoredCores(i) = Correlations(Zeropos(i));
        end
        
        CorrelationSelected = sort(SelectedCores,'ascend');
        CorrelationIgnored = sort(IgnoredCores,'descend');
        
        for R = 1:r
            slct = CorrelationIgnored(R);
            for i= 1:length(Correlations)
                if(Correlations(i) == slct)
                    NHV(i) = 1;
                    Len = Len + 1;
                    break;
                end
            end
        end
        
         for R = 1:(r-1)
            rmv = CorrelationSelected(R);
            for i= 1:length(Correlations)
                if(Correlations(i) == rmv)
                    NHV(i) = 0;
                    Len = Len - 1;
                    break;
                end
            end
         end
    end
    NewSolution = NHV;
end

function NewSolution = Ripple_Remove(r, d, NHV, Correlations)
% ORIGINAL
%   No modifications performed in this function      
    Len = OneCounter(NHV,length(NHV));
    while(d ~= Len)
                
        Zeropos = ZeroPositionDetector(NHV,length(NHV));
        Onepos = PositionDetector(NHV,length(NHV));
        
        for i=1:length(Onepos)
            SelectedCores(i) = Correlations(Onepos(i));
        end
        
        for i=1:length(Zeropos)
            IgnoredCores(i) = Correlations(Zeropos(i));
        end
        
        CorrelationSelected = sort(SelectedCores,'ascend');
        CorrelationIgnored = sort(IgnoredCores,'descend');
        
        for R = 1:r
            rmv = CorrelationSelected(R);
            for i= 1:length(Correlations)
                if(Correlations(i) == rmv)
                    NHV(i) = 0;
                    Len = Len - 1;
                    break;
                end
            end
        end
        
        for R = 1:(r-1)
            slct = CorrelationIgnored(R);
            for i= 1:length(Correlations)
                if(Correlations(i) == slct)
                    NHV(i) = 1;
                    Len = Len + 1;
                    break;
                end
            end
        end
    end
    NewSolution = NHV;
end

function position = ZeroPositionDetector(NHV,NVAR)
% ORIGINAL
%   No modifications performed in this function
    p = 1;
    position = [];
    
  for l = 1: NVAR
    if (NHV(l) == 0)
      position(p) = l;
      p = p + 1;
    end
  end
end

function position = PositionDetector(NHV,NVAR)
% ORIGINAL
%   No modifications performed in this function
    p = 1;
    position = [];
    
  for l = 1: NVAR
    if (NHV(l) == 1)
      position(p) = l;
      p = p + 1;
    end
  end
end

function [HM, F] = UpdateHM(NewFit,fitness,HM,HMS,NCHV,NVAR )
% ORIGINAL
%   No modifications performed in this function
    WorstIndex = 1;
    WorstFit=fitness(1);
    for i = 2:HMS
        if( fitness(i) < WorstFit )
            WorstFit = fitness(i);
            WorstIndex = i;
        end
    end

    if( NewFit > WorstFit )
        fitness(WorstIndex)=NewFit;
        for i = 1:NVAR
            HM(WorstIndex,i)=NCHV(i);
        end
    end
    F = fitness;
end