function [gmean, gmovo] = calculateGM(numCls, sensitivities, sizeTestTargets)
% Function to calculate GMean with production and Average GMean
    

    % Calculating GMean with only the non NaN values, i.e., for the present
    % classes
    senses = sensitivities(~isnan(sensitivities));
    gmean = prod(senses)^(1/sizeTestTargets);
    
    % G-mean OVO
    gs = zeros(1, nchoosek(numCls, 2));
    
    count = 1;
    for i=1:numCls
        for j=1:numCls
            if isnan(sensitivities(i)) || isnan(sensitivities(j))
                gs(count) = NaN;
                count = count+1;
            else
                gs(count) = (sensitivities(i)*sensitivities(j))^(0.5);
                count = count+1;
            end
        end
    end
    gmovo = nanmean(gs);
end