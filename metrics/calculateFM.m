function fm = calculateFM(numCls, confusionMatrix, sensitivities)
% Function to calculate F-Measure
    
    fs = zeros(1, numCls); %f-measure for each class
    
    for i=1:numCls
        if sum(confusionMatrix(i, :)) == 0
            fs(i) = NaN; %f-measure needs TPR
        elseif sum(confusionMatrix(:, i)) == 0
            fs(i) = NaN; %f-measure needs Precision
        else
            precision = confusionMatrix(i, i)/sum(confusionMatrix(:, i));
            fs(i) = 2*((sensitivities(i)*precision)/(sensitivities(i)+precision));
        end
    end
    
    % F-measure
    fm = nanmean(fs);
end