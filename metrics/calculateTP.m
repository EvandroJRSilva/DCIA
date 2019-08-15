function [otp, sensitivities] = calculateTP(numCls, confusionMatrix)
% Function to calculate the sum of all True Positives, as well as True 
% Positive rate of each class. In cases where a class is not present in
% test set, its True Positive rate will be NaN, as a result of dividing by
% zero.

    sensitivities = zeros(1, numCls); otp = 0;
    
    for i=1:numCls
        % If a line is entirely 0, this means that this class is not
        % present in test set. Its True Positive is NaN, as a result of
        % 0/0.
        if sum(confusionMatrix(i, :)) == 0
            sensitivities(i) = NaN;
        else
            sensitivities(i) = confusionMatrix(i, i)/sum(confusionMatrix(i, :));
            otp = otp + confusionMatrix(i, i);
        end
    end
end