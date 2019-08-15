function kappa = calculateKappa(confusionMatrix, numCls, otp, sizetestSet)
% Function to calculate Kappa metric.
    
    % Part of kappa metric, calculated separately
    kappa2 = 0;
    
    for i=1:numCls
        kappa2 = kappa2 + (sum(confusionMatrix(:, i)) * sum(confusionMatrix(i, :)));
    end
    
    kappa = ((otp*sizetestSet) - kappa2)/((sizetestSet^2) - kappa2);
end