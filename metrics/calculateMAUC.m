function mauc = calculateMAUC(output, targets, nClass)
% Function for calculating MAUC metric
    
    [auc, ~] = colAUC(output, targets, 'ROC');
    mauc = (2/(nClass*(nClass - 1)))*sum(auc); 
end