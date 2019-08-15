function performance = calculateAUCarea(output, targets)
% Function to calculate AUCarea
    
    [auc, ~] = colAUC(output, targets, 'ROC');
    auc2 = 0; % Numerator part of AUCarea formula
    
    % Calculating performances
    for a=1:size(auc, 1)-1
        auc2 = auc2 + (auc(a)*auc(a+1));
    end
    auc2 = auc2 + (auc(1)*auc(end));
    performance = auc2/size(auc, 1);
end