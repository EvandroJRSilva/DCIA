function confusionMatrix = generateConfusionMatrix(output, targets, numCls)
% Function to generate a confusion matrix. It is allowed absence of some 
% classes in the test set. In this case the correspondent line will
% continue with 0s in each column.
    
    confusionMatrix = zeros(numCls);
    
    for i=1:size(output, 1)
        if output(i) == targets(i)
            % Increasing confusionMatrix(i, i)
            confusionMatrix(targets(i), targets(i)) = ...
                confusionMatrix(targets(i), targets(i)) + 1;
        else
            % Increasing confusionMatrix(i, j). Instance from referred
            % class testTargets(i) is misclassified as output(i), or j.
            confusionMatrix(targets(i), output(i)) = ...
                confusionMatrix(targets(i), output(i)) + 1;
        end
    end
end