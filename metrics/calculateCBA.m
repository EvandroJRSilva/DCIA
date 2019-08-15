function cba = calculateCBA(numCls, confusionMatrix, sizetestTargets)
% Function to calculate CBA

    % Part of CBA formula
    cba1 = 0;
    
    for i=1:numCls
        % cba1 cases:
        %   Case 1
        %       - Predicted: yes
        %       - Test set: present
        %       - True positive >= 0
        %       - Formula: cba1 + (tpSize/max(size(predictCls, 1), size(realCls, 1)))
        %   Case 2
        %       - Predicted: yes
        %       - Test set: absent
        %       - True positive == 0
        %       - Formula: cba1 + (tpSize/max(size(predictCls, 1), size(realCls, 1)))
        %                  = cba1 + 0
        %   Case 3
        %       - Predicted: no
        %       - Test set: present
        %       - True positive == 0
        %       - Formula: cba1 + (tpSize/max(size(predictCls, 1), size(realCls, 1)))
        %                  = cba1 + 0
        %   Case 4
        %       - Predicted: no
        %       - Test set: absent
        %       - True positive == 0
        %       - Formula: none
        % Conclusion: proceed with the formula if predicted or present.
        
        % If the class is present, i.e., its line is not entirely 0, or if
        % it is predicted, i.e., its column ins not entirely 0.
        if sum(confusionMatrix(i, :)) > 0 || sum(confusionMatrix(:, i)) > 0
            cba1 = cba1 + ...
                confusionMatrix(i, i)/max(sum(confusionMatrix(i, :)), sum(confusionMatrix(:, i)));
        end
    end
    
    cba = cba1/sizetestTargets;
end