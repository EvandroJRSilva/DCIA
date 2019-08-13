function distance = solutionDistance(cent1, cent2)
    % Function to calculate the distance between two sets of centroids for
    % GSA
    newCent1 = reshape(cent1', 1, []);
    newCent2 = reshape(cent2', 1, []);
    
    distance = sqrt(sum((newCent1 - newCent2) .^ 2));
    
end