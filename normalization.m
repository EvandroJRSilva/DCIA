function [normTrainSet, normTestSet] = normalization(trainSet, testSet)
% Function to normalize data set with z-score.
%
% Z-Score is calculated with the following formula:
%   (A - mean(A))/std(A) , where A is an attribute.
%
% As test set needs to remain unknown, each attribute is normalized
% accordingly to mean and std from training set. It is supposed train and
% test set were drawn from the same population.
%
% Each set must be a matrix n x d, where n is the number of instances and d
% is the number of attributes.

    normTrainSet = zeros(size(trainSet, 1), size(trainSet, 2));
    normTestSet = zeros(size(testSet, 1), size(testSet, 2));
    
    for i=1:size(trainSet, 1)
        for j=1:size(trainSet, 2)
            if std(trainSet(:, j)) == 0
                % All values are equal, i.e., there is no z-score
                normTrainSet(i,j) = 0;
            else
                normTrainSet(i,j) = ...
                    (trainSet(i,j) - mean(trainSet(:, j)))/...
                    (std(trainSet(:, j)));
            end
        end
    end

    for i=1:size(testSet, 1)
        for j=1:size(testSet, 2)
            % Test set is normalized accordingly to trainSet, as these
            % data are supposed to be unknown
            if std(trainSet(:, j)) == 0
                % All values are equal, i.e., there is no z-score
                normTestSet(i,j) = 0;
            else
                normTestSet(i,j) = ...
                    (testSet(i,j) - mean(trainSet(:, j)))/...
                    (std(trainSet(:, j)));
            end
        end
    end
end