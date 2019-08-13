classdef Results
    %RESULTS is a class for recording outputs of experiments
    %   Detailed explanation goes here
    
    properties
        MAUC                %Performance as MAUC
        GMean               %Performance as G-mean
        GMOVO               %Performance as Average G-mean (OVO)
        Accuracy            %Performance as Accuracy
        Kappa               %Performance as Kohen's Kappa
        CBA                 %Performance as Class Balance Accuracy
        Sensitivities       %Set of Classes Sensitivies, i.e., true positivies for each class
        AUCA                %Performance as AUC Area
        FM                  %F-Measure. In multiclass case it is mean of fms
        Dim                 %Number of Dimensions after Feature Selection
        NumProt             %Number of Prototypes
    end
    
    methods
    end
    
end

