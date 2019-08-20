# DCIA
Dynamic Centroid Insertion and Adjustment (DCIA) is a Prototype Generation classification model. In this source code all its evolutions are presented, which begins with DCIAGSA (DCIA with Gravity Search Algorithm) and ends with three major algorithms: DCIASGSA.M5, DCIAPSO.M10 and DCIACSO.

The final algorithm consists of two main phases: (1) preprocessing and (2) processing. In the first phase data are normalized and attributes are selected. The second phase consists of three steps: (1) initialization, (2) adjustment and (3) addition of new prototypes.

Normalization of data is carried out with z-score, while attribute selection is accomplished with Competitive Swarm Optimizer (CSO). The second phase is run with any search algorithm. With DCIA some were tested: GSA, Simple GSA (SGSA), PSO and CSO. Each pool solution corresponds to a set of prototypes. During the initialization the first solution is the set with each class centroid. All other solutions are sets of one random instance from each class as prototypes. The adjustment is performed with the search algorithm, which tries to find a better positioning for the current prototypes. The last step, i.e., addition of new prototypes, tries to insert new prototypes for performance improvement. The second and third steps are repeated until a stop criterion is reached.
	  
## Files

 - **All DBs:** A folder with all data sets used in the experiment. Each data set consists in a structure with two fields: (1) a matrix *n x d* of features, in which *n* is the number of observations and *d* is the number of dimensions, or attributes; (2) a column vector *n x 1* with the class labels of each instance. Some data sets are different versions of the same data set, e.g., *amcs_abalone* and *uci_abalone*. These verions came after different preprocessing procedure, e.g., missing values treatment.
 - **CSO**
	 - **DCIACSO.m**: DCIACSO function file. The three steps of second phase;
	 - **Solution.m**: A class file. Each object of this file is a solution in CSO algorithm;
	 - **featureSelection.m**: Feature selection algorithm function;
	 - **main.m**: Main function, which loads data set and calls functions to run both phases od DCIACSO.
- **GSA**
	- **m1**
		- *main.m:* Main function for the modification one;
	- **m2**
		- *main.m:* Main function for the modification two;
	- **original**
		- *main.m:* Main function for the original DCIAGSA algorithm;
	- **DCIAGSA.m:** DCIAGSA function file;
	- **Solution.m:** A class file. Each object of this file is a solution in GSA algorithm.
-  **PSO**
	- **m1**
		- *DCIAPSO.m:* DCIAPSO function file adapted for modification one;
		- *main.m:* Main function for modification one;
	- **m2**
		- *featureSelection.m:* Feature selection function for modification two;
		- *main.m:* Main function for modification two;
	- **m3**
		- *featureSelection.m:* Feature selection function for modification three;
		- *main.m:* Main function for modification three;
	- **m4**
		- featureSelection.m:* Feature selection function for modification four;
		- *main.m:* Main function for modification four;
	- **m5**
		- featureSelection.m:* Feature selection function for modification five;
		- *main.m:* Main function for modification five;
	- **m6**
		- featureSelection.m:* Feature selection function for modification six;
		- *main.m:* Main function for modification six;
	- **m7**
		- featureSelection.m:* Feature selection function for modification seven;
		- *main.m:* Main function for modification seven;
	- **m8**
		- featureSelection.m:* Feature selection function for modification eight;
		- *main.m:* Main function for modification eight;
	- **m9**
		- featureSelection.m:* Feature selection function for modification nine;
		- *main.m:* Main function for modification nine;
	- **m10**
		- featureSelection.m:* Feature selection function for modification ten;
		- *main.m:* Main function for modification ten;
	- **m11**
		- featureSelection.m:* Feature selection function for modification eleven;
		- *main.m:* Main function for modification eleven;
	- **original**
		- *main.m:* Main function for the original DCIAPSO algorithm.
	- **DCIAPSO.m:** DCIAPSO function file. The three steps of second phase.
	- **Solution.m:** A class file. Each object of this file is a solution in PSO algorithm.
- **SGSA**
	- **m1**
		- *main.m:* Main function for modification one;
	- **m2**
		- *main.m:* Main function for modification two;
	- **m3**
		- *main.m:* Main function for modification three;
	- **m4**
		- *main.m:* Main function for modification four;
	- **m5**
		- featureSelection.m:* Feature selection function for modification five;
		- *main.m:* Main function for modification five;
	- **original**
		- *main.m:* Main function for the original DCIASGSA algorithm;
	- **DCIASGSA.m:** DCIAPSO function file;
	- **Solution.m:** A class file. Each object of this file is a solution in SGSA algorithm;
- **metrics:** Folder with functions to calculate multiclass performance with some metrics. They are: AUCarea, CBA, F-measure, G-mean, Kappa and MAUC. The other files calculate some data to feed the metrics functions. They are: the confusion matrix, the true positive rate for each class and AUC for each pair of class;
- **About Versions.txt:** Text file that summarizes all DCIA versions;
- **Results.m:** Class file. An object of this class is able to hold the results for all metrics and for each cross-validation fold;
- **distance.m:** Function to calculate Euclidean distance among vectors;
- **dobscv.m:** Function to divide data with DOB-SCV;
- **getDataSets.m:** Function to load, by name, data sets from ALL DBs folder;
- **kfoldIndices.m:** Function to divide data with k-fold cross-validation;
- **normalization.m:** Function to normalize data with z-score;
- **solutionDistance.m:** Function to calculate distance among Solution objects.

## How to use

First you must choose which algorithm you want to run. After you choose you have to put the corresponding main file, and the other files from the same folder, in the Matlab path and run it. It is important to remove the other main files from the path. In this way the algorithm will call the correct functions, e.g., the correct feature selection algorithm.

Inside all main files there are some *path* variables, which are used to set the folder where data sets are and the folder where results will be saved. The *mainPath* variable is the path of Matlab itself, commonly *C:\Users\My Documents\Matlab\\*. The *algPath* is the algorithm name, e.g., *DCIA\PSO\\*. The *version* refers to the modification and may be *mX\\* or *original\\*.  Finaly, the *resultsFolder* is where the results will be saved. These variables are optional to use in the sense you may load data sets from other folder and save anywhere you want.

Whenever you run a main file it will load a set of data sets, normalize data, select features, run DCIA and save the results. You just need to wait it to finish the process and then you may see the results in the results folder. These codes are compatible with Matlab2018a.
