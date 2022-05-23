# Advanced ML course projects

Projects for successful completion of the course 'Advanced Machine Learning' led by Prof. Joachim Buhmann at ETH Zurich, winter semester 2020. Hints were given in the tutorials and the code was fully implemented by students. Projects were designed to be achievable with small computational resources and relying on standard ML packages such as `sklearn`. The use of pre-trained neural networks and "fancy" models was explicitly discouraged. Evaluation was performed on unknown test data with public and private leaderboards (Kaggle-style) and performance graded according to the leaderboard ranks.

## Project 1
Task: Predict a person’s age from pre-processed MRI data. <br>
Challenges: data was corrupted to include irrelevant features, outliers and perturbations. Evaluation metric: R^2 coefficient. <br>
We implemented two models either with LightGBM or Gradient Boosting. 

## Project 2
Task: Multi-class classification of diseases from pre-processed image data. <br>
Challenges: strong class imbalance. <br>
Evaluation metric: balanced multiclass accuracy. <br>
After trying different classifiers and methods it turned out that a standard balanced SVM Classifier performed best.

## Project 3
Task: Multi-class classification of ECG time series data. The dataset was featured in the PhysioNet 2017 challenge [[1]](https://ieeexplore.ieee.org/abstract/document/8331486). <br>
Challenges: Domain-specific manual feature extraction e.g. heartbeat frequency, amplitude, interval segment durations etc. <br>
Evaluation metric: F1-Score. <br>
The finally selected model by a team colleague included outlier detection with IsolationForest and classification with GradientBoostingClassifier.

## Project 4
Task: Automated sleep scoring (multi-class classification) from EEG & EMG time series data for brain-state analysis. <br>
Challenges: Severe class imbalance, temporal consistency, inter-subject variability, domain-specific manual feature extraction e.g. frequency bands, fourier signal decomposition, power spectrum domain. <br>
Evaluation metric: Balanced accuracy score. <br>
We spent a lot of time on good quality feature extraction and achieved a good score using a simple balanced SVM Classifier.
