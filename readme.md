# Machine Learning A to Z (By Kiril and Hadelin from Udemy)
## This file will contain all the notebooks I will be creating while learning Machine Learning
### The notes are prepared by [Samrat Mitra](https://github.com/lionelsamrat10)

## What is Machine Learning ?

<p>
  <b>Machine learning</b> is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
</p>
<p>
The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.
</p>

## Types of Machine Learning

<p>
  <b>Supervised machine learning algorithms</b> can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able to provide targets for any new input after sufficient training. The learning algorithm can also compare its output with the correct, intended output and find errors in order to modify the model accordingly.
</p>
<p>
In contrast, <b>unsupervised machine learning algorithms</b> are used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. The system doesn’t figure out the right output, but it explores the data and can draw inferences from datasets to describe hidden structures from unlabeled data.
</p>
<p>
  <b>Semi-supervised machine learning algorithms</b> fall somewhere in between supervised and unsupervised learning, since they use both labeled and unlabeled data for training typically a small amount of labeled data and a large amount of unlabeled data. The systems that use this method are able to considerably improve learning accuracy. Usually, semi-supervised learning is chosen when the acquired labeled data requires skilled and relevant resources in order to train it / learn from it. Otherwise, acquiring unlabeled data generally doesn’t require additional resources.
</p>
<p>
  <b>Reinforcement machine learning algorithms</b> is a learning method that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best; this is known as the reinforcement signal.
</p>

## 1. Data Preprocessing
### Data Preprocessing Template ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Data%20Preprocessing/data_preprocessing_tools_samrat.ipynb)
## 2. Regression (Supervised Learning)
<p>
Regression analysis is a statistical method to model the relationship between a dependent (target) and independent (predictor) variables with one or more independent variables. More specifically, Regression analysis helps us to understand how the value of the dependent variable is changing corresponding to an independent variable when other independent variables are held fixed. It predicts continuous/real values such as temperature, age, salary, price, etc.
</p>
<p>
Regression is a supervised learning technique which helps in finding the correlation between variables and enables us to predict the continuous output variable based on the one 
or more predictor variables. It is mainly used for prediction, forecasting, time series modeling, and determining the causal-effect relationship between variables.
</p>
<p>  In Regression, we plot a graph between the variables which best fits the given datapoints, using this plot, the machine learning model can make predictions about the data. In simple words, "Regression shows a line or curve that passes through all the datapoints on target-predictor graph in such a way that the vertical distance between the datapoints and the regression line is minimum." The distance between datapoints and line tells whether a model has captured a strong relationship or not.
</p>
<p>
<b>Some examples of regression can be as: Prediction of rain using temperature and other factors, Determining Market trends, Prediction of road accidents due to rash driving.</b>
</p>

### 2 a. Simple Linear Regression ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Regression/Simple%20Linear%20Regression/simple_linear_regression_samrat.ipynb)
### 2 b. Multiple Linear Regression ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Regression/Multiple%20Linear%20Regression/multiple_linear_regression_samrat.ipynb)
### 2 c. Polynomial Regression (Non Linear) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Regression/Polynomial%20Regression/polynomial_regression_samrat.ipynb)  
### 2 d. Support Vector Regression (SVR) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Regression/SVR%20(Support%20Vector%20Regression)/support_vector_regression_samrat.ipynb)
### 2 e. Decision Tree Regression ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Regression/Decision%20Tree%20Regression/decision_tree_regression_samrat.ipynb)
### 2 f. Random Forest Regression ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Regression/Random%20Forest%20Regression/random_forest_regression_samrat.ipynb)

## 3. Classification (Supervised Learning)

<p><b>The Classification algorithm</b> is a Supervised Learning technique that is used to identify the category of new observations on the basis of training data. In Classification, a program learns from the given dataset or observations and then classifies new observation into a number of classes or groups. Such as, Yes or No, 0 or 1, Spam or Not Spam, cat or dog, etc. Classes can be called as targets/labels or categories.</p>
<p>Unlike regression, the output variable of Classification is a category, not a value, such as "Green or Blue", "fruit or animal", etc. Since the Classification algorithm is a Supervised learning technique, hence it takes labeled input data, which means it contains input with the corresponding output.</p>
<p>In classification algorithm, a discrete output function(y) is mapped to input variable(x).</p>
<p>The best example of an ML classification algorithm is <b>Email Spam Detector.</b></p>
<p>The main goal of the Classification algorithm is to identify the category of a given dataset, and these algorithms are mainly used to predict the output for the categorical data.</p>
<p>Classification algorithms can be better understood using the below diagram. In the below diagram, there are two classes, class A and Class B. These classes have features that are similar to each other and dissimilar to other classes.</p>

### 3 a. Logistic Regression ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Classification/Logistic%20Regression/logistic_regression_samrat.ipynb)
### 3 b. K Nearest Neighbor Classifier (K-NN) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Classification/K%20Nearest%20Neighbor%20Classifier/k_nearest_neighbors_samrat.ipynb)
### 3 c. Support Vector Machine Classifier (SVM) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Classification/Support%20Vector%20Machine(SVM)/support_vector_machine_samrat.ipynb)
### 3 d. Kernel SVM ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/tree/main/Classification/Kernel%20SVM)
### 3 e. Naive Bayes' Classification ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Classification/Naive%20Bayes%20Classification/naive_bayes_samrat.ipynb)
### 3 f. Decision Tree Classification ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Classification/Decision%20Tree%20Classifier/decision_tree_classification_samrat.ipynb)
### 3 g. Random Forest Classification ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Classification/Decision%20Tree%20Classifier/decision_tree_classification_samrat.ipynb)

## 4. Clustering (Unsupervised Learning)

<p><b>Clustering</b> or cluster analysis is a machine learning technique, which groups the unlabelled dataset. It can be defined as "A way of grouping the data points into different clusters, consisting of similar data points. The objects with the possible similarities remain in a group that has less or no similarities with another group."</p>
<p>t does it by finding some similar patterns in the unlabelled dataset such as shape, size, color, behavior, etc., and divides them as per the presence and absence of those similar patterns.</p>
<p>It is an unsupervised learning
method, hence no supervision is provided to the algorithm, and it deals with the unlabeled dataset.</p>
<p>After applying this clustering technique, each cluster or group is provided with a cluster-ID. ML system can use this id to simplify the processing of large and complex datasets.</p>
<p>The clustering technique can be widely used in various tasks. Some most common uses of this technique are:
<b>
Market Segmentation,
Statistical data analysis,
Social network analysis,
Image segmentation,
Anomaly detection, etc.
</b>
</p>

### 4 a. KMeans Clustering ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Clustering/K%20Means%20Clustering/k_means_clustering_samrat.ipynb)
### 4 b. Hierarchical Clustering ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Clustering/Hierarchical%20Clustering/hierarchical_clustering_samrat.ipynb)

## 5. Association Rule Learning
### 5 a. Apriori ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Association%20Rule%20Learning/Apriori/apriori_samrat.ipynb)
### 5 b. Eclat ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Association%20Rule%20Learning/Eclat/eclat_samrat.ipynb)

## 6. Reinforcement Learning
### 6 a. Upper Confidence Bound (UCB) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Reinforcement%20Learning/Upper%20Confidence%20Bound%20(UCB)/upper_confidence_bound_samrat.ipynb)
### 6 b (Part-01). Thompson Sampling for 10000 Rounds ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Reinforcement%20Learning/Thompson%20Sampling/Copy_of_thompson_sampling_samrat_for_10000_rounds.ipynb)
### 6 b (Part-02). Thompson Sampling for 500 Rounds (Here it performs better than the UCB Algorithm) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Reinforcement%20Learning/Thompson%20Sampling/thompson_sampling_samrat_for_500_rounds.ipynb )

## 7. Natural Language Processing (NLP)
### 7 a. Sentiment Analysis ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Natural%20Language%20Processing(NLP)/Sentiment%20Analysis%20of%20Restaurant%20Reviews/natural_language_processing_samrat.ipynb)

## 8. Deep Learning
### 8 a. Artificial Neural Network (ANN) ✅ [Google Colab File Available here](https://github.com/lionelsamrat10/machine-learning-a-to-z/blob/main/Deep%20Learning/Artificial%20Neural%20Network%20(ANN)/ANN%20For%20Classification/artificial_neural_network_for_classification_samrat.ipynb)
