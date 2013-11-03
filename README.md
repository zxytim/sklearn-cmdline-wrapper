sklearn-cmdline-wrapper
=======================

access sklearn (scikit-learn) machine learning library via command line.

Introduction
------------
### What is sklearn?
__sklearn__ is a python machine learning library comprised of various
machine learning algorithms. See [here](http://scikit-learn.org/stable/) for detail.

### Why command line?
Although you can write a python script every time you like to use functionalities
provided by sklearn, it is still annoying to write duplicated script again and again.
It is both error prone and inefficient, especially when you want to perform comparisons
on different learning algorithms.

This off-the-shell command line tool is here to make your life easier

Prerequisite
------------
Of course You have to install sklearn :) , move steps to the [official website of sklearn](http://scikit-learn.org/stable/)

This script is only tested on version 0.14.1.
For lower versions, something like ```AdaBoostClassifier``` will not work.


Features
--------
- only ONE script, you can copy and paste it as you like.
- only supervised learning tasks are supported currently
- parameters(limited, but sufficient) of model can be passed via command-line,
	you can even make the model utilize mult-cores.
- automatically use sparse matrix if model supports
- libsvm format input

For detailed information: ```./learner.py -h```

Todo (let's make it a better script)
------------------------------------
- compatibility between different versions of sklearn
- support for unsupervised learning tasks
- data visualization for unsupervised learning tasks
- more metrics
- more input file type (like csv, or just space separated columns,
better if there's automatically detection).

Known Issue
-----------
Due to internal data structures transition, this script is not
memory-efficient. It may eat up more memory that you expected.
For this reason, it is not recommend to use this script on "Big Data"

Contact
-------
Xinyu Zhou <zxytim[at]gmail[dot]com>

Help
----
output of ```./learner.py -h```

	usage: learner.py [-h] -t {fit,predict,fitpredict,f,p,fp,doc}
					  [--training-file TRAINING_FILE] [--test-file TEST_FILE]
					  [--model-input MODEL_INPUT] [--model-output MODEL_OUTPUT]
					  [-m {logisticr,knnc,mnb,perceptron,lsvc,lasso,abc,ridge,abr,elasticnet,bnb,knnr,sgdc,etr,rfr,nusvr,gbc,dtc,linearr,svc,rfc,etc,gbr,dtr,svr}]
					  [--prediction-file PREDICTION_FILE]
					  [--model-format {pickle,joblib}] [--show-metrics]
					  [-v [VERBOSE]]
					  [model_options [model_options ...]]

	command line wrapper for some models in scikit-learn

	positional arguments:
	  model_options         additional paramters for specific model of format
							"name:type:val", effective only when training is
							needed. type is either int, float or str, which
							abbreviates as i, f and s.

	optional arguments:
	  -h, --help            show this help message and exit
	  -t {fit,predict,fitpredict,f,p,fp,doc}, --task {fit,predict,fitpredict,f,p,fp,doc}
							task to process, see help for detailed information
	  --training-file TRAINING_FILE
							input: training file, svm format by default
	  --test-file TEST_FILE
							input: test file, svm format by default
	  --model-input MODEL_INPUT
							input: model input file, used in prediction
	  --model-output MODEL_OUTPUT
							output: model output file, used in fitting
	  -m {logisticr,knnc,mnb,perceptron,lsvc,lasso,abc,ridge,abr,elasticnet,bnb,knnr,sgdc,etr,rfr,nusvr,gbc,dtc,linearr,svc,rfc,etc,gbr,dtr,svr}, --model {logisticr,knnc,mnb,perceptron,lsvc,lasso,abc,ridge,abr,elasticnet,bnb,knnr,sgdc,etr,rfr,nusvr,gbc,dtc,linearr,svc,rfc,etc,gbr,dtr,svr}
							model, specified in fitting
	  --prediction-file PREDICTION_FILE
							output: prediction file
	  --model-format {pickle,joblib}
							model format, pickle(default) or joblib
	  --show-metrics        show metric after prediction
	  -v [VERBOSE], --verbose [VERBOSE]
							verbose level, -v <level> or multiple -v's or
							something like -vvv

	task specification:
		task name: fit, f
			required arguments: training_file, model, model_output
			optional arguments: model_options
		task name: predict, p
			required arguments: test_file, model_input, prediction_file
			optional arguments:
		task name: fitpredict, fp
			required arguments: training_file, model, test_file, prediction_file
			optional arguments: model_options, model_output
		task name: doc
			required arguments: model
			optional arguments:
	Notes:
		1. model abbreviation correspondence:
			Abbreviation Model
			abc          AdaBoostClassifier
			abr          AdaBoostRegressor
			bnb          BernoulliNB
			dtc          DecisionTreeClassifier
			dtr          DecisionTreeRegressor
			elasticnet   ElasticNet
			etc          ExtraTreesClassifier
			etr          ExtraTreesRegressor
			gbc          GradientBoostingClassifier
			gbr          GradientBoostingRegressor
			knnc         KNeighborsClassifier
			knnr         KNeighborsRegressor
			lasso        Lasso
			linearr      LinearRegression
			logisticr    LogisticRegression
			lsvc         LinearSVC
			mnb          MultinomialNB
			nusvr        NuSVR
			perceptron   Perceptron
			rfc          RandomForestClassifier
			rfr          RandomForestRegressor
			ridge        Ridge
			sgdc         SGDClassifier
			svc          SVC
			svr          SVR

		2. model compatible with sparse matrix:
			KNeighborsRegressor, SGDClassifier, LinearRegression, LogisticRegression, LinearSVC, KNeighborsClassifier, Ridge, Perceptron, NuSVR, SVR

	Examples:
		1. fit(train) a SVR model with sigmoid kernel:
			./learner.py -t f --training-file training-data --model svr \
					--model-output model.svr kernel:s:sigmoid

		2. predict using precomputed model:
			./learner.py -t p --test-file test --model-input model.svr
				--prediction-file pred-result

		3. fit and predict, model saved, verbose output, and show metrics:
			./learner.py -t fp --training-file training-data --model svr \
				--model-output model.svr --test-file test-data \
				--prediction-file pred-result -v --show-metrics

		4. pass parameters for svc model, specify linear kernel:
			./learner.py --task fp --training-file training-data --model svc \
				--test-file test-data --prediction-file pred-result \
				--show-metrics kernel:s:linear

		5. show documents:
			./learner.py -t doc --model svc
