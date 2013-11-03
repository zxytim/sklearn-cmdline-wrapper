#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: learner.py
# $Date: Sun Nov 03 19:24:21 2013 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
#
# TODO:
#   generalize metrics, see:
#       http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

import sklearn
from sklearn.neighbors import * # KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import * # SVC, SVR, NuSVR, LinearSVC

# LinearRegression, LogisticRegression
# SGDClassifier, Perceptron, Ridge, Lasso, ElasticNet
from sklearn.linear_model import *

# MultinomialNB, BernoulliNB
from sklearn.naive_bayes import *

from sklearn.tree import * # DecisionTreeClassifier, DecisionTreeRegressor

from scipy import sparse

# RandomForestClassifier, RandomForestRegressor
# ExtraTreesClassifier, ExtraTreesRegressor
# AdaBoostClassifier, AdaBoostRegressor
# GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import *

# save model
from sklearn.externals import joblib
import pickle

# metrics
from sklearn import metrics

import numpy as np
from collections import defaultdict
import argparse
import sys
import types


models = {
        'linearr': LinearRegression,
        'logisticr': LogisticRegression,
        'knnc': KNeighborsClassifier,
        'knnr': KNeighborsRegressor,
        'svc': SVC,
        'svr': SVR,
        'nusvr': NuSVR,
        'lsvc': LinearSVC,
        'sgdc': SGDClassifier,
        'dtc': DecisionTreeClassifier,
        'dtr': DecisionTreeRegressor,
        'rfc': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'etc': ExtraTreesClassifier,
        'etr': ExtraTreesRegressor,
        'abc': AdaBoostClassifier,
        'abr': AdaBoostRegressor,
        'gbc': GradientBoostingClassifier,
        'gbr': GradientBoostingRegressor,
        'perceptron': Perceptron,
        'ridge': Ridge,
        'lasso': Lasso,
        'elasticnet': ElasticNet,
        'mnb': MultinomialNB,
        'bnb': BernoulliNB,
        }

sparse_models = set([
    SVR,
    NuSVR,
    LinearSVC,
    KNeighborsClassifier,
    KNeighborsRegressor,
    SGDClassifier,
    Perceptron,
    Ridge,
    LogisticRegression,
    LinearRegression,
    ])

args = []

# table is a 2-D string list
# return a printed string
def format_table(table):
    if len(table) == 0:
        return ''
    col_length = defaultdict(int)
    for row in table:
        for ind, item in enumerate(row):
            col_length[ind] = max(col_length[ind], len(item))

    # WARNING: low efficiency, use string buffer instead
    ret = ''
    for row in table:
        for ind, item in enumerate(row):
            fmtstr = '{{:<{}}}' . format(col_length[ind])
            ret += fmtstr.format(item) + " "
        ret += "\n"
    return ret

def get_model_abbr_help():
    lines = format_table([['Abbreviation', 'Model']] + map(lambda item: [item[0], item[1].__name__], \
        sorted(models.items()))).split('\n')
    return "\n".join(map(lambda x: ' ' * 8 + x, lines))

class VerboseAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # print 'values: {v!r}'.format(v=values)
        if values==None:
            values='1'
        try:
            values=int(values)
        except ValueError:
            values=values.count('v')+1
        setattr(args, self.dest, values)

def get_args():
    description = 'command line wrapper for some models in scikit-learn'

    tasks = ['fit', 'predict', 'fitpredict',
            'f', 'p', 'fp',
            'doc']

    # (task_names, required_params, optional_params)
    task_arg_setting = [
            (['fit', 'f'],
                ['training_file', 'model', 'model_output'],
                ['model_options']),
            (['predict', 'p'],
                ['test_file', 'model_input', 'prediction_file'],
                []),
            (['fitpredict', 'fp'],
                ['training_file', 'model', 'test_file', 'prediction_file'],
                ['model_options', 'model_output']),
            (['doc'],
                ['model'],
                [])
            ]

    epilog = "task specification:\n{}" . format(
            "\n" . join([
                '    task name: {}\n        required arguments: {}\n        optional arguments: {}' . format(
                    *map(lambda item: ", " . join(item), setting)) \
                            for setting in task_arg_setting]))
    epilog += "\n"

    epilog += "Notes:\n"
    epilog += "    1. model abbreviation correspondence:\n"
    epilog += get_model_abbr_help() + '\n'
    epilog += "    2. model compatible with sparse matrix:\n"
    epilog += ' ' * 8 + ", " . join(map(lambda x: x.__name__, sparse_models))
    epilog += '\n'
    epilog += '\n'

    epilog += 'Examples:\n'
    epilog += """\
    1. fit(train) a SVR model with sigmoid kernel:
        ./learner.py -t f --training-file training-data --model svr \\
                --model-output model.svr kernel:s:sigmoid

    2. predict using precomputed model:
        ./learner.py -t p --test-file test --model-input model.svr
            --prediction-file pred-result

    3. fit and predict, model saved, verbose output, and show metrics:
        ./learner.py -t fp --training-file training-data --model svr \\
            --model-output model.svr --test-file test-data \\
            --prediction-file pred-result -v --show-metrics

    4. pass parameters for svc model, specify linear kernel:
        ./learner.py --task fp --training-file training-data --model svc \\
            --test-file test-data --prediction-file pred-result \\
            --show-metrics kernel:s:linear

    5. show documents:
        ./learner.py -t doc --model svc
"""


    parser = argparse.ArgumentParser(
            description = description, epilog = epilog,
            formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--task',
            choices = tasks,
            help = 'task to process, see help for detailed information',
            required = True)
    parser.add_argument('--training-file',
            help = 'input: training file, svm format by default')
    parser.add_argument('--test-file',
            help = 'input: test file, svm format by default')
    parser.add_argument('--model-input',
            help = 'input: model input file, used in prediction')
    parser.add_argument('--model-output',
            help = 'output: model output file, used in fitting')
    parser.add_argument('-m', '--model',
            help = 'model, specified in fitting',
            choices = models)
    parser.add_argument('--prediction-file',
            help = 'output: prediction file')

    parser.add_argument('--model-format',
            choices = ['pickle', 'joblib'],
            default = 'pickle',
            help = 'model format, pickle(default) or joblib')

    parser.add_argument('--show-metrics',
            action = 'store_true',
            help = 'show metric after prediction')

    parser.add_argument('-v', '--verbose',
            help = 'verbose level, -v <level> or multiple -v\'s or something like -vvv',
            nargs = '?',
            default = 0,
            action = VerboseAction)

    parser.add_argument('model_options',
            nargs = '*',
            help = """\
additional paramters for specific model of format "name:type:val", \
effective only when training is needed. type is either int, float or str, \
which abbreviates as i, f and s.""")

    args = parser.parse_args()

    model_options = dict()
    for opt in args.model_options:
        opt = opt.split(':')
        assert len(opt) == 3, 'model option format error'
        key, t, val = opt
        if t == 'i':
            t = 'int'
        elif t == 'f':
            t = 'float'
        elif t == 's':
            t = 'str'

        model_options[key] = eval(t)(val)

    args.model_options = model_options

    # make task name a full name
    for setting in task_arg_setting:
        if args.task in setting[0]:
            args.task = setting[0][0]

    # check whether paramters for specific task is met
    def check_params(task, argnames):
        if args.task in task:
            for name in argnames:
                if not(name in args.__dict__ and args.__dict__[name]):
                    info = 'argument `{}\' must present in `{}\' task' .\
                            format("--" + name.replace('_', '-'), task[0])
                    raise Exception(info)

    try:
        for setting in task_arg_setting:
            check_params(setting[0], setting[1])
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)

    # add a verbose print method to args
    def verbose_print(self, msg, vb = 1): # vb = verbose_level
        if vb <= self.verbose:
            print(msg)

    args.vprint = types.MethodType(verbose_print, args, args.__class__)


    return args

def read_svmformat_data(fname):
    x, y = [], []
    with open(fname) as f:
        lines = f.readlines()
        nlines = len(lines)
        for cnt, line in enumerate(lines):
            line = line.rstrip().split()
            y.append(float(line[0]))
            xtmp = []
            for i in xrange(1, len(line)):
                ind, val = line[i].split(':')
                xtmp.append((int(ind), float(val)))
            x.append(xtmp)
    return x, y

def write_labels(fname, y_pred):
    count_types = defaultdict(int)
    for y in y_pred:
        count_types[type(y)] += 1
    most_prevalent_type = sorted(map(lambda x: (x[1], x[0]), count_types.iteritems()))[0][1]
    if most_prevalent_type == float:
        typefmt = '{:f}'
    else:
        typefmt = '{}'

    with open(fname, 'w') as fout:
        for y in y_pred:
            fout.write(typefmt.format(y) + '\n')


def get_model(args):
    model = models[args.model](**args.model_options)
    return model

def get_dim(X):
    dim = -1
    for x in X:
        for ind, val in x:
            if ind > dim:
                dim = ind
    return dim + 1

def sparse2dense(X, dim):
    ret = np.zeros((len(X), dim))
    for cnt, x in enumerate(X):
        for ind, val in x:
            ret[cnt][ind] = val
    return ret

def sparse2csr_matrix(X, dim):
    data, row, col = [], [], []
    for ind, x in enumerate(X):
        t = list(zip(*x))
        row.extend([ind] * len(t[0]))
        col.extend(t[0])
        data.extend(t[1])
    sparse.csr_matrix((data, (row, col)), shape = (len(X), dim))

def preprocess_data(model, Xs):
    arg_list = True
    if len(Xs) == 0:
        return []
    if type(Xs[0][0]) != list:
        arg_list = False
        Xs = [Xs]
    dim = max(map(get_dim, Xs))
    if model in sparse_models:
        ret = map(lambda X: sparse2csr_matrix(X, dim), Xs)
    else:
        ret = map(lambda X: sparse2dense(X, dim), Xs)
    if not arg_list:
        assert len(ret) == 1
        ret = ret[0]
    return ret

def save_model(fname, model):
    if args.model_format == 'pickle':
        fd = open(fname, 'wb')
        pickle.dump(model, fd)
        fd.close()
    else:
        joblib.dump(model, fname)

def load_model(fname):
    if args.model_format == 'pickle':
        fd = open(fname, 'rb')
        model = pickle.load(fd)
        fd.close()
        return model
    else:
        return joblib.load(fname)

# TODO: generalize
def show_metrics(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred))


def task_fit(args):
    model = get_model(args)

    args.vprint('reading training file {} ...' . format(args.training_file))
    X_train, y_train = read_svmformat_data(args.training_file)

    args.vprint('preprocessing training data ...')
    X_train = preprocess_data(model, X_train)

    args.vprint('training model {} ...' . format(model.__class__.__name__))
    model.fit(X_train, y_train)

    args.vprint('saving model ...')
    save_model(args.model_output, model)

def task_predict(args):
    model = load_model(args.model_input)

    args.vprint('reading test file {} ...' . format(args.test_file))
    X_test, y_test = read_svmformat_data(args.test_file)

    args.vprint('preprocessing test data ...')
    X_test = preprocess_data(model, X_test)

    args.vprint('predicting ...')
    y_pred = model.predict(X_test)

    args.vprint('writing predictions ...')
    write_labels(args.prediction_file, y_pred)

    if args.show_metrics:
        show_metrics(y_test, y_pred)

def task_fitpredict(args):
    model = get_model(args)

    args.vprint('reading training file {} ...' . format(args.training_file))
    X_train, y_train = read_svmformat_data(args.training_file)

    args.vprint('reading test file {} ...' . format(args.test_file))
    X_test, y_test = read_svmformat_data(args.test_file)

    args.vprint('preprocessing training and test data ...')
    X_train, X_test = preprocess_data(model, [X_train, X_test])

    args.vprint('training model {} ...' . format(model.__class__.__name__))
    model.fit(X_train, y_train)

    args.vprint('predicting ...')
    y_pred = model.predict(X_test)

    if args.model_output:
        args.vprint('saving model ...')
        save_model(args.model_output, model)

    args.vprint('writing predictions ...')
    write_labels(args.prediction_file, y_pred)

    if args.show_metrics:
        show_metrics(y_test, y_pred)


def main():
    global args
    args = get_args()

    if args.task == 'doc':
        print(models[args.model].__doc__)
    else:
        task_worker = dict(
                fit = task_fit,
                predict = task_predict,
                fitpredict = task_fitpredict)
        task_worker[args.task](args)

if __name__ == '__main__':
    main()

# vim: foldmethod=marker
