import sys
import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

def main(csv=None, dataset=None, splitdir=None):
    if csv is None:
        parser = ArgumentParser()
        parser.add_argument('--csv', type=str, help='Path to csv file')
        parser.add_argument('--dataset', type=str, help='Dataset name')
        parser.add_argument('--splitdir', type=str, default=None, help='Path to split directory')
        args = parser.parse_args()
    else:
        args = type('args', (object,), {'csv': csv, 'dataset': dataset, 'splitdir': splitdir})

    df = pd.read_csv(args.csv)
    ids = list(np.load(os.path.join(args.dataset, 'ids.npy')))
    embeddings = np.load(os.path.join(args.dataset, 'embeddings.npy'))

    # Cleans the csv from slides not in DB. Prints the number of slides removed.
    df['is_in_db'] = df.apply(lambda x: x['ID'] in ids, axis=1)
    n_removed = df[df['is_in_db'] == False].shape[0]
    df = df[df['is_in_db']]

    # Reorders X according to the order of the csv.
    Is = [ids.index(i) for i in df['ID'].values]
    X = np.vstack([embeddings[i, :] for i in Is])
    y = df['label'].values


    # Splits the data into 5 test/train splits.
    splits = split_data(X, y, list(df['ID'].values), args.splitdir)

    # Performs 5-fold cross-validation.
    scores = []
    for i, (train, test) in enumerate(splits):
        train_size = len(train)
        test_size = len(test)
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        logreg = LogisticRegression(C=10, max_iter=1000, class_weight='balanced')
        logreg.fit(X_train, y_train)

        if len(np.unique(y)) == 2:
            y_pred = logreg.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)
        else:
            y_pred = logreg.predict_proba(X_test)
            score = roc_auc_score(y_test, y_pred, multi_class="ovr")
        scores.append(score)

    # Nice printing of the results reporting the name of the csv used and the split strategy.
    if args.splitdir is None:
        split_str = 'random'
    else:
        split_str = 'fixed'

    # Prints spaces before and after the results
    print('\n')
    header = '--- Results for {} using {} split: ---'.format(args.csv, split_str)
    print(header)
    if args.splitdir:
        print('Split directory: {}'.format(args.splitdir))

    print('train / test size: {}/{}'.format(train_size, test_size))
    print('Number of slides removed: {}'.format(n_removed))
    print('Mean AUC: {}'.format(np.mean(scores)))
    print('Std AUC: {}'.format(np.std(scores)))
    print('-' * len(header))
    print('\n')

def split_data(X, y, ids, splitdir, n_splits=5):
    """
    Splits the data into 5 test/train splits. Returns a tuple of two lists of indices.
    If splitdir is None, the data is split randomly. Otherwise, the data is split according to the
    split files in splitdir.
    """
    if splitdir is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        splits = skf.split(X, y)
    else:
        splits = get_index_iterator_from_splitdir(splitdir, ids)
    return splits

def get_index_iterator_from_splitdir(splitdir, names):
    """
    return a list of tuples (train, test) where train and test are the indices of the samples in the train and test set
    takes them in the splitdir folder corresponding to the current experiment (must coincide with the good data table)

    args:
        splitdir: path to the folder containing the splits
        prop_train: size of the training set, folder name containing the splits
        names: list of names of the samples

    returns:
        list of tuples (train, test) where train and test are the indices of the samples in the train and test set
    """
    splits = [pd.read_csv(x) for x in glob(f'{splitdir}/split_[0-9].csv')]
    for split in splits:
        train = np.where(np.isin(names, split['train']))[0]
        test = np.where(np.isin(names, split['test']))[0]
        yield train, test

if __name__ == '__main__':
    main()
