import pickle
import numpy as np
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''

        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier = weak_classifier

        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples,n_features = X.shape
        W = np.ones(n_samples) / n_samples
        self.a_list = []
        self.W_list = []
        self.clf_list = []
        self.W_list.append(W)

        for i in range(self.n_weakers_limit):  
            clf = self.weak_classifier(max_depth=6,min_samples_split=10,min_samples_leaf=10,random_state=1010)

            clf.fit(X,y,sample_weight=self.W_list[i])
            ht = clf.predict(X)

            err = np.sum( (ht!=y) * W )
            print( 'base',i,' learner acc:', np.sum(ht==y) / len(y))
            if err > 0.5:
                print('err is bigger than 0.5')
                break
            if err==0:
                print('err is zero !')
                break
            a = 0.5 * np.log((1-err) / err)
            Z = np.sum( W * np.exp(-a*y*ht) )
            W = W * np.exp(-a*y*ht) / Z
            self.a_list.append(a)
            self.W_list.append(W)
            self.clf_list.append(clf)

        pass


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        n_samples = len(X)
        pred = np.zeros((n_samples))
        n_weakers = len(self.clf_list)
        for i in range(n_weakers):
            pred_i = self.clf_list[i].predict(X)

            pred += self.a_list[i] * pred_i
        return pred

        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''

        pred = self.predict_scores(X)
        ret = list(map(lambda x : 1 if x > threshold else -1 ,pred))
        return ret

        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
