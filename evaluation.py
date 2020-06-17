from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix, auc


class Evaluator:
    """
    this class compute all measures scores for given classifier and plot the errors and the roc curve
    """
    def __init__(self, prediction, test,  b):
        self.test = test
        self.prediction = prediction
        self.precision_score = precision_score(test['label'], prediction, average="binary")
        self.recall_score = recall_score(test['label'], prediction, average="binary")
        self.f1_score = f1_score(test['label'], prediction, average="binary")
        self.confusion_matrix = confusion_matrix(test['label'], prediction)
        self.fb_score = fbeta_score(test['label'], prediction, b, average="binary")
        self.accuracy_score = accuracy_score(test['label'], prediction)
        self.beta = b

    def get_accuracy_score(self):
        """
        this function return the accuracy score
        """
        return float("%0.4f" % self.accuracy_score)

    def get_recall_score(self):
        """
        this function return the recall score
        """
        return float("%0.4f" % self.recall_score)

    def get_precision_score(self):
        """
        this function return the precision score
        """
        return float("%0.4f" % self.precision_score)

    def get_f1_score(self):
        """
        this function return F1 score
        """
        return float("%0.4f" % self.f1_score)

    def get_fb_score(self):
        """
        this function return Fb score
        """
        return float("%0.4f" % self.fb_score)

    def get_confusion_matrix(self):
        """
        this function return the confusion matrix
        """
        return self.confusion_matrix

    def show_error(self):
        """
        this function print the amount of errors and print them
        """
        count = 0
        error_count = 0
        for i, row in self.test.iterrows():
            if row['label'] != self.prediction[count]:
                print("text: " + row['text'])
                print("true label: " + str(row['label']))
                print("pred label: " + str(self.prediction[count]))
                error_count += 1
            count += 1
        print("found " + str(error_count) + " errors")

    def get_evaluation(self):
        """
        this function return all the scores, print the errors and plot the roc curve
        """
        scores = [self.get_recall_score(), self.get_precision_score(), self.get_accuracy_score(), self.get_f1_score(),
                  self.get_fb_score()]
        # self.show_error()
        return scores
