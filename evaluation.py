from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


class Evaluator:
    """
    this class compute all measures scores for given classifier and plot the errors and the roc curve
    """
    def __init__(self, prediction, test,  b):
        self.test = test
        self.prediction = prediction
        self.precision_score = precision_score(test['label'], prediction, average="macro", pos_label=1)
        self.recall_score = recall_score(test['label'], prediction, average="macro", pos_label=1)
        self.f1_score = f1_score(test['label'], prediction, average="macro", pos_label=1)
        self.confusion_matrix = confusion_matrix(test['label'], prediction)
        self.fb_score = fbeta_score(test['label'], prediction, b, average="macro", pos_label=1)
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
                print("--------------------------------")
                print("text: " + row['text'])
                print("true label: " + str(row['label']))
                print("pred label: " + str(self.prediction[count]))
                print("--------------------------------")
                error_count += 1
            count += 1
        print("found " + str(error_count) + " errors")

    def get_evaluation(self, title, data_name):
        """
        this function return all the scores, print the errors and plot the roc curve
        """
        scores = [self.get_recall_score(), self.get_precision_score(), self.get_accuracy_score(), self.get_f1_score(),
                  self.get_fb_score()]
        # self.plot_confusion_matrix(title + "\n" + data_name)
        return scores

    def plot_confusion_matrix(self, title):
        """
        this function plot the confusion matrix
        """
        class_names = [0, 1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(self.confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix- ' + title, y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

