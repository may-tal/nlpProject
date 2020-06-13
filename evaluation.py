from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self, test, prediction, b):
        self.test = test
        self.prediction = prediction
        self.precision_score = precision_score(test['label'], prediction)
        self.recall_score = recall_score(test['label'], prediction)
        self.f1_score = f1_score(test['label'], prediction)
        self.confusion_matrix = confusion_matrix(test['label'], prediction)
        self.fb_score = fbeta_score(test['label'], prediction, b)
        self.accuracy_score = accuracy_score(test['label'], prediction)

    def get_recall_score(self):
        return self.recall_score

    def get_precision_score(self):
        return self.precision_score

    def get_f1_score(self):
        return self.f1_score

    def get_fb_score(self):
        return self.fb_score

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def show_error(self):
        count = 0
        for row in self.test:
            if row['label'] != self.prediction[count]:
                print("text: " + row['text'])
                print("true label: " + row['label'])
                print("pred label: " + self.prediction[count])
            count += 1

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.test['label'], self.prediction, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.title("ROC curve")
        plt.ylabel("true positive rate")
        plt.xlabel("false positive rate")
        plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        plt.show()



