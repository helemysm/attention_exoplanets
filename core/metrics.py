

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support



def compute_metrics(labels, predicted_result, outputs_list):
"""
This method compute confusion matrix and plot precision recall curve
------
"""    
    labels_ = np.array(labels.to(device).numpy()[0] if isinstance(labels, torch.Tensor) else [i.to(device) for i in labels])

    predicted_result_ = predicted_result.to(device).numpy()[0] if isinstance(predicted_result, torch.Tensor) else [i.to(device) for i in predicted_result]
    outputs_list_ = [list(i[0]) for i in outputs_list]

    print("Confusion Matrix")
    print(confusion_matrix(labels_, predicted_result_))

    print(classification_report(labels_,predicted_result_))
    
    pos_probs = np.array(outputs_list_)[:, 1]
    pos_probs = np.array(outputs_list_)
    
    precision, recall, _ = precision_recall_curve(labels_, pos_probs)
    
    plt.plot(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
