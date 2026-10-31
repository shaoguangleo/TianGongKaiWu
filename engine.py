"""训练、评估和预测共用的核心逻辑。"""
import numpy as np
import torch


def classification_metrics(labels, predictions, num_classes):
    # 第1段：混淆矩阵行是真值、列是预测；缺失类F1按0报告并保留support。
    cm = np.zeros((num_classes,num_classes), dtype=int)
    for a,b in zip(labels,predictions):
        cm[a,b] += 1
    tp = cm.diagonal()
    precision = np.divide(tp,cm.sum(0),out=np.zeros(num_classes),where=cm.sum(0)>0)
    recall = np.divide(tp,cm.sum(1),out=np.zeros(num_classes),where=cm.sum(1)>0)
    f1 = np.divide(2*precision*recall,precision+recall,
                   out=np.zeros(num_classes),where=precision+recall>0)
    return dict(accuracy=float(tp.sum()/max(1,cm.sum())), macro_f1=float(f1.mean()),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                support=cm.sum(1).tolist(), confusion_matrix=cm.tolist())


def run_epoch(model, loader, device, num_classes, optimizer=None):
    # 第2段：optimizer存在则训练，否则关闭梯度；验证和测试不更新参数。
    training = optimizer is not None
    model.train(training)
    labels, predictions, records = [], [], []
    loss_sum, count = 0., 0
    with torch.set_grad_enabled(training):
        for x,y,ids in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits,y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # 第3段：概率仅供输出；未经校准，不能解释为科学可靠度。
            probs = logits.detach().softmax(-1).cpu()
            pred = probs.argmax(-1)
            labels.extend(y.cpu().tolist()); predictions.extend(pred.tolist())
            loss_sum += loss.item()*len(y); count += len(y)
            for sid,truth,p,pr in zip(ids,y.cpu().tolist(),pred.tolist(),probs.tolist()):
                records.append(dict(sample_id=sid,label=truth,prediction=p,probabilities=pr))
    result = classification_metrics(labels,predictions,num_classes)
    result['loss'] = loss_sum / max(1,count)
    return result,records
