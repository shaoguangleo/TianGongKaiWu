# 验证记录

Python 3.12.14；torch 2.15.0+cu132；numpy 2.3.5；astropy 8.0.2；pytest 9.2.0。CPU运行。

- python -m pytest -q：5 passed。
- 模拟数据：120训练、30验证、30测试，共180例。
- 12轮训练、验证选优、checkpoint恢复、独立测试、单图预测均完成。
- 模拟测试accuracy/macro-F1均为1.0，不能外推为真实RACS性能。并列macro-F1保留较早模型，所以best.pt为第2轮，概率较低。
- 单FITS预测已生成runs/prediction.json。
- 未验证GPU、真实RACS数据、跨机器一致性。
- 源码已进行作者自查；本包未做独立代码审查。

完整示例日志见runs/history.json及test_metrics.json。
