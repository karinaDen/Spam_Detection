# Spam_Detection

Основываясь на данных предоставленных в [статье](https://downloads.hindawi.com/journals/scn/2022/1862888.pdf?_gl=1*rrh19a*_ga*MTcyMzY4NTY4OS4xNzE0NjY1OTQ4*_ga_NF5QFMJT5V*MTcxNDY2NTk0OC4xLjAuMTcxNDY2NTk0OC42MC4wLjA.&_ga=2.74513872.359558069.1714665948-1723685689.1714665948). Лучшими моделями для определания спама являются SVM и Random Forest. По результатам тестирования и анализа ROC AUC метрик, на данном не большом датасете SVM показал чуть лучшие результаты 

UPD: модель roberta претрейненная на более крупном датасете для классификации скама, показала в итоге наилучший результат (почти 1), что в целом вполне ожидаемо, так как предобучена на большом датасете, лучше понимает контекст и устойчива к шуму, лучше справляется с нелинейностью