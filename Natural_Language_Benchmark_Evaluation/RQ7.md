# RQ7: Evaluation on Natural Language Benchmark

To validate whether the proposed approach is applicable to normal texts, we evaluated it on the **WIKIMIA dataset** [3], a publicly available normal text benchmark. The dataset was selected because of its high quality and public accessibility. Additionally, it had been used for the evaluation of our baseline (i.e., **Min-k% Prob**) [2]. The WIKIMIA dataset is composed of **seen** items that were created before 2017, and **unseen** items that were created after 2023. This evaluation was conducted using **ChatGPT 3.5**, a widely adopted and versatile language model known for its efficacy in handling both code and natural language tasks.

During the evaluation, we requested the model to accomplish a **text completion task**, where incomplete text is provided as input to the model, which is then tasked with generating the next tokens. The **token-level accuracy** metric was used to evaluate the performance of the model on the given testing datasets. In addition to accuracy, we also calculated the **naturalness of text** using the equations outlined in Section 3.3 of our paper. To compute naturalness, we trained an **n-gram model** using a separate text corpus of 60 million words collected from Wikipedia. This corpus is distinct from both the **seen** and **unseen** datasets used for testing, ensuring that the training data for the n-gram model does not overlap with the text data used for contamination detection.

  To distinguish between contaminated and clean data, we trained an **SVM classifier** using the following features:(1) The model’s performance, measured by token-level accuracy, and (2) The difficulty of the task, which is based on the naturalness of the text. The SVM classifier was trained on the model’s performance and the naturalness of text files from the WIKIMIA dataset, following the data participation methodology detailed in Section 4.2 of our paper.

  The results of the evaluation are shown in the table below. From the table, we observe that on all key metrics the proposed approach, i.e., **Natural-DaCoDe**, outperforms both baselines, i.e., **Loss Attack** and **Min-K% Prob**. It demonstrated a substantial performance boost, achieving an accuracy of **83.32%**, substantially higher than that of Loss Attack (**62.12%**) and Min-K% Prob (**68.31%**). The relative improvement in accuracy is 34% = (83.32 - 62.12%) / 62.12% (compared to Loss Attack) and 22% = (83.32 - 68.31%) / 68.31% (compared to Min-K% Prob). Additionally, **Natural-DaCoDe** exhibited a **true positive rate (TPR)** of **84.11%** and a significantly lower **false positive rate (FPR)** of **22.93%**, resulting in an **AUC score of 0.88**. Its AUC score is substantially greater than that of Min-K% Prob (**74%**) and Loss Attack (**64%**).

We conclude based on the preceding analysis that the proposed approach could be applied to normal texts besides source code, and it remains more accurate than the state-of-the-art baselines in this field. The replication package of the evaluation is publicly available at [1].

### Table 1. Performance Comparison on Normal Text Benchmark

| Approaches     | Accuracy (%) | TPR (%) | FPR (%) | AUC (%) |
|----------------|--------------|---------|---------|---------|
| **Our Approach (Natural-DaCoDe)** | **83.32** | **84.11** | **22.93** | **88.00** |
| **Min-K% Prob** | 68.31        | 76.09   | 37.98   | 74.00   |
| **Loss Attack** | 62.12        | 64.03   | 41.21   | 64.00   |

## References
[1](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Natural_Language_Benchmark_Evaluation) :Naturalnessbasedappraoch. 2024. Natural Language Benchmark Evaluation. https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Natural_Language_Benchmark_Evaluation. Accessed:2024.

[2]:Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettle-moyer. 2024. Detecting Pretraining Data from Large Language Models. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=zWqr3MQuNs

[3](https://huggingface.co/datasets/swj0419/WikiMIA) :SWJ0419. [n. d.]. WikiMIA. https://huggingface.co/datasets/swj0419/WikiMIA. Accessed: 2024.
