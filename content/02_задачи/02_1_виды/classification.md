---
title: Классификация
---
Классификация — определение принадлежности объекта к одной или нескольким категориям на основе его признаков.

Бинарная классификация — два взаимоисключающих класса ($K = 2$):

$$f: \mathbb{R}^d \to \{0, 1\} \quad \text{или} \quad f(\mathbf{x}) = P(Y = 1 \mid \mathbf{X} = \mathbf{x}) \in [0, 1]$$

Многоклассовая классификация — несколько взаимоисключающих классов ($K > 2$):

$$f: \mathbb{R}^d \to \{1, \dots, K\} \quad \text{или} \quad \mathbf{f}(\mathbf{x}) = \begin{pmatrix} P(Y = 1 \mid \mathbf{X} = \mathbf{x}) \\ \vdots \\ P(Y = K \mid \mathbf{X} = \mathbf{x}) \end{pmatrix} \in \Delta^{K-1}$$

где $\Delta^{K-1} = \left\{\mathbf{p} \in [0, 1]^K \;\middle|\; \sum_{k=1}^K p_k = 1\right\}$ — вероятностный симплекс.

Мультилейбл-классификация — набор из $K$ независимых бинарных признаков (объект может принадлежать нескольким классам одновременно):

$$f: \mathbb{R}^d \to \{0, 1\}^K \quad \text{или} \quad \mathbf{f}(\mathbf{x}) = \begin{pmatrix} P(Y_1 = 1 \mid \mathbf{X} = \mathbf{x}) \\ \vdots \\ P(Y_K = 1 \mid \mathbf{X} = \mathbf{x}) \end{pmatrix} \in [0, 1]^K$$
