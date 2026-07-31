---
title: Reccurent neural network
aliases:
  - Рекуррентные нейронные сети
  - RNN
---
Рекуррентная нейронная сеть (RNN) — архитектура для обработки последовательностей путем последовательного применения рекуррентного оператора с общей памятью (разделяемыми весами).

| Свойство     | Описание                                                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Вход         | Тензор $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{in}}}$ или батч $\mathbf{X} \in \mathbb{R}^{B \times T \times d_{\text{in}}}$                                                                     |
| Выход        | Состояния $\mathbf{H} \in \mathbb{R}^{T \times d_h}$, вектор $\boldsymbol{h}_T \in \mathbb{R}^{d_h}$ или вероятности $\hat{\mathbf{Y}} \in \mathbb{R}^{T \times d_{\text{out}}}$                        |

> [!math]+ Математика
> Оператор: $f: \mathbb{R}^{T \times d_{\text{in}}} \to \mathbb{R}^{T \times d_{\text{out}}}$
> 
> Вычисление скрытого состояния:
> 
> $$\boldsymbol{h}_t = \sigma\left( \mathbf{W}_{hh} \boldsymbol{h}_{t-1} + \mathbf{W}_{xh} \boldsymbol{x}_t + \boldsymbol{b}_h \right), \quad t = 1, \dots, T$$
> 
> Классификация последовательности по последнему состоянию:
> 
> $$\hat{\boldsymbol{y}} = \operatorname{softmax}\left( \mathbf{W}_{hy} \boldsymbol{h}_T + \boldsymbol{b}_y \right)$$
> 
> Обозначения:
> 
> - $\mathbf{X} = [\boldsymbol{x}_1, \dots, \boldsymbol{x}_T]^\top \in \mathbb{R}^{T \times d_{\text{in}}}$ — матрица входа из векторов $\boldsymbol{x}_t \in \mathbb{R}^{d_{\text{in}}}$;
> - $\boldsymbol{h}_t \in \mathbb{R}^{d_h}$ — вектор скрытого состояния ($\boldsymbol{h}_0 = \boldsymbol{0}$);
> - $\mathbf{W}_{xh} \in \mathbb{R}^{d_h \times d_{\text{in}}}, \mathbf{W}_{hh} \in \mathbb{R}^{d_h \times d_h}, \mathbf{W}_{hy} \in \mathbb{R}^{d_{\text{out}} \times d_h}$ — матрицы весов;
> - $\boldsymbol{b}_h \in \mathbb{R}^{d_h}, \boldsymbol{b}_y \in \mathbb{R}^{d_{\text{out}}}$ — векторы смещений;
> - $\sigma(\cdot)$ — поэлементная активация (обычно $\tanh$).
> 
> Расчет обновления на шаге $t$:
> 
> 1. Линейная комбинация:
> 
>    $$\boldsymbol{z}_t = \mathbf{W}_{xh} \boldsymbol{x}_t + \mathbf{W}_{hh} \boldsymbol{h}_{t-1} + \boldsymbol{b}_h$$
> 
> 2. Активация состояния:
> 
>    $$\boldsymbol{h}_t = \tanh(\boldsymbol{z}_t)$$
> 
> 3. Вероятности отклика:
> 
>    $$\hat{\boldsymbol{y}}_t = \operatorname{softmax}\left( \mathbf{W}_{hy} \boldsymbol{h}_t + \boldsymbol{b}_y \right)$$

---

## Обучение

[[оптимизация|Оптимизация]] параметров $\mathbf{W}$ и $\boldsymbol{b}$ минимизацией функции потерь [[sgd|методом обратного распространения ошибки во времени (BPTT)]].

| Свойство                   | Описание                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Обучаемые параметры        | Матрицы $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$ и векторы $\boldsymbol{b}_h, \boldsymbol{b}_y$                      |
| Гиперпараметры             | Размерность $d_h$, число слоев, длина $T$, порог Gradient Clipping                                                              |


---

## Инференс

Последовательный расчет состояний от $t = 1$ до $t = T$ для входа $\mathbf{X}_{\text{new}} \in \mathbb{R}^{T \times d_{\text{in}}}$:

$$\hat{\mathbf{Y}} = f(\mathbf{X}_{\text{new}})$$
