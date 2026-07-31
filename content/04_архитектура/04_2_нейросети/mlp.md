---
title: Multilayer perceptron
aliases:
  - Многослойный перцептрон
  - MLP
---
Многослойный перцептрон (multilayer perceptron) — глубокая нейросеть прямого распространения (feedforward), реализующая параметрическую функцию последовательной композицией полносвязных слоев и нелинейных активаций.

| Свойство     | Описание                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Вход         | Вектор $\boldsymbol{x} \in \mathbb{R}^{d_{\text{in}}}$ или батч $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{in}}}$          |
| Выход        | Вектор откликов $\boldsymbol{y} \in \mathbb{R}^{d_{\text{out}}}$ (логиты или вероятности)                                       |

> [!math]+ Математика
> Оператор: $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$
> 
> $$\boldsymbol{h}_0 = \boldsymbol{x}$$
> 
> $$\boldsymbol{h}_l = \sigma_l \left( \mathbf{W}_l \boldsymbol{h}_{l-1} + \boldsymbol{b}_l \right), \quad l = 1, \dots, L-1$$
> 
> $$\hat{\boldsymbol{y}} = \sigma_L \left( \mathbf{W}_L \boldsymbol{h}_{L-1} + \boldsymbol{b}_L \right)$$
> 
> где:
> 
> - $\boldsymbol{x} \in \mathbb{R}^{d_0}$ — входной вектор ($d_0 = d_{\text{in}}$);
> - $L$ — число слоев;
> - $\mathbf{W}_l \in \mathbb{R}^{d_l \times d_{l-1}}$ — веса $l$-го слоя;
> - $\boldsymbol{b}_l \in \mathbb{R}^{d_l}$ — смещения $l$-го слоя;
> - $\sigma_l(\cdot)$ — поэлементная активация $l$-го слоя;
> - $\boldsymbol{h}_l \in \mathbb{R}^{d_l}$ — скрытое состояние $l$-го слоя.
> 
> Расчет двухслойного персептрона для входа $\boldsymbol{x}$:
> 
> 1. Аффинный шаг скрытого слоя:
> 
>    $$\boldsymbol{z}_1 = \mathbf{W}_1 \boldsymbol{x} + \boldsymbol{b}_1$$
> 
> 2. Активация:
> 
>    $$\boldsymbol{h}_1 = \sigma_1(\boldsymbol{z}_1)$$
> 
> 3. Выходной слой:
> 
>    $$\hat{\boldsymbol{y}} = \sigma_2(\mathbf{W}_2 \boldsymbol{h}_1 + \boldsymbol{b}_2)$$

---

## Обучение

[[оптимизация|Оптимизация]] параметров $\{\mathbf{W}_l, \boldsymbol{b}_l\}_{l=1}^L$ [[sgd|методом обратного распространения ошибки]].

| Свойство                   | Описание                                                                                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Обучаемые параметры        | Веса $\{\mathbf{W}_l\}_{l=1}^L$ и смещения $\{\boldsymbol{b}_l\}_{l=1}^L$                                                                             |
| Гиперпараметры             | Число слоев $L$, размерности $d_l$, функции активации, параметры регуляризации                                                                        |

---

## Инференс

Прямое вычисление активаций от входа к выходу для объекта $\boldsymbol{x}_{\text{new}} \in \mathbb{R}^{d_{\text{in}}}$:

$$\hat{\boldsymbol{y}} = f(\boldsymbol{x}_{\text{new}})$$
