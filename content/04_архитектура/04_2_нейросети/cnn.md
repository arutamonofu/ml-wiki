---
title: Convolutional neural network
aliases:
  - CNN
  - Сверточные нейронные сети
---
Свёрточная нейронная сеть (Convolutional Neural Network, CNN) — архитектура для обработки пространственных сеточных данных путем иерархического применения локальных свёрток с разделяемыми весами.

| Свойство     | Описание                                                                                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Вход         | Пространственный тензор $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$                                                                                   |
| Выход        | Вектор откликов $\boldsymbol{y} \in \mathbb{R}^K$, логиты $\boldsymbol{z} \in \mathbb{R}^K$ или тензор $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times H' \times W'}$ |

> [!math]+ Математика
> Оператор: $f: \mathbb{R}^{C_{\text{in}} \times H \times W} \to \mathcal{Y}$
> 
> Элемент карты признаков $k$-го канала на слое $l$ в позиции $(i, j)$:
> 
> $$Y_{k, i, j}^{(l)} = \sigma \left( \sum_{c=1}^{C_{l-1}} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} W_{k, c, m, n}^{(l)} \cdot X_{c, \, i \cdot s + m - p, \, j \cdot s + n - p}^{(l-1)} + b_k^{(l)} \right)$$
> 
> Выходное разрешение ($H', W'$):
> 
> $$H' = \left\lfloor \frac{H + 2p - K}{s} \right\rfloor + 1, \quad W' = \left\lfloor \frac{W + 2p - K}{s} \right\rfloor + 1$$
> 
> Классификация через GAP:
> 
> $$\hat{\boldsymbol{y}} = \operatorname{softmax}\left( \mathbf{W}_{\text{FC}} \cdot \operatorname{GAP}\left( f_L \circ \dots \circ f_1(\mathbf{X}) \right) + \boldsymbol{b}_{\text{FC}} \right)$$
> 
> где:
> 
> - $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ — входной тензор;
> - $\mathbf{W}_k^{(l)} \in \mathbb{R}^{C_{l-1} \times K \times K}$ — $k$-е ядро свёртки $K \times K$;
> - $s$ — stride, $p$ — padding;
> - $\sigma(\cdot)$ — функция активации;
> - $b_k^{(l)} \in \mathbb{R}$ — свободный член $k$-го ядра.
> 
> Расчет 2D-свёртки ($C_{\text{in}} = 1$, ядро $3 \times 3$, $s=1$, $p=0$) в позиции $(i, j)$:
> 
> 1. Взвешенная сумма:
> 
>    $$z_{i, j} = \sum_{m=0}^{2} \sum_{n=0}^{2} W_{m, n} \cdot X_{i+m, j+n} + b$$
> 
> 2. Активация:
> 
>    $$Y_{i, j} = \sigma(z_{i, j})$$

---

## Обучение

[[оптимизация|Оптимизация]] ядер $\mathbf{W}^{(l)}$ и смещений $\boldsymbol{b}^{(l)}$ [[sgd|методом обратного распространения ошибки]].

| Свойство                   | Описание                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Обучаемые параметры        | Веса ядер $\mathbf{W}^{(l)}$, параметры полносвязных слоев и нормализации                                                                                      |
| Гиперпараметры             | Слои $L$, фильтры $C_l$, размер ядра $K$, stride $s$, padding $p$, активация                                                                                   |

---

## Инференс

Прямой проход для объекта $\mathbf{X}_{\text{new}} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$:

$$\hat{\boldsymbol{y}} = f(\mathbf{X}_{\text{new}})$$
