---
title: LSTM & GRU
aliases:
  - Long Short-Term Memory
  - Gated Recurrent Unit
---
LSTM (Long Short-Term Memory) и GRU (Gated Recurrent Unit) — [[rnn|рекуррентные]] ячейки для обработки последовательностей с механизмами гейтирования и аддитивного обновления состояния.

| Свойство     | Описание                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Вход         | Тензор $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{in}}}$ или батч $\mathbf{X} \in \mathbb{R}^{n \times T \times d_{\text{in}}}$                                           |
| Выход        | Скрытые состояния $\mathbf{H} \in \mathbb{R}^{T \times d_h}$ (и $\mathbf{C} \in \mathbb{R}^{T \times d_h}$ для LSTM) или $\boldsymbol{h}_T$                                  |

> [!math]+ Математика
> Пошаговое обновление состояния:
> 
> - LSTM: $f: (\mathbb{R}^{d_{\text{in}}}, \mathbb{R}^{d_h}, \mathbb{R}^{d_h}) \to (\mathbb{R}^{d_h}, \mathbb{R}^{d_h})$
> - GRU: $f: (\mathbb{R}^{d_{\text{in}}}, \mathbb{R}^{d_h}) \to \mathbb{R}^{d_h}$
> 
> Уравнения ячейки LSTM:
> 
> 1. Вектор гейта забывания:
> 
>    $$\boldsymbol{f}_t = \sigma\left( \mathbf{W}_f \boldsymbol{x}_t + \mathbf{U}_f \boldsymbol{h}_{t-1} + \boldsymbol{b}_f \right)$$
> 
> 2. Вектор входного гейта и кандидат:
> 
>    $$\boldsymbol{i}_t = \sigma\left( \mathbf{W}_i \boldsymbol{x}_t + \mathbf{U}_i \boldsymbol{h}_{t-1} + \boldsymbol{b}_i \right)$$
> 
>    $$\tilde{\boldsymbol{c}}_t = \tanh\left( \mathbf{W}_c \boldsymbol{x}_t + \mathbf{U}_c \boldsymbol{h}_{t-1} + \boldsymbol{b}_c \right)$$
> 
> 3. Обновление долгосрочной памяти:
> 
>    $$\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t$$
> 
> 4. Выходной гейт и скрытое состояние:
> 
>    $$\boldsymbol{o}_t = \sigma\left( \mathbf{W}_o \boldsymbol{x}_t + \mathbf{U}_o \boldsymbol{h}_{t-1} + \boldsymbol{b}_o \right)$$
> 
>    $$\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t)$$
> 
> Уравнения модификации GRU:
> 
> $$\boldsymbol{z}_t = \sigma(\mathbf{W}_z \boldsymbol{x}_t + \mathbf{U}_z \boldsymbol{h}_{t-1} + \boldsymbol{b}_z), \quad \boldsymbol{r}_t = \sigma(\mathbf{W}_r \boldsymbol{x}_t + \mathbf{U}_r \boldsymbol{h}_{t-1} + \boldsymbol{b}_r)$$
> 
> $$\boldsymbol{h}_t = (1 - \boldsymbol{z}_t) \odot \boldsymbol{h}_{t-1} + \boldsymbol{z}_t \odot \tanh\left( \mathbf{W}_h \boldsymbol{x}_t + \mathbf{U}_h (\boldsymbol{r}_t \odot \boldsymbol{h}_{t-1}) + \boldsymbol{b}_h \right)$$
> 
> Где:
> 
> - $\boldsymbol{x}_t \in \mathbb{R}^{d_{\text{in}} \times 1}$ — вектор-столбец входа;
> - $\boldsymbol{h}_{t-1} \in \mathbb{R}^{d_h \times 1}$ — прошлый вектор скрытого состояния;
> - $\boldsymbol{c}_{t-1} \in \mathbb{R}^{d_h \times 1}$ — прошлый вектор памяти (только для LSTM);
> - $\mathbf{W} \in \mathbb{R}^{d_h \times d_{\text{in}}}, \mathbf{U} \in \mathbb{R}^{d_h \times d_h}$ — весовые матрицы;
> - $\boldsymbol{b} \in \mathbb{R}^{d_h \times 1}$ — векторы смещений;
> - $\sigma(\cdot)$ — сигмоида, $\odot$ — произведение Адамара.
> 
> Расчет пошагового обновления на шаге $t$:
> 
> 1. Вычисление коэффициентов гейтов.
> 2. Формирование кандидата нового состояния.
> 3. Обновление состояния (суммирование в LSTM / интерполяция в GRU).

---

## Обучение

[[оптимизация|Оптимизация]] весов и смещений методом [[sgd|BPTT (Backpropagation Through Time)]].

| Свойство                   | Описание                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Обучаемые параметры        | 4 набора весов для LSTM / 3 для GRU                                                                            |
| Гиперпараметры             | Размерность $d_h$, число слоев, Dropout, порог Clipping                                                        |

---

## Инференс

Расчет состояний от $t = 1$ до $T$ для $\mathbf{X}_{\text{new}} \in \mathbb{R}^{T \times d_{\text{in}}}$:

$$\mathbf{H} = f(\mathbf{X}_{\text{new}})$$
