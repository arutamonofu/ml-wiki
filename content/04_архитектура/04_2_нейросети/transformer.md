---
title: Transformer
aliases:
  - Трансформер
---
Трансформер (transformer) — нейросетевая архитектура для обработки последовательностей на основе механизмов многоголового внутреннего внимания (multi-head self-attention) без рекурсии и свёрток.

| Свойство     | Описание                                                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Вход         | Последовательность индексов токенов $\boldsymbol{x} \in \{1, \dots, V\}^T$ или тензор эмбеддингов $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$          |
| Выход        | Контекстуализированные векторы $\mathbf{H} \in \mathbb{R}^{T \times d_{\text{model}}}$ или логиты следующих токенов $\hat{\mathbf{Y}} \in \mathbb{R}^{T \times V}$ |

> [!math]+ Математика
> Оператор блока: $f: \mathbb{R}^{T \times d_{\text{model}}} \to \mathbb{R}^{T \times d_{\text{model}}}$
> 
> 1. Вход с позиционным кодированием:
> 
>    $$\mathbf{Z}^{(0)} = \mathbf{X} + \mathbf{P}$$
> 
>    $$P_{t, 2i} = \sin\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right), \quad P_{t, 2i+1} = \cos\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)$$
> 
> 2. Многоголовое внимание (MHA):
> 
>    $$\operatorname{MHA}(\mathbf{Z}) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O$$
> 
>    $$\text{head}_i = \operatorname{Attention}(\mathbf{Z} \mathbf{W}_i^Q, \, \mathbf{Z} \mathbf{W}_i^K, \, \mathbf{Z} \mathbf{W}_i^V) = \operatorname{softmax}\left( \frac{(\mathbf{Z} \mathbf{W}_i^Q)(\mathbf{Z} \mathbf{W}_i^K)^\top}{\sqrt{d_k}} \right) (\mathbf{Z} \mathbf{W}_i^V)$$
> 
> 3. Обновление блока (Pre-LN) и FFN:
> 
>    $$\mathbf{Z}^{(l)'} = \mathbf{Z}^{(l-1)} + \operatorname{MHA}(\operatorname{LayerNorm}(\mathbf{Z}^{(l-1)}))$$
> 
>    $$\mathbf{Z}^{(l)} = \mathbf{Z}^{(l)'} + \operatorname{FFN}(\operatorname{LayerNorm}(\mathbf{Z}^{(l)'}))$$
> 
>    $$\operatorname{FFN}(\mathbf{Z}) = \sigma(\mathbf{Z} \mathbf{W}_1 + \boldsymbol{b}_1) \mathbf{W}_2 + \boldsymbol{b}_2$$
> 
> где:
> 
> - $T$ — длина последовательности;
> 
> - $d_{\text{model}}$ — размерность векторных представлений;
> 
> - $h$ — число голов ($d_k = d_v = d_{\text{model}} / h$);
> 
> - $\mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}, \mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ — матрицы проекций внимания;
> 
> - $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, \mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}, \boldsymbol{b}_1 \in \mathbb{R}^{d_{\text{ff}}}, \boldsymbol{b}_2 \in \mathbb{R}^{d_{\text{model}}}$ — веса FFN;
> 
> - $\mathbf{P} \in \mathbb{R}^{T \times d_{\text{model}}}$ — позиционное кодирование.
> 
> Расчет блока для матрицы $\mathbf{X}$:
> 
> 1. Сложение эмбеддингов с позиционным кодированием.
> 
> 2. MHA с LayerNorm и остаточной связью.
> 
> 3. Попозиционный FFN с LayerNorm и остаточной связью.

---

## Обучение

[[оптимизация|Оптимизация]] параметров эмбеддингов, внимания ($\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O$), FFN ($\mathbf{W}_1, \mathbf{W}_2$) и LayerNorm [[sgd|методом обратного распространения ошибки]].

| Свойство                   | Описание                                                                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Обучаемые параметры        | Проекции внимания, веса FFN, таблица эмбеддингов, параметры LayerNorm                                                                                                                            |
| Гиперпараметры             | Число слоев $L$, размерность $d_{\text{model}}$, число голов $h$, размерность $d_{\text{ff}}$, длина контекста $T$, размер словаря $V$                                                           |

---

## Инференс

Режимы работы:

1. Энкодерный: Параллельный прямой проход по всем токенам за один шаг.
2. Авторегрессионный: Пошаговая генерация с KV-кэшированием прошлых ключей и значений.
