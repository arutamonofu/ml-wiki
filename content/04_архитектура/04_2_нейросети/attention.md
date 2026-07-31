---
title: Attention
aliases:
  - Механизм внимания
---
Механизм внимания (attention) — блок нейросети, агрегирующий векторы значений (Values) с весами релевантности их ключей (Keys) текущему запросу (Query).

| Свойство     | Описание                                                                                                                                                                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Вход         | Тензоры запросов $\mathbf{Q} \in \mathbb{R}^{T_Q \times d_q}$, ключей $\mathbf{K} \in \mathbb{R}^{T_K \times d_k}$ и значений $\mathbf{V} \in \mathbb{R}^{T_K \times d_v}$ |
| Выход        | Взвешенный контекстный тензор $\mathbf{H} \in \mathbb{R}^{T_Q \times d_v}$                                                                                                 |

> [!math]+ Математика
> Оператор: $f: (\mathbb{R}^{T_Q \times d_q}, \mathbb{R}^{T_K \times d_k}, \mathbb{R}^{T_K \times d_v}) \to \mathbb{R}^{T_Q \times d_v}$
> 
> Матричная форма (softmax построчно по оси $T_K$):
> 
> $$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \operatorname{Score}(\mathbf{Q}, \mathbf{K}) \right) \mathbf{V}$$
> 
> Функции сходства $\operatorname{Score}(\mathbf{q}, \mathbf{k})$:
> 
> - Аддитивная (Bahdanau): $\operatorname{Score}(\mathbf{q}, \mathbf{k}) = \mathbf{v}_a^\top \tanh(\mathbf{W}_q \mathbf{q}^\top + \mathbf{W}_k \mathbf{k}^\top)$;
> - Мультипликативная (Luong): $\operatorname{Score}(\mathbf{q}, \mathbf{k}) = \mathbf{q} \mathbf{W}_a \mathbf{k}^\top$;
> - Скалярное произведение (Vaswani, $d_q = d_k$): $\operatorname{Score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \mathbf{k}^\top}{\sqrt{d_k}}$.
> 
> Проекции Self-Attention ($\mathbf{X} \in \mathbb{R}^{T \times d_{\text{in}}}$, $T_Q = T_K = T$):
> 
> $$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
> 
> где $\mathbf{W}_Q \in \mathbb{R}^{d_{\text{in}} \times d_q}, \mathbf{W}_K \in \mathbb{R}^{d_{\text{in}} \times d_k}, \mathbf{W}_V \in \mathbb{R}^{d_{\text{in}} \times d_v}$.
> 
> Расчет для запроса $\mathbf{q} \in \mathbb{R}^{1 \times d_q}$ и $T_K$ пар ключей/значений:
> 
> 1. Логиты:
> 
>    $$a_i = \operatorname{Score}(\mathbf{q}, \mathbf{k}_i), \quad i = 1, \dots, T_K$$
> 
> 2. Веса:
> 
>    $$\alpha_i = \operatorname{softmax}(\mathbf{a})_i = \frac{\exp(a_i)}{\sum_{j=1}^{T_K} \exp(a_j)}$$
> 
> 3. Контекст:
> 
>    $$\mathbf{h} = \sum_{i=1}^{T_K} \alpha_i \mathbf{v}_i \in \mathbb{R}^{1 \times d_v}$$

---

## Обучение

[[оптимизация|Оптимизация]] параметров $\operatorname{Score}(\cdot)$ и матриц проекций $\mathbf{W}$ [[sgd|методом обратного распространения ошибки]].

| Свойство                   | Описание                                                                                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Обучаемые параметры        | Веса $\operatorname{Score}$ ($\mathbf{W}_a, \mathbf{v}_a$) и матрицы проекций $\mathbf{W}_Q \in \mathbb{R}^{d_{\text{in}} \times d_q}, \mathbf{W}_K \in \mathbb{R}^{d_{\text{in}} \times d_k}, \mathbf{W}_V \in \mathbb{R}^{d_{\text{in}} \times d_v}$ |
| Гиперпараметры             | Вид $\operatorname{Score}$, размерности $d_q, d_k, d_v$, число голов $h$, тип маски                                                                                                                                                                    |


---

## Инференс

Расчет контекста для входов $\mathbf{Q}_{\text{new}}, \mathbf{K}_{\text{new}}, \mathbf{V}_{\text{new}}$:

$$\mathbf{H} = \operatorname{softmax}\left( \operatorname{Score}(\mathbf{Q}_{\text{new}}, \mathbf{K}_{\text{new}}) \right) \mathbf{V}_{\text{new}}$$
