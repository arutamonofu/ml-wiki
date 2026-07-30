---
title: Support vector machine
aliases:
  - SVM
  - Метод опорных векторов
---
Метод опорных векторов (SVM) — алгоритм классификации и регрессии, строящий разделяющую гиперплоскость с максимальным зазором в исходном или спрямляющем пространстве.

| Свойство     | Описание                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------- |
| Задачи       | [[classification\|Классификация]], [[regression\|регрессия]]                                       |
| Тип обучения | С учителем                                                                                         |
| Вход         | Матрица признаков $\mathbf{X} \in \mathbb{R}^{n \times d}$                                         |
| Выход        | Вектор откликов $\hat{\boldsymbol{y}} \in \{-1, 1\}^n$ или $\hat{\boldsymbol{y}} \in \mathbb{R}^n$ |

> [!math]+ Математика
>
> Для объекта $\boldsymbol{x} \in \mathbb{R}^d$:
>
> $$f(\boldsymbol{x}) = \operatorname{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\boldsymbol{x}_i, \boldsymbol{x}) + b \right)$$
>
> Для выборки $\mathbf{X} \in \mathbb{R}^{n \times d}$:
>
> $$\hat{\boldsymbol{y}} = \operatorname{sign}\left( \mathbf{K}(\mathbf{X}, \mathbf{X}_\text{SV}) (\boldsymbol{\alpha}_\text{SV} \odot \boldsymbol{y}_\text{SV}) + b\mathbf{1} \right)$$
>
> Где:
>
> - $\mathbf{X} \in \mathbb{R}^{n \times d}$ — матрица признаков;
> - $\mathbf{X}_\text{SV} \in \mathbb{R}^{n_\text{SV} \times d}$ — матрица опорных векторов ($\alpha_i > 0$);
> - $\mathbf{K}(\mathbf{X}, \mathbf{X}_\text{SV}) \in \mathbb{R}^{n \times n_\text{SV}}$ — матрица Грама;
> - $\boldsymbol{\alpha}_\text{SV} \in \mathbb{R}^{n_\text{SV}}$ — вектор двойственных множителей Лагранжа;
> - $\boldsymbol{y}_\text{SV} \in \{-1, 1\}^{n_\text{SV}}$ — метки опорных векторов;
> - $K(\boldsymbol{x}_i, \boldsymbol{x})$ — ядерная функция;
> - $b \in \mathbb{R}$ — смещение;
> - $\mathbf{1} \in \mathbb{R}^n$ — единичный вектор;
> - $\odot$ — произведение Адамара;
> - $\operatorname{sign}(\cdot)$ — функция знака.

---

## Обучение

Двойственная задача квадратичного программирования для мягкого зазора:

$$\max_{\boldsymbol{\alpha}} \left( \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(\boldsymbol{x}_i, \boldsymbol{x}_j) \right)$$

при $0 \le \alpha_i \le C$ и $\sum_{i=1}^{n} \alpha_i y_i = 0$.

Решается алгоритмом SMO или градиентными методами.

| Свойство | Описание |
| --- | --- |
| Параметры | Двойственные множители $\boldsymbol{\alpha} \in \mathbb{R}^n$, смещение $b \in \mathbb{R}$ (или веса $\boldsymbol{w} \in \mathbb{R}^d$ для линейного ядра) |
| Гиперпараметры | Регуляризация $C$, ядерная функция $K(\cdot, \cdot)$, точность сходимости |
| Временная сложность | $\mathcal{O}(n^2 d)$ – $\mathcal{O}(n^3 d)$ |
| Пространственная сложность | $\mathcal{O}(n^2)$ (или $\mathcal{O}(n \cdot b_\text{cache})$ при кэшировании) |

---

## Инференс

Для нового объекта $\boldsymbol{x}_\text{new} \in \mathbb{R}^d$:

$$\hat{y} = \operatorname{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\boldsymbol{x}_i, \boldsymbol{x}_\text{new}) + b \right)$$

| Свойство | Описание |
| --- | --- |
| Временная сложность | Линейное ядро: $\mathcal{O}(d)$ на объект<br>Нелинейное ядро: $\mathcal{O}(n_{\text{SV}} \cdot d)$ на объект |
| Пространственная сложность | Линейное ядро: $\mathcal{O}(d)$ для весов $\boldsymbol{w}$ и $b$<br>Нелинейное ядро: $\mathcal{O}(n_{\text{SV}} \cdot d)$ для опорных векторов и весов |

---

## Достоинства и недостатки

Достоинства:

- Гарантия глобального оптимума благодаря выпуклости задачи.
- Построение нелинейных границ через ядра.
- Устойчивость в пространствах высокой размерности.

Недостатки:

- Чувствительность к [[масштабирование|масштабу признаков]].
- Высокая вычислительная сложность на больших выборках.
- Сложность подбора гиперпараметров и чувствительность к выбросам.
