---
title: DBSCAN
---
DBSCAN — плотностный алгоритм кластеризации, выделяющий связные области высокой плотности и помечающий разреженные данные как шум.

| Свойство     | Описание                                                                       |
| ------------ | ------------------------------------------------------------------------------ |
| Задачи       | [[clustering\|Кластеризация]], [[anomaly_detection\|обнаружение аномалий]]     |
| Тип обучения | Без учителя                                                                    |
| Вход         | Матрица признаков $\mathbf{X} \in \mathbb{R}^{n \times d}$                     |
| Выход        | Вектор меток $\hat{\boldsymbol{y}} \in \{-1, 1, \dots, K\}^n$ (где $-1$ — шум) |

> [!math]+ Математика
> Пусть $N_\varepsilon(\boldsymbol{x}_i) = \{\boldsymbol{x}_j \in \mathbf{X} \mid \rho(\boldsymbol{x}_i, \boldsymbol{x}_j) \le \varepsilon\}$ — $\varepsilon$-окрестность $\boldsymbol{x}_i$.
> 
> Типы объектов:
> 
> - Корневой: $|N_\varepsilon(\boldsymbol{x}_i)| \ge m$ (множество $\mathbf{X}_\text{core}$);
> - Пограничный: $|N_\varepsilon(\boldsymbol{x}_i)| < m$, но $\exists \boldsymbol{x}_c \in \mathbf{X}_\text{core} : \boldsymbol{x}_i \in N_\varepsilon(\boldsymbol{x}_c)$;
> - Шумовой: не корневой и не пограничный.
> 
> Достижимость и связность:
> 
> 1. $\boldsymbol{x}_j$ прямо плотностно достижим из $\boldsymbol{x}_i$, если $\boldsymbol{x}_i \in \mathbf{X}_\text{core}$ и $\boldsymbol{x}_j \in N_\varepsilon(\boldsymbol{x}_i)$.
> 2. $\boldsymbol{x}_j$ плотностно достижим из $\boldsymbol{x}_i$, если есть цепочка $\boldsymbol{x}_i = \boldsymbol{p}_1, \dots, \boldsymbol{p}_k = \boldsymbol{x}_j$, где $\boldsymbol{p}_{l+1}$ прямо достижим из $\boldsymbol{p}_l$.
> 3. $\boldsymbol{x}_i$ и $\boldsymbol{x}_j$ плотностно связаны, если $\exists \boldsymbol{x}_o \in \mathbf{X}_\text{core}$, из которого они плотностно достижимы.
> 
> Кластер $C_k$ — максимальное множество плотностно-связанных объектов:
> 
> $$y_i = \begin{cases} k, & \text{если } \boldsymbol{x}_i \in C_k \\ -1, & \text{если } \boldsymbol{x}_i \text{ — шум} \end{cases}$$

---

## Обучение

Обход выборки и объединение плотностно-достижимых объектов вокруг корневых точек в компоненты связности.

| Свойство                   | Описание                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Сохраняемые структуры      | Корневые объекты $\mathbf{X}_\text{core} \subset \mathbf{X}$ и их метки                                     |
| Гиперпараметры             | Радиус $\varepsilon$, порог $m$, метрика $\rho$                                                             |
| Временная сложность        | $\mathcal{O}(n^2)$ при полном переборе<br>$\mathcal{O}(n \log n)$ с пространственными деревьями ($d \ll n$) |
| Пространственная сложность | $\mathcal{O}(n \cdot d)$ для признаков и меток                                                              |

---

## Инференс

Эвристика для новых данных $\boldsymbol{x}_\text{new} \in \mathbb{R}^d$:

Поиск ближайшего корневого объекта $\boldsymbol{x}_c^* = \operatorname{argmin}_{\boldsymbol{x}_c \in \mathbf{X}_\text{core}} \rho(\boldsymbol{x}_\text{new}, \boldsymbol{x}_c)$:

$$\hat{y}_\text{new} = \begin{cases} \operatorname{label}(\boldsymbol{x}_c^*), & \text{если } \rho(\boldsymbol{x}_\text{new}, \boldsymbol{x}_c^*) \le \varepsilon \\ -1, & \text{иначе} \end{cases}$$

| Свойство | Описание |
| --- | --- |
| Временная сложность (на 1 запрос) | $\mathcal{O}(n_\text{core} \cdot d)$ при полном переборе<br>$\mathcal{O}(d \log n_\text{core})$ с пространственным деревом |
| Пространственная сложность | $\mathcal{O}(n_\text{core} \cdot d)$ для корневых объектов |

---

## Достоинства и недостатки

Достоинства:

- Произвольная форма кластеров и автоопределение их количества;
- Явная фильтрация шума;
- Простая геометрическая интерпретация.

Недостатки:

- Чувствительность к [[масштабирование|масштабу признаков]] и метрике;
- Деградация в высокой размерности ($d \gg 1$) и при переменной плотности;
- Сложный подбор $\varepsilon$ и $m$.
