---
title: Diffusion model
aliases:
  - Диффузионная модель
---
Диффузионная модель (Diffusion Model) — генеративная нейросеть, восстанавливающая данные из [[распределение_нормальное_(гаусса)|гауссовского]] шума путем пошагового шумоподавления.

| Свойство     | Описание                                                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Вход         | Зашумленный тензор $\mathbf{X}_t \in \mathbb{R}^{C \times H \times W}$, шаг $t \in \{1, \dots, T\}$, условие $\boldsymbol{c}$                                  |
| Выход        | Предсказанный шум $\boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \boldsymbol{c}) \in \mathbb{R}^{C \times H \times W}$ или очищенный объект $\hat{\mathbf{X}}_0$ |

> [!math]+ Математика
> Оператор сети: $f: (\mathbb{R}^{C \times H \times W}, \mathbb{R}, \mathcal{C}) \to \mathbb{R}^{C \times H \times W}$
> 
> Прямой марковский процесс:
> 
> $$q(\mathbf{X}_t \mid \mathbf{X}_{t-1}) = \mathcal{N}\left(\mathbf{X}_t; \sqrt{1 - \beta_t} \mathbf{X}_{t-1}, \beta_t \mathbf{I}\right)$$
> 
> Прямой переход на шаг $t$ из $\mathbf{X}_0$:
> 
> $$\mathbf{X}_t = \sqrt{\bar{\alpha}_t} \mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
> 
> Обратный процесс:
> 
> $$p_\theta(\mathbf{X}_{t-1} \mid \mathbf{X}_t, \boldsymbol{c}) = \mathcal{N}\left(\mathbf{X}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{X}_t, t, \boldsymbol{c}), \boldsymbol{\Sigma}_\theta(\mathbf{X}_t, t, \boldsymbol{c})\right)$$
> 
> $$\boldsymbol{\mu}_\theta(\mathbf{X}_t, t, \boldsymbol{c}) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{X}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \boldsymbol{c}) \right)$$
> 
> Обозначения:
> 
> - $\mathbf{X}_0$ — исходный объект
> 
> - $\mathbf{X}_t$ — зашумленный тензор на шаге $t \in \{1, \dots, T\}$
> 
> - $\beta_t \in (0, 1)$ — расписание дисперсии
> 
> - $\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i$
> 
> - $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ — истинный гауссовский шум
> 
> - $\boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \boldsymbol{c})$ — шум, предсказанный сетью
> 
> Расчет шага для вектора $\mathbf{x}_0$ и условия $\boldsymbol{c}$:
> 
> 1. Зашумление на шаге $t$:
> 
>    $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$
> 
> 2. Предсказание шума:
> 
>    $$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \boldsymbol{c})$$
> 
> 3. Среднее обратного шага:
> 
>    $$\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}} \right)$$

---

## Обучение

[[оптимизация|Оптимизация]] параметров сети $\boldsymbol{\epsilon}_\theta$ минимизацией MSE между истинным и предсказанным шумом [[sgd|методом обратного распространения ошибки]].

Упрощенный функционал потерь:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{X}_0, \boldsymbol{\epsilon}, \boldsymbol{c}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \, t, \, \boldsymbol{c}\right) \right\|^2 \right]$$

| Свойство                   | Описание                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Обучаемые параметры        | Веса сети $\boldsymbol{\epsilon}_\theta$ (блоков U-Net / DiT и проекторов условий)                                       |
| Гиперпараметры             | Число шагов $T$, расписание $\beta_t$, архитектура, вероятность сброса $p_{\text{uncond}}$                               |

---

## Инференс

Последовательное сэмплирование от $\mathbf{X}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ до $t = 1$:

$$\mathbf{X}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{X}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{X}_t, t, \boldsymbol{c}) \right) + \sigma_t \boldsymbol{z}, \quad \boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
