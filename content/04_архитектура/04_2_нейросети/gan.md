---
title: Generative adversarial network
aliases:
  - Генеративно-состязательные сети
  - GAN
---
Генеративно-состязательная сеть (GAN) — архитектура из двух параметрических сетей (генератора и дискриминатора), соревнующихся в антагонистической игре.

| Свойство     | Описание                                                                                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Вход         | Шум $\boldsymbol{z} \in \mathbb{R}^{d_z}$ (генератор) и объекты $\boldsymbol{x} \in \mathbb{R}^{d_x}$ (дискриминатор)                               |
| Выход        | Синтетический объект $\hat{\boldsymbol{x}} \in \mathbb{R}^{d_x}$ и вероятность $D(\boldsymbol{x}) \in [0, 1]$                                      |

> [!math]+ Математика
> Модели:
> 
> - Генератор: $G_{\boldsymbol{\theta}}: \mathbb{R}^{d_z} \to \mathbb{R}^{d_x}$
> - Дискриминатор: $D_{\boldsymbol{\phi}}: \mathbb{R}^{d_x} \to [0, 1]$
> 
> Минимаксная целевая функция:
> 
> $$\min_G \max_D V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z} \sim p_z(\boldsymbol{z})}[\log(1 - D(G(\boldsymbol{z})))]$$
> 
> Функция потерь генератора (Non-Saturating):
> 
> $$\mathcal{L}_G = -\mathbb{E}_{\boldsymbol{z} \sim p_z(\boldsymbol{z})}[\log D(G(\boldsymbol{z}))]$$
> 
> Обозначения:
> 
> - $\boldsymbol{x} \sim p_{\text{data}}(\boldsymbol{x})$ — реальный объект;
> - $\boldsymbol{z} \sim p_z(\boldsymbol{z})$ — шум из $\mathcal{N}(\mathbf{0}, \mathbf{I})$;
> - $G(\boldsymbol{z}) = \hat{\boldsymbol{x}}$ — сгенерированный объект;
> - $D(\boldsymbol{x}) \in [0, 1]$ — оценка реальности.
> 
> Итерация для батча размера $B$:
> 
> 1. Генерация:
> 
>    $$\hat{\boldsymbol{x}}_i = G_{\boldsymbol{\theta}}(\boldsymbol{z}_i), \quad i = 1, \dots, B$$
> 
> 2. Оценки:
> 
>    $$d_{\text{real}, i} = D_{\boldsymbol{\phi}}(\boldsymbol{x}_i), \quad d_{\text{fake}, i} = D_{\boldsymbol{\phi}}(\hat{\boldsymbol{x}}_i)$$
> 
> 3. Потери:
> 
>    $$\mathcal{L}_D = -\frac{1}{B} \sum_{i=1}^B \left( \log d_{\text{real}, i} + \log(1 - d_{\text{fake}, i}) \right), \quad \mathcal{L}_G = -\frac{1}{B} \sum_{i=1}^B \log d_{\text{fake}, i}$$

---

## Обучение

Поочередная настройка $\boldsymbol{\phi}$ и $\boldsymbol{\theta}$ до равновесия Нэша ($p_g = p_{\text{data}}$ и $D(\boldsymbol{x}) = 0.5$).

Цикл:

1. Шаг $D$: [[оптимизация|максимизация]] точности разделения реальных и синтетических объектов.
2. Шаг $G$: [[оптимизация|минимизация]] вероятности распознавания синтетических объектов.

| Свойство                   | Описание                                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| Обучаемые параметры        | Веса $G$ ($\boldsymbol{\theta}$) и $D$ ($\boldsymbol{\phi}$)                                             |
| Гиперпараметры             | Размерность $d_z$, шаги $k$, темпы обучения $\eta_G, \eta_D$, тип loss-функции                           |

---

## Инференс

Генерация объекта путем прохода шума $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ через $G$:

$$\hat{\boldsymbol{x}} = G(\boldsymbol{z})$$

Дискриминатор не используется.
