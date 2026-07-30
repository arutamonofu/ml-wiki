---
title: Autoencoder
aliases:
  - Автокодировщик
---
Автоэнкодер (Autoencoder, AE) — нейросетевая архитектура для обучения признакам без учителя, реализующая сжатие и восстановление данных:

1. Сжатие (Энкодер $e_{\boldsymbol{\theta}}$): Вход $\boldsymbol{x} \in \mathbb{R}^{d_{\text{in}}}$ отображается в латентный вектор $\boldsymbol{z} \in \mathbb{R}^{d_z}$ ($d_z \ll d_{\text{in}}$).
2. Восстановление (Декодер $d_{\boldsymbol{\phi}}$): Вход реконструируется из латентного вектора: $\hat{\boldsymbol{x}} = d_{\boldsymbol{\phi}}(\boldsymbol{z}) \in \mathbb{R}^{d_{\text{in}}}$.

| Свойство     | Описание                                                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Вход         | Вектор признаков $\boldsymbol{x} \in \mathbb{R}^{d_{\text{in}}}$ или пространственно-временной тензор $\mathbf{X}$                       |
| Выход        | Реконструированный вход $\hat{\boldsymbol{x}} \in \mathbb{R}^{d_{\text{in}}}$ или латентный вектор $\boldsymbol{z} \in \mathbb{R}^{d_z}$ |

> [!math]+ Математика
> Оператор реконструкции: $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{in}}}$
> 
> $$\hat{\boldsymbol{x}} = f(\boldsymbol{x}) = d_{\boldsymbol{\phi}}(e_{\boldsymbol{\theta}}(\boldsymbol{x}))$$
> 
> Энкодер $e_{\boldsymbol{\theta}}: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_z}$:
> 
> $$\boldsymbol{z} = e_{\boldsymbol{\theta}}(\boldsymbol{x})$$
> 
> Декодер $d_{\boldsymbol{\phi}}: \mathbb{R}^{d_z} \to \mathbb{R}^{d_{\text{in}}}$:
> 
> $$\hat{\boldsymbol{x}} = d_{\boldsymbol{\phi}}(\boldsymbol{z})$$
> 
> Вариационный автоэнкодер (VAE, $e_{\boldsymbol{\theta}}(\boldsymbol{x})$ возвращает параметры распределения):
> 
> $$\boldsymbol{\mu}_{\boldsymbol{z}}, \log\boldsymbol{\sigma}_{\boldsymbol{z}}^2 = e_{\boldsymbol{\theta}}(\boldsymbol{x}), \quad \boldsymbol{\sigma}_{\boldsymbol{z}} = \exp\left(\frac{1}{2}\log\boldsymbol{\sigma}_{\boldsymbol{z}}^2\right), \quad \boldsymbol{z} = \boldsymbol{\mu}_{\boldsymbol{z}} + \boldsymbol{\sigma}_{\boldsymbol{z}} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
> 
> Обозначения:
> 
> - $\boldsymbol{x} \in \mathbb{R}^{d_{\text{in}}}$ — входной вектор признаков;
> - $\boldsymbol{z} \in \mathbb{R}^{d_z}$ — латентный вектор узкого места ($d_z \ll d_{\text{in}}$);
> - $\hat{\boldsymbol{x}} \in \mathbb{R}^{d_{\text{in}}}$ — реконструированный вход;
> - $\boldsymbol{\theta}, \boldsymbol{\phi}$ — параметры энкодера и декодера;
> - $\boldsymbol{\mu}_{\boldsymbol{z}}, \boldsymbol{\sigma}_{\boldsymbol{z}} \in \mathbb{R}^{d_z}$ — среднее и стандартное отклонение в VAE;
> - $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ — случайный шум.
> 
> Расчет для одномерного слоя с активацией $\sigma$:
> 
> 1. Энкодер:
> 
>    $$\boldsymbol{z} = \sigma(\mathbf{W}_e \boldsymbol{x} + \boldsymbol{b}_e) \in \mathbb{R}^{d_z}$$
> 
> 2. Декодер:
> 
>    $$\hat{\boldsymbol{x}} = \sigma(\mathbf{W}_d \boldsymbol{z} + \boldsymbol{b}_d) \in \mathbb{R}^{d_{\text{in}}}$$
> 
> 3. Ошибка реконструкции (MSE):
> 
>    $$\mathcal{L}_{\text{MSE}}(\boldsymbol{x}, \hat{\boldsymbol{x}}) = \frac{1}{d_{\text{in}}} \|\boldsymbol{x} - \hat{\boldsymbol{x}}\|^2$$

---

## Обучение

[[оптимизация|Оптимизация]] параметров $\boldsymbol{\theta}$ и $\boldsymbol{\phi}$ минимизацией функции потерь (MSE, BCE) между $\boldsymbol{x}$ и $\hat{\boldsymbol{x}}$ [[sgd|методом обратного распространения ошибки]].

В VAE оптимизируется нижняя вариационная оценка правдоподобия (ELBO):

$$\operatorname{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \mathbb{E}_{q_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})}[\log p_{\boldsymbol{\phi}}(\boldsymbol{x} \mid \boldsymbol{z})] - D_{\text{KL}}\left(q_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}) \,\|\, p(\boldsymbol{z})\right)$$

$$\mathcal{L}_{\text{VAE}}(\boldsymbol{\theta}, \boldsymbol{\phi}) = -\operatorname{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi})$$

| Свойство                   | Описание                                                                                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Обучаемые параметры        | Веса и смещения энкодера $\boldsymbol{\theta}$ и декодера $\boldsymbol{\phi}$                                                                        |
| Гиперпараметры             | Размерность латентного пространства $d_z$, архитектуры подсетей, весовой коэффициент $\beta$, тип функции потерь                                     |


---

## Инференс

Расчет для объекта $\boldsymbol{x}_{\text{new}} \in \mathbb{R}^{d_{\text{in}}}$ или случайного вектора $\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

Режимы работы:

1. Извлечение представлений:
   $$\boldsymbol{z} = e_{\boldsymbol{\theta}}(\boldsymbol{x}_{\text{new}})$$

2. Реконструкция и детекция аномалий:
   $$\hat{\boldsymbol{x}} = d_{\boldsymbol{\phi}}(e_{\boldsymbol{\theta}}(\boldsymbol{x}_{\text{new}}))$$

3. Генерация (для VAE / VQ-VAE):
   $$\hat{\boldsymbol{x}} = d_{\boldsymbol{\phi}}(\boldsymbol{z})$$
