---
title: Data augmentation
aliases:
  - Аугментация данных
---
Data Augmentation (аугментация данных) — метод регуляризации, расширяющий обучающую выборку путем применения к объектам случайных трансформаций, сохраняющих их истинные метки.

> [!math]+ Математика
> Пусть $T_{\boldsymbol{\theta}}: \mathcal{X} \to \mathcal{X}$ — оператор трансформации с параметром $\boldsymbol{\theta} \sim p(\boldsymbol{\theta})$.
>
> Обучение сводится к минимизации ожидаемого риска:
>
> $$\min_{\boldsymbol{w}} \mathbb{E}_{(\boldsymbol{x}, y) \sim \mathcal{D}} \left[ \mathbb{E}_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta})} \left[ \mathcal{L}\left(f(T_{\boldsymbol{\theta}}(\boldsymbol{x}); \boldsymbol{w}), y\right) \right] \right]$$
>
> где:
>
> - $\boldsymbol{x} \in \mathcal{X}, y \in \mathcal{Y}$ — объект и его метка из датасета $\mathcal{D}$;
> - $T_{\boldsymbol{\theta}}(\boldsymbol{x})$ — функция трансформации (аффинные преобразования, кроппинг и т. д.);
> - $\mathcal{L}$ — функция потерь, $f(\cdot; \boldsymbol{w})$ — модель с параметрами $\boldsymbol{w}$.

Гиперпараметры: вероятность применения $p_\text{aug}$, тип и диапазоны параметров трансформаций.

---

## Достоинства и недостатки

Достоинства:

- Расширение выборки без сбора и разметки новых данных

Недостатки и ограничения:

- Нагрузка на CPU при генерации трансформаций перед передачей данных на GPU
- Риск искажения истинного класса объекта при чрезмерных трансформациях
