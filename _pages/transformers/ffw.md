---
layout: page
title: Redes hacia delante
description: El modelo básico de computación neuronal
date: 2023-03-14
permalink: /materials/transformers/ffw

nav: false
---

{% include figure.html path="assets/img/transformers/math-triangle.png" title="ai-generated image" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Nota: este capítulo es parte de la serie "[Un recorrido peso a peso por el transformer][guia-transformer]", donde se presenta una guía para aprender cómo funcionan las redes neuronales que procesan textos y cómo se programan.
{: .small-text}

[guia-transformer]: ../transformers


## Redes feed-forward

<i class="fas fa-clock"></i> 3 horas

Tras las introducciones anteriores, estamos ahora preparados para abordar el estudio del modelo básico de red neuronal conocido como red neuronal hacia delante (*feed-forward neural network*) en el capítulo "[Neural Networks and Neural Language Models][neural]" [<i class="fas fa-file"></i>][neural]. En la mayoría de las tareas actuales de procesamiento del lenguaje natural no usaremos arquitecturas tan sencillas, pero las redes *feed-forward* aparecen como componente en los modelos avanzados basados en la arquitectura *transformer* que veremos más adelante.

Todo el capítulo es relevante, aunque probablemente se hable de cosas que ya conoces. Puedes saltar o leer con menos detenimiento las secciones 7.6.3 ("Computation Graphs") y 7.6.4 ("Backward differentiation on computation graphs").

[neural]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/7.pdf


## Anotaciones al libro

Es recomendable que estudies estos comentarios después de una primera lectura del capítulo y antes de la segunda lectura.

#### Apartado


## Implementación en PyTorch

Una implementación de un [modelo de lengua][pylm] [<i class="fas fa-file"></i>][pylm] con redes *feedforward* como el que se ha visto en el capítulo de redes *feedforward*. La implementación coincide con la del artículo "[A neural probabilistic language model](https://dl.acm.org/doi/10.5555/944919.944966)" de 2003, que puedes consultar si necesitas más información.

[pylm]: https://github.com/jaspock/me/blob/master/assets/code/transformers/ff-neural-lm.py
