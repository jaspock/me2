---
layout: page
title: Un recorrido peso a peso por el transformer
description: Guía para aprender cómo funcionan las redes neuronales que procesan textos y cómo se programan
img: assets/img/transformers/math-mandelbrot.png
importance: 1
category: neural
nav: false
authors:
  - name: Juan Antonio Pérez Ortiz
    url: "https://cvnet.cpd.ua.es/curriculum-breve/es/perez-ortiz-juan-antonio/15404"
    affiliations:
      name: Universitat d'Alacant

bibliography: 2022-09-22-guia.bib
---

<!--
<img src="/assets/img/transformers/math-mandelbrot.png" alt="Recreación del transformer" width="256px" height="256px" class="rounded">
-->

{% include figure.html path="assets/img/transformers/math-mandelbrot.png" title="ai-generated image" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}


## Contenidos

- [Introducción](#introducción)
- [Regresión logística](regresor.html)
- [Embbedings incontextuales](embeddings.html)
- [Redes hacia delante](ffw.html)
- [Transformers y modelos de atención](attention.html)

- [Aprender a programar con PyTorch](pytorch.html)
- [Apuntes de PyTorch](apuntes.html)


## Introducción

Esta guía propone una camino para entender cómo funciona realmente la red neuronal más usada en el campo del procesamiento del lenguaje natural (conocida como *transformer*). Se siguen para ello las explicaciones teóricas de algunos de los capítulos de un buen libro sobre la materia. Se va aprendiendo sobre la marcha (si es necesario) el lenguaje de programación Python, así como los elementos básicos de una librería llamada PyTorch que permite, entre otras cosas, programar redes neuronales que se entrenen y ejecuten sobre GPUs. A continuación, se estudia una implementación ya existente del transformer programada con PyTorch. El objetivo último es poder modificar este código para experimentar con algún problema sencillo que implique el uso del lenguaje humano. La idea es obtener un buen conocimiento que permita afrontar tareas más complejas más adelante y no tanto desarrollar algo llamativo que enseñar a todo el mundo desde el primer momento.

Algunos aspectos los puedes ir estudiando en paralelo. A la vez que aprendes sobre modelos neuronales, puedes ir iniciándote en [Python](#python), NumPy e incluso, una vez vistos los dos anteriores, [PyTorch](#pytorch). También puedes repasar en paralelo los [elementos de álgebra y cálculo](#conceptos-previos) que hayas olvidado. El estudio del código del transformer no deberías abordarlo hasta tener bien asimilados todos los conceptos anteriores.

Para entender a nivel matemático y conceptual las redes neuronales nos vamos a basar en la tercera edición (todavía inacabada) del libro "[Speech and Language Processing][libroarch]" de Dan Jurafsky y James H. Martin. Los siguientes apartados indican qué capítulos y secciones son relevantes para nuestros propósitos. **Importante**: dado que la versión en línea del libro está inacabada y se actualiza de vez en cuando no solo con nuevos contenidos, sino también con reestructuraciones de los ya existentes y movimientos de secciones de un capítulo a otro, en esta guía se incluyen enlaces y referencias a una [versión del libro alojada en Internet Archive][libroarch] que probablemente no se corresponde con la más actual (que puedes encontrar [aquí][libro]).

[libro]: https://web.stanford.edu/~jurafsky/slp3/
[libroarch]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/

En principio, escribir un programa que explote modelos basados en aprendizaje automático es muy sencillo. Por ejemplo, las siguientes líneas de código usan un modelo transformer para continuar un texto dado:

{% highlight python %}
from transformers import pipeline
generator = pipeline('text-generation', model = 'gpt2')
generator("Hello, I'm a language model and", max_length = 30, num_return_sequences=3)
{% endhighlight %}

Aunque la existencia de librerías de alto nivel es sumamente importante en ciertos contextos, si solo usas el código anterior:

- No entenderás cómo funciona realmente el modelo.
- No podrás crear otros modelos para experimentar con otros problemas.
- No sabrás cómo entrenar un modelo propio ni qué elementos influyen en la calidad o el tiempo del entrenamiento.
- No entenderás otros modelos neuronales que se usan en el procesamiento del lenguaje natural.
- Verás tu programa, en definitiva, como una caja negra que hace cosas mágicas. 

Esta guía pretende ayudarte a abrir la caja y ser capaz de observar su interior con conocimiento de causa.


## Conceptos previos

Los elementos básicos de álgebra, cálculo y probabilidad necesarios para manejarte con soltura en el mundo del procesamiento del lenguaje natural los puedes encontrar en las secciones "Linear Algebra", "Calculus" (junto con "Automatic differentiation") y "Probability and Statistics" del [capítulo 2][cap2] [<i class="fas fa-file"></i>](cap2) del libro "Dive into Deep Learning". Otros como la teoría de la información (19.11) o el principio de máxima verosimilitud (19.7) se abordan en un [apéndice][apéndice] [<i class="fas fa-file"></i>](apéndice) del mismo libro.

[cap2]: https://d2l.ai/chapter_preliminaries/index.html
[apéndice]: https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html


## Para saber más

Para ampliar conocimientos, puedes consultar estos libros que, en su mayoría, están disponibles en la web:

- "[Speech and Language Processing][jurafskybib]" de Dan Jurafsky y James H. Martin. Sin fecha de publicación aún, pero con un borrador avanzado disponible en la web. Explica detalladamente los conceptos y modelos más relevantes en el procesamiento del lenguaje natural sin entrar en detalles de implementación. Es el libro en el que se basa esta guía.
- "[Deep Learning with PyTorch Step-by-Step: A Beginner's Guide][pytorchstep]" (2022) de Daniel Voigt Godoy. Este es un libro de pago con versión digital o en papel (en tres volúmenes, en este caso). Existe una versión en español para los primeros capítulos. Escrito en un estilo directo y sencillo con multitud de detalles y ejemplos.
- "[Dive into Deep Learning][d2l]" de Aston Zhang, Zachary C. Lipton, Mu Li y Alexander J. Smola. Se adentra con mucho detalle en la implementación de los modelos más relevantes del aprendizaje profundo. Hay una versión en [papel][cambridge] publicada en 2023.
- "[Understanding Deep Learning][understanding]" por Simon J.D. Prince. A publicar en 2023. Lleno de imágenes y figuras que ayudan a entender todos los conceptos.
- La serie "[Probabilistic Machine Learning: An Introduction][pml1]" (2022) y "[Probabilistic Machine Learning: Advanced Topics][pml2]" (2023) de Kevin Murphy aborda con más profundidad diversos elementos del aprendizaje automático.
- "[Deep learning for Natural Language Processing: A Gentle Introduction][gentle]" de Mihai Surdeanu y Marco A. Valenzuela-Escárcega. También en elaboración. Contiene código en algunos capítulos.

[jurafskybib]: https://web.stanford.edu/~jurafsky/slp3/
[gentle]: https://clulab.org/gentlenlp.html
[understanding]: https://udlbook.github.io/udlbook/
[pytorchstep]: https://leanpub.com/pytorch
[d2l]: http://d2l.ai/
[pml1]: https://probml.github.io/pml-book/book1.html
[pml2]: https://probml.github.io/pml-book/book2.html
[cambridge]: https://www.cambridge.org/es/academic/subjects/computer-science/pattern-recognition-and-machine-learning/dive-deep-learning?format=PB

La siguiente lista contiene enlaces a algunos cursos en vídeo impartidos por investigadores o universidades reconocidos:

- "[Machine Learning Specialization][mlspec] de Andrew Ng.
- "[Stanford CS229][cs229] - Machine Learning" de Andrew Ng; [web][cs229web] del curso.
- "[Stanford CS230][cs230] - Deep Learning" de Andrew Ng; [web][cs230web] del curso.
- "[Stanford CS224n][cs224n] - Natural Language Processing with Deep Learning" por Christopher Manning; [web][cs224nweb] del curso.
- "[Stanford  CS25][cs25]  - Transformers United"; [web][cs25web] del curso.
- "[MIT 6.S191][mit191] - Introduction to Deep Learning" de Alexander Amini and Ava Soleimany; [web][mit191web] del curso.
- "Neural Networks: [Zero to Hero][zero2hero]" de Andrew Karpathy.

[mlspec]: https://www.youtube.com/playlist?list=PLxfEOJXRm7eZKJyovNH-lE3ooXTsOCvfC
[cs229]: https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
[cs229web]: https://cs229.stanford.edu/
[cs230]: https://youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb
[cs230web]: https://cs230.stanford.edu/
[cs224n]: https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ
[cs224nweb]: http://web.stan  ford.edu/class/cs224n/
[cs25]: https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM
[cs25web]: https://web.stanford.edu/class/cs25/
[mit191]: https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
[mit191web]: http://introtodeeplearning.com
[zero2hero]: https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
