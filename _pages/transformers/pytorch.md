---
layout: page
title: Aprender a programar con PyTorch
description: Estudio de la librería con la que programar los modelos neuronales
date: 2023-02-14
permalink: /materials/transformers/pytorch

nav: false
---
   
{% include figure.html path="assets/img/transformers/book-robot.png" title="ai-generated image" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Nota: este capítulo es parte de la serie "[Un recorrido peso a peso por el transformer][guia-transformer]", donde se presenta una guía para aprender cómo funcionan las redes neuronales que procesan textos y cómo se programan.
{: .small-text}

[guia-transformer]: ../transformers

La librería [PyTorch][pytorch] es junto a [TensorFlow][tf] y [JAX][jax] una de las librerías más populares para programar modelos neuronales a bajo nivel. Dado que nuestro objetivo es comprender y modificar el código del transformer, es necesario que aprendas a usar PyTorch. 

[tf]: https://www.tensorflow.org/
[jax]: https://github.com/google/jax
[pytorch]: https://pytorch.org/


## Python

<i class="fas fa-clock"></i> 2 horas

Python es a día de hoy el lenguaje de programación más usado en el mundo del procesamiento del lenguaje natural. Para usar PyTorch necesitarás conocer los elementos fundamentales del lenguaje Python. Se trata de un lenguaje dinámico, pero probablemente no se parecerá mucho a otros lenguajes de este tipo que conozcas. Pese a ello, no tiene sentido que aprendas Python desde cero, ya que muchos elementos básicos del lenguaje (bucles, funciones, clases, etc.) no se alejan mucho de los que ya conoces para otros lenguajes.

Los cursos de aprendizaje automático suelen incluir una introducción a Python para programadores experimentados en otros lenguajes. En este área es también muy frecuente usar librerías para cálculo científico como NumPy. Aunque nosotros usaremos librerías más específicas como PyTorch, esta comparte muchos principios de diseño con NumPy, por lo que es recomendable que aprendas algo de NumPy también. 

Sigue estos tutoriales. Usa más de una fuente para así aprender más:

- El tutorial "[Python Numpy Tutorial (with Jupyter and Colab)][cs231]" [<i class="fas fa-file"></i>][cs231] del curso "CS231n: Deep Learning for Computer Vision" de Stanford University. Observa que en la parte superior aparece una insignia que dice "Open in Colab". Clicando en ella puedes abrir un entorno de ejecución basado en Python en la nube desde la que ejecutar el código de ejemplo como un cuaderno de Python. 
- Las diapositivas "[Python Review Session][review]" [<i class="fas fa-file"></i>][review] del curso "CS224n: Natural Language Processing with Deep Learning" de Stanford University. También tienes un [cuaderno de Python][cuaderno] [<i class="fas fa-file"></i>][cuaderno] que puedes descargar y subir a Google Colab.

[cs231]: https://cs231n.github.io/python-numpy-tutorial/
[review]: https://web.stanford.edu/class/cs224n/readings/cs224n-python-review.pdf
[cuaderno]: https://web.stanford.edu/class/cs224n/readings/python_tutorial.ipynb

Cuando necesites profundizar o entender ciertas estructuras de Python, puedes consultar la [documentación oficial][oficial] de Python.

[oficial]: https://docs.python.org/3.10/tutorial/index.html


## Cuadernos

<i class="fas fa-clock"></i> 1 hora

El uso de los cuadernos de Jupyter y de Google Colab se explica en los apartados 20.1 ("[Using Jupyter Notebooks][notebooks]" [<i class="fas fa-file"></i>)][notebooks] y 20.4 ("[Using Google Colab][colab]" [<i class="fas fa-file"></i>][colab]) del libro de "Dive into Deep Learning".

[notebooks]: https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html
[colab]: https://d2l.ai/chapter_appendix-tools-for-deep-learning/colab.html


## PyTorch

<i class="fas fa-clock"></i> 3 horas

PyTorch no es en principio una librería sencilla de entender. Al estudiar de primeras el código de un modelo neuronal programado en PyTorch, es posible que no entiendas ciertas partes o que no seas capaz de deducir todo el comportamiento implícito subyacente. Es por ello que un esfuerzo inicial de estudio de la librería es necesario. 

Puedes hacerte una idea somera de los fundamentos de PyTorch siguiendo la breve [introducción a PyTorch][intro] [<i class="fas fa-file"></i>][intro] que se incluye como parte del libro "Dive into Deep Learning"; observa que puedes ver los ejemplos de PyTorch en la página web, pero también abrir un cuaderno en Colab o en SageMaker Studio Lab. Sin embargo, como se ha mencionado, necesitarás ahondar más en los entresijos de la librería: sigue para ello el tutorial en vídeo de más de 2 horas de esta [playlist oficial de PyTorch][playlist]. [<i class="fas fa-file"></i>][playlist] Mírate al menos los 4 primeros vídeos ("Introduction to PyTorch", "Introduction to PyTorch Tensors", "The Fundamentals of Autograd" y "Building Models with PyTorch").

Como complemento a lo anterior, puedes consultar también el [tutorial oficial][tutoficial] de PyTorch. Asegúrate de seleccionar tu versión de PyTorch en la esquina superior izquierda. Es especialmente didáctico este corto [tutorial con ejemplos sencillos] de ajuste de la función $a +bx + cx^2 + dx^3$. Finalmente, cuando necesites profundizar en algún elemento particular de PyTorch, puedes recurrir a la [documentación oficial de PyTorch][docutorch].

[intro]: https://d2l.ai/chapter_preliminaries/ndarray.html
[tutoficial]: https://pytorch.org/tutorials/beginner/basics/intro.html
[docutorch]: https://pytorch.org/docs/stable/index.html
[tutorial con ejemplos sencillos]: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
[playlist]: https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN


## Depuración

Depurar es una de las mejores maneras de comprender todos los secretos del código que estás estudiando. Es importante que te sientas cómodo con el proceso de depuración. A continuación, se dan algunos consejos para depurar código en Python usando el entorno de VS Code, pero instrucciones similares son válidas para otros entornos de desarrollo.

Para depurar el fichero actual, coloca uno o más puntos de interrupción en el código clicando en la pequeña columna que hay a la izquierda de los números de línea hasta que aparezca una marca roja. Luego, asegúrate de que tienes el entorno de ejecución de Python adecuado seleccionado en la barra inferior y ejecuta `Run/Start Debugging` (`F5`) en el menú. El código se ejecutará hasta que se alcance el punto de interrupción. Puedes hacer que el código se ejecute paso a paso usando `Run/Step Over` o `Run/Step Into`, pero ganarás mucho tiempo si te aprendes los atajos de teclado correspondientes (`F10`y `F11`, respectivamente). Además de inspeccionar las variables desde la interfaz, te será muy útil acceder a la paleta de órdenes (`Help/Show All Commands`o bien `Ctrl+Shift+P`) y mostrar el terminal de depuración tecleando `Debug Console`. En este terminal puedes ejecutar cualquier código Python para, por ejemplo:

- ver el tipo de una variable: ```type(learning_rate)```
- ver el contenido de un tensor: ```x```
- ver la forma de un tensor: ```x.shape```
- ver la salida de una función ya definida escribiendo líneas como ```sigmoid(torch.tensor([-10.2,0.1,10.2]))``` o ```torch.logical_not(mask[:10])```

Si el tensor es grande, suele ser suficiente para nuestros propósitos ver solo unos cuantos de sus elementos, lo que puedes hacer mediante indexación como en ```x[:10]```. Para que sistemáticamente PyTorch muestre solo los primeros elementos de cada dimensión y use una cantidad limitada de decimales puedes hacer una sola vez:

```python
torch.set_printoptions(precision=4, edgeitems= 4, threshold=30)
```

## Implementaciones adicionales

El proyecto [MinT][MinT] incluye diferentes tutoriales con implementaciones desde cero de modelos tipo BERT, GPT, BART o T5. El código es ligeramente más extenso que el que hemos estudiado, pero puede servir para afianzar conocimientos en una fase avanzada. El proyecto [x-transformers] sigue un enfoque similar.

Existe cierto pique entre los desarrolladores por conseguir una implementación del transformer lo más compacta posible. Algunas de ellas son [minGPT][mingpt], [nanoGPT][nanogpt] y [picoGPT][picogpt]. Un aspecto destacable de estas es que son capaces de cargar los pesos de GPT-2 y realizar inferencia con ellos. Andrej Karpathy, el desarrollador de minGPT y nanoGPT tiene un [vídeo][video] muy pedagógico en el que explica el funcionamiento de su implementación.

[MinT]: https://github.com/dpressel/mint
[x-transformers]: https://github.com/lucidrains/x-transformers

[mingpt]: https://github.com/karpathy/minGPT
[nanogpt]: https://github.com/karpathy/nanoGPT
[picogpt]: https://github.com/jaymody/picoGPT
[video]: https://youtu.be/kCc8FmEb1nYç
