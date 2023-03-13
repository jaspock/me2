---
layout: page
title: Regresión logística
description: Conociendo al hermano pequeño de las redes neuronales
date: 2023-03-14
permalink: /materials/transformers/regresor

nav: false
---

<!--
   The theme distillpub sometimes uses katex for math rendering. In that case, equations look ugly in dark mode as
   they use a dark color. Styles on custom.scss won't work because the equation is rendered inside a web component
   that sets an inner color. To have the color of the equation match the theme, we need to use the following hack. 
   In file assets/js/distillpub/template.v2.js, replace the substring setting properties for span.kate to
   color: var(--global-text-color). The color is also set in transforms.v2.js but it seems that it's not 
   necessary to change it there.
-->

{% include figure.html path="assets/img/transformers/book-blue.png" title="ai-generated image" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Nota: este capítulo es parte de la serie "[Un recorrido peso a peso por el transformer][guia-transformer]", donde se presenta una guía para aprender cómo funcionan las redes neuronales que procesan textos y cómo se programan.
{: .small-text}

[guia-transformer]: ../transformers

## Regresores

<i class="fas fa-clock"></i> 3 horas

Nuestros primeros pasos por el camino de baldosas amarillas nos llevan al hermano pequeño de la familia de redes neuronales: el regresor logístico, al que, en general, ni siquiera se considera una red neuronal. Por el camino, aprenderás muchos elementos del aprendizaje automático que luego seguirán aplicándose a modelos más complejos. Comienza, por tanto, con el capítulo "[Logistic Regression][logistic]" [<i class="fas fa-file"></i>][logistic]

[logistic]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/5.pdf

Casi la totalidad del capítulo es muy relevante para nuestros propósitos al tratarse de un capítulo introductorio, pero puedes saltarte lo siguiente:

- La introducción que hay antes del apartado 5.1, ya que se basa en elementos de capítulos anteriores (clasificador bayesiano) que no has visto.
- El apartado 5.2.2 ("Other classification tasks and features").
- El apartado 5.2.4 ("Choosing a classifier").
- La sección 5.7 ("Regularization").
- Por último, no es necesario que comprendas la sección 5.10 ("Advanced: Deriving the Gradient Equation") antes de pasar a capítulos posteriores, pero es muy recomendable que seas capaz de derivar por ti mismo la función de pérdida más pronto que tarde. Cuando los modelos se vayan haciendo más complejos obtener la derivada manualmente será una tarea ardua e innecesaria, pero hacerlo ahora mejorará tu perspectiva del entrenamiento de redes neuronales.

Un aspecto básico del entrenamiento de redes neuronales es el principio de estimación por máxima verosimilitud. La explicación del capítulo sobre este método puede complementarse con este [breve tutorial][tutorialmle] [<i class="fas fa-file"></i>][tutorialmle]

[tutorialmle]: https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/08/19/01-Maximum-likelihood-estimation.html

Este capítulo es probablemente el más complejo de todos y el que ofrece una mayor curva de aprendizaje.


## Regresores implementados en PyTorch

Estas son algunos implementaciones en PyTorch de los regresores que hemos estudiado en unas pocas decenas de líneas de código (normalmente menos de 100). Asegúrate de que terminas entendiendo el código lo suficiente como para sentirte con ánimo de poder modificarlo para adaptarlo a otras necesidades.

- Un [regresor logístico][pylog] [<i class="fas fa-file"></i>][pylog] que clasifica muestras bidimensionales sintéticas en dos clases. Se usan solo los elementos más básicos de PyTorch para poder tener una implementación lo más detallada posible. Como ejercicio, puedes hacer una traza y analizar qué tamaños tienen los tensores. Puedes jugar también con el número de pasos de entrenamiento  y la tasa de aprendizaje para ver cómo evoluciona el entrenamiento. Explora diversas posiciones de los centros de las clases y de la dispersión de los datos alrededor de estos y observa cómo cambia la frontera de decisión. Elimina el sesgo (*bias*) de las ecuaciones y observa cómo se restringe la forma de la frontera de decisión al obligar a esta a pasar por el origen de coordenadas.
- Un [regresor softmax para clasificar imágenes de dígitos][pysoft] [<i class="fas fa-file"></i>][pysoft]. Las imágenes y etiquetas de los dígitos se toman de un conjunto de datos muy conocido llamado MNIST. Como ejercicio, puedes simplificar este código para que realice una tarea de clasificación de sentimiento sobre un conjunto de datos sintéticos muy pequeño que se defina explícitamente en el propio programa; puedes inspirarte en el código de los siguientes ejemplos para ello.

Si no lo has hecho ya, puedes empezar a aprender Python y PyTorch siguiendo el [capítulo][cappy] correspondiente de esta serie.

[cappy]: ../python
[pylog]: https://github.com/jaspock/me/blob/master/assets/code/transformers/logistic-regressor.py
[pysoft]: https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/softmax-regressor.py

#### Actividad práctica 

Utiliza las herramientas de depuración como se explicaron [aquí][debug] para explorar paso a paso el código de los regresores. Analiza el valor y tipo de las variables, así como la forma de cada tensor y asegúrate de que entiendes qué representa cada dimensión.

[debug]: pytorch.html#depuración

## Broadcasting en PyTorch

Observa que en la ecuación 5.12 el vector $$\mathbf{b}$$ se obtiene copiando repetidamente el valor escalar $$b$$. Cuando ecuaciones como esta se implementan en PyTorch, no es necesario hacer esta copia explícita gracias al mecanismo de *broadcasting* que se activa automáticamente en algunas ocasiones cuando se combinan tensores de tamaños en principio incompatibles:

{% highlight python %}
import torch 
b = 10
X = torch.tensor([[1,2,3],[4,5,6]])
w = torch.tensor([-1,0,1])
yp = torch.matmul(X,w) + b
{% endhighlight %}


## Entropía

Considera el caso en que el que un suceso $$x$$ puede ocurrir con una probabilidad $$p_x$$ modelada mediante una determinada distribución de probabilidad $p$. Supongamos que queremos calcular la cantidad de información $$I(x)$$ de dicho suceso o, en palabras más sencillas, la *sorpresa* que nos produciría que este suceso tuviera lugar. Como primera aproximación, es fácil ver que la inversa de la probabilidad, $$1/p_x$$, da un valor mayor cuando la probabilidad es pequeña (nos sorprende más que ocurra algo improbable) y un valor menor cuando la probabilidad es mayor (nos sorprende poco que ocurra algo muy probable).

Además, parece lógico que la cantidad de información de un suceso seguro (con probabilidad 1) sea 0. Para conseguirlo, dado que el logaritmo es una función monótona creciente, podemos aplicar el logaritmo al cociente anterior sin que cambie el orden relativo de dos sucesos con diferentes probabilidades:

$$
I(x) = \log\left(\frac{1}{p_x}\right) = - \log (p_x)
$$

La cantidad de información se mide en bits si el logaritmo es en base 2 y en *nats* si es en base $$e$$. La entropía $$H$$ de la distribución de probabilidad es una medida de la información promedio de todos los posibles sucesos. Para obtenerla basta con ponderar la información de cada suceso por su probabilidad y sumar sobre todos los sucesos:

$$
H(p) = - \sum_{x} p_x \log(p_x)
$$

Comprueba que la entropía es máxima si todos los sucesos son equiprobables. La entropía cruzada entre dos distribuciones de probabilidad mide la sorpresa que nos provoca un determinado suceso si usamos una distribución de probabilidad $q$ alternativa a la probabilidad real $p$:

$$
H(p,q) = - \sum_{x} p_x \log(q_x)
$$

Puedes ver, por lo tanto, que la fórmula 5.21 del libro coincide con la ecuación anterior: maximizar la verosimilitud respecto a los parámetros del modelo es equivalente a minimizar la entropía cruzada $$H(y,\hat{y})$$.


## Ejercicios de repaso

Estos ejercicios te permitirán repasar los conceptos más importantes de este capítulo.

1. La disposición de los elementos en matrices y vectores puede ser diferente a la utilizada en la sección 5.2.3. Lo realmente importante es que se realice el productor escalar de cada una de las $$m$$ muestras con el vector de pesos $$\mathbf{w}$$. Indica qué tamaños deberían tener las matrices y vectores si en lugar de una ecuación como la 5.14, usamos una de la forma $$\mathbf{y} = \mathbf{w} \mathbf{X} + \mathbf{b}$$.
2. Calcula la derivada de la función de coste respecto al umbral $$b$$. Si te basas en la derivada de la función de coste respecto a los pesos $$w$$, que está calculada en el libro, llegarás rápido a la solución.
3. Tras entrenar un regresor logístico, le aplicamos una entrada $$\mathbf{x}$$ y calculamos la derivada $$\partial \hat{y} / \partial \mathbf{x}_i$$ para un cierto $$i$$. ¿Qué mide esta derivada? Piensa en el concepto básico de la derivada y en cómo mide la *sensibilidad* del valor de un función respecto a un cambio en una de sus variables.
