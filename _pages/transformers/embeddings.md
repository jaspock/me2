---
layout: page
title: Embeddings incontextuales
description: Un algoritmo de la familia word2vec que permite representar palabras con números
date: 2023-03-14
permalink: /materials/transformers/embeddings

nav: false
---

{% include figure.html path="assets/img/transformers/engine-heart.png" title="ai-generated image" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Nota: este capítulo es parte de la serie "[Un recorrido peso a peso por el transformer][guia-transformer]", donde se presenta una guía para aprender cómo funcionan las redes neuronales que procesan textos y cómo se programan.
{: .small-text}

[guia-transformer]: ../transformers


## Embeddings

<i class="fas fa-clock"></i> 4 horas

Hasta este momento, a la hora de representar palabras o frases hemos usado características (*features*) que nos han permitido trabajar de forma matemática con elementos lingüísticos. Estas características, sin embargo, son relativamente arbitrarias y exigen un esfuerzo por nuestra parte en su definición. Sin embargo, existe, como vamos a ver ahora, una manera más fundamentada para representar palabras con números que no requiere supervisión humana. Los embeddings incontextuales se explican en el capítulo "[Vector Semantics and Embeddings][embeddings]" [<i class="fas fa-file"></i>][embeddings]

Puedes pasar más o menos rápido por las secciones 6.1 a 6.3, ya que son meramente descriptivas. Lee con detenimiento la sección 6.4 sobre la similitud del coseno. Sáltate directamente las secciones 6.5 a 6.7. Céntrate sobre todo en la sección 6.8 ("Word2vec"). Lee con menos detenimiento las secciones 6.9 a 6.12.

[embeddings]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/6.pdf


## Notación de Einstein

Consideremos el caso en el que tenemos un mini-batch de palabras objetivo representadas por sus embeddings $\mathbf{w}_1,\mathbf{w}_2,\ldots,\mathbf{w}_E$. Para cada palabra objetivo anterior, tenemos una palabra contextual asociada en el conjunto $\mathbf{c}_1,\mathbf{c}_2,\ldots,\mathbf{c}_E$. Para simplificar, no consideramos las muestras negativas, pero el análisis que vamos a hacer es totalmente extensible al caso en el que se incluyan. 

Sea $N$ el tamaño de los embeddings. Queremos calcular el producto escalar de cada $\mathbf{w}_i$ con cada $\mathbf{c}_i$, cálculo este que ya has visto que es fundamental en el entrenamiento y uso de los modelos de skip-grams. Para obtener estos productos escalares usando PyTorch y beneficiarnos de la eficiencia de las operaciones matriciales calculadas en GPUs, podemos empaquetar por filas los embeddings de las palabras objetivo en una matriz $A$ de tamaño $E \times N$ y los embeddings de las palabras contextuales por columnas en una matriz $B$ de tamaño $N \times E$. Si calculamos el producto $A \cdot B$ obtendremos una matriz de tamaño $E \times E$ en la que cada elemento $i,j$ es el producto escalar de $\mathbf{w}_i$ con $\mathbf{c}_j$. 

Sin embargo, nosotros solo estamos interesados en una pequeña parte de todos estos productos escalares. En concreto, aquellos que forman parte de la diagonal del resultado, que serán los de la forma  $\mathbf{w}_i$ $\mathbf{c}_i$. La multiplicación de matrices es muy ineficiente en este caso para nuestros propósitos, pero si buscamos en la documentación de PyTorch no encontraremos en principio una operación que se ajuste exactamente a nuestros intereses. 

Existe, sin embargo, en PyTorch una manera eficiente y compacta de definir operaciones matriciales basada en la notación de Einstein, de la que puedes aprender un poco leyendo hasta el apartado 2.8 aproximadamente del tutorial "[Einsum is all you need](https://rockt.github.io/2018/04/30/einsum)" <i class="fas fa-file"></i>. En particular, podemos observar que nos interesa obtener un vector $\mathbf{d}$ tal que:

$$
\mathbf{d}_i = \mathbf{w}_i \cdot \mathbf{c}_i = \sum_{j} \mathbf{w}_{i,j} \, \mathbf{c}_{j,i}
$$

Usando la notación de Einstein con la función de PyTorch `einsum`, podemos escribir la operación matricial anterior y obtener el tensor unidimensional que queremos como sigue:

```python
d = torch.einsum('ij,ji->i', A, B)
```

## Implementación en PyTorch

Una implementación del algoritmo [skip-gram][pyskip] [<i class="fas fa-file"></i>][pyskip] para la obtención de embeddings incontextuales que sigue las pautas marcadas en el libro de Jurafsky y Martin. La implementación anterior se basa en [otra](https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/skipgrams-original.py) que sigue el enfoque del trabajo "[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)" de 2013, que no se ajusta totalmente al que hemos estudiado nosotros.

[pyskip]: https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/skipgrams-jurafsky.py

No es necesario que entiendas este código antes de pasar al siguiente capítulo del libro, pero recuerda estudiarlo cuando lo hayas hecho. Llegados a este punto, no obstante, va siendo recomendable que comiences a familiarizarte con el uso de PyTorch, quizás en paralelo al estudio del próximo bloque.


## Ejercicios de repaso

1. Al buscar relaciones de analogía entre word embeddings intentamos, dadas las palabras A y B (relacionadas entre ellas), encontrar en el espacio de embeddings dos palabras, C y D (también relacionadas tanto entre ellas como con A y B) para los que tenga sentido afirmar que "A es a B como C es a D". Matemáticamente, se trata de encontrar cuatro palabras cuyos embeddings cumplan ciertas propiedades geométricas basadas en las distancias entre sus vectores. Por ejemplo, es habitual observar que estas propiedades se cumplen si A=man, B=woman, C=king y D=queen (diremos que "man es a woman como king es a queen"). Supongamos que hemos obtenido mediante el algoritmo skip-gram unos embeddings de palabras para el inglés que permiten encontrar este tipo de analogías. Considera la lista de palabras L = {planes, France, Louvre, UK, Italy, plane, people, man, woman, pilot, cloud}. Indica con qué palabra de la lista L tiene más sentido sustituir @1, @2, @3, @4, @5 y @6 en las siguientes expresiones para que se cumplan las analogías:

   - A=Paris, B=@1, C=London, D=@2
   - A=plane, B=@3, C=person, D=@4
   - A=mother, B=father, C=@5, D=@6


