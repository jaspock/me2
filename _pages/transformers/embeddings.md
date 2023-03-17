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

Comienza leyendo la sección 6.2. Después, estudia la sección 6.4 sobre la similitud del coseno. Sáltate completamente las secciones 6.5 a 6.7. Céntrate, sobre todo, en la sección 6.8 ("Word2vec"), pero lee con menos detenimiento las secciones 6.9 a 6.12.

[embeddings]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/6.pdf

#### Apartado 6.2

Los embeddings permiten representar palabras con números de forma que palabras que aparecen en contextos similares tengan representaciones similares. Esta representación numérica es esencial para poder hacer que las redes neuronales (u otros modelos de aprendizaje automático) puedan trabajar con el lenguaje humano. Si dada una frase podemos representar cada una de sus palabras mediante un vector, podemos entonces intentar aprender un clasificador que, por ejemplo, determine el sentimiento de la frase a partir de de la suma de los embeddings de sus palabras.

Aunque a lo largo de las últimas décadas se han desarrollado muchos métodos para obtener embeddings, algoritmos como el *word2vec* que estudiaremos en este capítulo fueron pioneros al permitir obtener representaciones *profundas* (en el sentido de que cada palabra se representa con un vector de cientos o miles de números reales), que mejoraron sustancialmente los resultados en numerosas tareas de procesamiento del lenguaje natural. Una característica de estos embeddings que hacen que no sean los más usados actualmente es que aprenden representaciones no contextuales, es decir, que una palabra como *bajo* con sus múltiples sentidos (búscala en el diccionario) se representa de la misma manera en todos los contextos. Como veremos más adelante, los embeddings contextuales como los obtenidos con los transformers ofrecen una representación única para cada palabra en cada oración en la que aparece.

La idea de por qué vectores de mayor tamaño permiten acercar y alejar de forma más adecuada las palabras en el espacio vectorial en base a su similitud se puede entender con el siguiente ejemplo. Supón que solo tuviéramos 2 dimensiones para representar palabras aparentemente con baja similitud como *fish*, *Mars*, *pencil* o *hat*. Si asumimos el cuadrado de lado 1.5 como espacio de trabajo, dado que estas palabras están poco relacionadas, tiene sentido llevarlas a las esquinas.

{% include figure.html path="assets/img/transformers/embeddings-sq1" title="words representations" class="img-fluid rounded z-depth-1" width="364px" height="256px" %}

Si ahora quisiéramos encontrar una representación bidimensional para *Sunday* podríamos colocarla cerca de *hat* (a fin de cuentas, tradicionalmente la gente ha reservado para el domingo sus mejores sombreros), lo cual de paso aleja a *Sunday* de *Mars* o *pencil* con las que es más difícil que comparta contextos. 

{% include figure.html path="assets/img/transformers/embeddings-sq2" title="words representations" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Ahora supongamos que queremos añadir a nuestra lista la palabra *small*. Dado que esta palabra puede acompañar a *fish*, *hat* o *pencil* y dado que Marte (si lo consideramos como un planeta, obviando que puede ser otras cosas como un dios, una marca de chocolatinas o parte del nombre de un famoso jardín parisino) es uno de los planetas más pequeños del sistema solar, podemos colocar *small* justo a mitad de camino de estas cuatro palabras. Sin embargo, esto también la acerca a *Sunday* con la que parece compartir pocos contextos.

{% include figure.html path="assets/img/transformers/embeddings-sq3" title="words representations" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

¿Te vas dando cuenta del problema? Es como si no hubiera espacio suficiente para cumplir con todas las restricciones. Una posible solución temporal es intercambiar *Sunday* y *hat*, pero podemos intuir que conforme vayamos añadiendo más palabras al pequeño espacio vectorial, el problema se irá haciendo más y más evidente.

{% include figure.html path="assets/img/transformers/embeddings-sq4" title="words representations" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Si ahora quisiéramos añadir *Phobos*, evidentemente debería situarse cerca de *Mars* al ser un satélite de este planeta, pero eso la acercaría a palabras como *hat* y es difícil encontrar frases en las que se hable de lo usos del sombrero en el satélite Phobos. La clave está en poder añadir una tercera dimensión al espacio vectorial, lo cual nos permitiría separar *Phobos* del resto de palabras excepto *Mars*.

{% include figure.html path="assets/img/transformers/embeddings-sq5" title="words representations" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Es fácil deducir que aumentando el número de dimensiones del espacio vectorial, podemos representar las relaciones entre palabras con mayor precisión. Cuando pasemos a cientos o miles de dimensiones, algunas palabras estarán cerca de otras en algunos de las dimensiones, pero no en otras. Observa, en cualquier caso, que hay un problema que no vamos a resolver por ahora: que una palabra como *Mars* con varios sentidos diferentes tenga un único embedding. De hecho, podríamos decir que la mayoría de las palabras tienen sentidos diferentes en cada oración distinta en las que las utilicemos: la palabra *gato* en "El gato está dormido" y "El gato está asustado" no representa exactamente la misma idea de animal, pese a que discreticemos el concepto más profundo que tenemos en nuestra mente cuando queremos hablar del minino en una u otra situación. Este problema de no poder representar la semántica en un caso específico se resolverá más adelante con los embeddings contextuales.

#### Apartado 6.3

Esta sección es opcional, pero puedes echarle un vistazo si quieres conocer enfoques más clásicos para la obtención de embeddings basados en contar co-ocurrencias de palabras en documentos.

#### Apartado 6.4

Este es un apartado muy breve que, aun así, introduce una idea básica: la *similitud* del coseno (no lo llamamos *distancia* porque no cumple con todas las propiedades necesarias) entre dos vectores que representan los embeddings de dos palabras es una medida del parecido entre las palabras correspondientes. Si dos vectores son idénticos, su similitud es 1; si son ortogonales, su similitud es 0; si son opuestos, su similitud es -1. 

Más aún, el producto escalar es proporcional a la similitud del coseno, pero implica menos cálculos, por lo que lo usaremos como una aproximación eficiente. Recordemos que el producto escalar de dos vectores $$\mathbf{a}$$ y $$\mathbf{b}$$ es:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$$

Esta idea de sumar los productos dos a dos de los elementos de dos vectores aparece también en la multiplicación de matrices, por lo que podemos pensar en la multiplicación de dos matrices $$A \times B$$ como la obtención de todos los posibles productos escalares entre las filas de $$A$$ y las columnas de $$B$$. Recuerda esta idea en el tema del transformer cuando usemos productos de matrices para compactar el cálculo de una serie de productos escalares y hacerlo todo de una vez.

Hay un párrafo en el que se menciona que al ser los valores de los vectores de embeddings positivos, el coseno estará siempre entre cero y uno. Ten en cuenta que esta afirmación es para el caso de embeddings obtenidos mediante procedimientos clásicos, que no hemos visto aquí, salvo por el ejemplo de la tabla de este apartado (en la que aparecen palabras como *cherry* o *pie*). En estos casos, los valores de los vectores son positivos porque resultan de contar frecuencias de co-ocurrencia de dos palabras en una colección de documentos. Sin embargo, en el caso de los embeddings de palabras que veremos a continuación, los valores de los vectores pueden ser negativos, por lo que el coseno puede ser menor que cero.  

En cualquier caso, aunque el valor del coseno esté acotado entre -1 y 1, el del producto escalar no lo está. 

#### Apartado 6.5

Este apartado introduce el algoritmo *skip-gram* para el cálculo de embeddings de palabras. Este algoritmo puede verse como una aplicación del regresor logístico binario que ya hemos estudiado, por lo que es recomendable que le des un repaso antes. Vamos a tener una serie de datos positivos (la salida deseada será 1) y una serie de datos negativos (la salida deseada será 0). Los datos positivos son pares de palabras que aparecen en el mismo contexto, mientras que los datos negativos son pares de palabras que no aparecen en el mismo contexto. El objetivo del algoritmo es aprender a predecir si dos palabras aparecen en el mismo contexto o no.

¿Pero si la salida del regresor es un único escalar entre 0 y 1, de dónde salen los embeddings? La idea es que cada palabra va a venir representada a la entrada por un vector diferente de pesos, de manera que la salida del regresor cuantificará cómo de probable es que dos palabras aparezcan en el mismo contexto. Y ya hemos visto que el producto escalar nos sirve para medir la similitud de dos vectores. Pero como este producto escalar no está acotado, le aplicamos la función sigmoide para normalizarlo en este rango.

La principal diferencia con el regresor logístico binario que ya vimos es que allí teníamos un vector de características $$\mathbf{x}$$ que representaba la entrada y un vector de pesos $$\mathbf{w}$$ que representaba los parámetros aprendibles por el algoritmo de descenso por gradiente. El vector de entrada venía determinado por los datos del problema y sus valores no se aprendían. En el caso de *skip-gram*, los dos vectores en juego son parámetros a aprender y estos vienen determinados por las dos palabras que queremos representar en cada momento. Una vez finalizado el entrenamiento, el vector de cada palabra será su vector de embeddings, que podremos usar para representarla en tareas adicionales.

Una de las grandes ventajas de este algoritmo es que se basa en una técnica de aprendizaje *auto supervisada* (*self-supervised*) que no requiere el costoso proceso de etiquetar manualmente los datos. No es que el algoritmo no necesite datos etiquetados, ya que necesita saber cuándo la salida deseada para dos palabras es 0 y cuándo es 1. Pero no necesita que estas etiquetas vengan ya dadas, sino que las puede obtener de forma automática a partir de los datos de entrada. Para ello solo necesitamos una colección de textos (por ejemplo, textos de la Wikipedia o noticias descargadas de internet) y definir una ventana de contexto. La ventana de contexto es un número entero que indica cuántas palabras a la izquierda y a la derecha de la palabra objetivo queremos considerar como contexto. El algoritmo extrae las muestras positivas recorriendo una a una las palabras del texto y considerando como contexto las palabras que hay a su izquierda y a su derecha dentro de la ventana. Las muestras negativas se obtienen de forma aleatoria, seleccionando dos palabras al azar del texto y considerando que no aparecen en el mismo contexto.

En el apartado 6.8.2 se dice que "the noise words are chosen according to their weighted unigram frequency $$p_α(w)$$, where $$α$$ is a weight". Ten en cuenta que la $$w$$ se refiere aquí a una palabra que se va a elegir para una muestra negativa y no tiene que ver que con la palabra objetivo que se denota también con $$w$$. Cuando dice que las palabras ruidosas se escogen en base a su probabilidad de unigramas, se refiere únicamente a que se escogen en base a su frecuencia de aparición en el texto: palabras más frecuentes se escogerán más veces que palabras menos frecuentes para no terminar aprendiendo representaciones de palabras que no aparecen apenas en el texto y de las que no vamos a tener apenas muestras positivas a costa de aprender peores representaciones de las palabras que más nos interesan. Observa también cómo el uso de una alteración no lineal en forma del exponente $$\alpha$$ suaviza ligeramente estas frecuencias para no penalizar en exceso a las palabras poco frecuentes.

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


