---
layout: page
title: Transformers y modelos de atención
description: Conociendo los modelos de atención y su aplicación en el modelo transformer
date: 2023-03-14
permalink: /materials/transformers/attention

nav: false
---

{% include figure.html path="assets/img/transformers/engine-words.png" title="ai-generated image" class="img-fluid rounded z-depth-1" width="256px" height="256px" %}

Nota: este capítulo es parte de la serie "[Un recorrido peso a peso por el transformer][guia-transformer]", donde se presenta una guía para aprender cómo funcionan las redes neuronales que procesan textos y cómo se programan.
{: .small-text}

[guia-transformer]: ../transformers


## Fundamentos de los transformers

<i class="fas fa-clock"></i> 6 horas

Ahora sí podemos acometer el estudio de los elementos básicos de la arquitectura transformer siguiendo el capítulo "[Deep Learning Architectures for Sequence Processing][basictransformer]" [<i class="fas fa-file"></i>][basictransformer]. Aquí entenderás qué significa una de las ecuaciones más importantes de los últimos años dentro del aprendizaje automático <d-cite key="vaswani-attention-2017"></d-cite>:

$$
\text{Atención}(Q,K,V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) \, V
$$

Salta las secciones 9.2 a 9.6, que se centran en otro modelo alternativo para el procesamiento de secuencias, las redes neuronales recurrentes, que se han venido usando menos en el área del procesamiento del lenguaje natural tras la llegada del transformer.

[basictransformer]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/9.pdf


## Atención escalada

Un factor que puede parecer arbitrario en la ecuación de la atención es la división por la raíz cuadrada de la dimensión de la clave. Para entender la motivación de esta operación, observa que cuanto mayor es el tamaño de los embeddings, mayor es el resultado de cada producto escalar $$q_i k_j$$. El problema es que cuando la función softmax se aplica a valores muy altos, su carácter exponencial hace que asigne valores muy pequeños a todos los elementos excepto al que tiene el valor más alto. Es decir, cuando la función softmax se satura, tiende a un vector *one-hot*. Esto provocará que la atención se centre en un único token e ignore el resto, lo que no es un comportamiento deseable.

Consideremos que $$Q$$ y $$K$$ tienen tamaño $$B \times T \times C$$, donde $$C$$ es el tamaño de las consultas y las claves. Para simplificar, si asumimos que los elementos de las matrices $$Q$$ y $$K$$ tienen varianza alrededor de 1, la varianza de los elementos del producto será del orden de $$C$$. Como se cumple que, dado un escalar $$m$$, $$\mathrm{var}(mX) = m^2 \mathrm{var}(x)$$, al multiplicar cada elemento por $$1 / \sqrt{C}$$, la varianza del producto matricial se reduce en $$\left(1 / \sqrt{C}\right)^2 = 1 / C$$. Por tanto, si la varianza de los elementos de $$Q$$ y $$K$$ es 1, ahora la varianza del producto matricial también estará alrededor de 1. 

El siguiente código permite comprobar los extremos anteriores:

```python
import torch

B, T, C = 10, 10, 5000
m = 1
Q = torch.randn(B, T, C)*m
K = torch.randn(B, T, C)*m

print(f'Variance of Q: {Q.var().item():.2f}')
print(f'Variance of K: {K.var().item():.2f}')
# variances close to m**2

QK = Q.matmul(K.transpose(-2, -1))

print(f'Variance of QK: {QK.var().item():.2f}') 
# very high variance close to C*(m**4)!

s = torch.softmax(QK, dim=-1)
torch.set_printoptions(precision=2)
print(f'Mean value of highest softmax: {s.max(dim=-1)[0].mean().item():.2f}') 
# max value of each channel close to 1

QK = QK / (C ** 0.5)

print(f'Variance of QK after normalization: {QK.var().item():.2f}')
# variance close to m**4

s = torch.softmax(QK, dim=-1)
print(f'Mean value of highest softmax: {s.max(dim=-1)[0].mean().item():.2f}') 
# max value of each channel smaller than 1
```

En general, si la varianza de los elementos de $$Q$$ y $$K$$ es $$m$$, la varianza del producto matricial estará alrededor de $$m^4 C$$. Si $$m=2$$, por ejemplo, la normalización no nos deja elementos con varianza de 1, pero sí la reduce para dejarla en el orden de $$m^4 = 16$$.


## La arquitectura transformer completa

<i class="fas fa-clock"></i> 2 horas

En el bloque anterior, hemos estudiado todos los elementos del transformer, pero en aplicaciones en las que una secuencia de tokens se transforma en otra secuencia de tokens, las capas del transformen se asocian a dos submodelos bien diferenciados y conectados entre ellos mediante mecanismos de atención: el codificador y el descodificador. El capítulo "[Machine translation][mt]" [<i class="fas fa-file"></i>][mt] se centra en la arquitectura completa. En este capítulo solo es necesario que estudies las secciones 10.2 y 10.6.

[mt]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/10.pdf


## Aspectos adicionales sobre el uso de transformers en procesamiento del lenguaje natural

Hay algunos elementos adicionales a la definición de los diferentes modelos neuronales que son también relevantes cuando estos se aplican en el área del procesamiento del lenguaje natural. 

- Estudia el mecanismo de búsqueda en haz (*beam search*) descrito en la sección 10.5.
- Lee lo que se dice sobre la obtención de subpalabras en las secciones 10.7.1 y 2.4.3.


## Modelos preentrenados

<i class="fas fa-clock"></i> 2 horas

El capítulo "[Transfer Learning with Contextual Embeddings and Pre-trained language models][bert]" [<i class="fas fa-file"></i>][bert] estudia los modelos preentrenados y cómo adaptarlos a nuestras necesidades. Este material puede ser opcional o incluso innecesario para ti; consulta con el profesor. En el caso de que abordes su estudio, pueden ser relevantes la introducción y las secciones 11.1 y 11.2. 

[bert]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/11.pdf


## Implementación en PyTorch

Estudiamos aquí dos arquitecturas muy similares:

- Una implementación del [transformer][pytr] [<i class="fas fa-file"></i>][pytr]. La implementación sigue la del artículo "[Attention is all you need](https://arxiv.org/abs/1706.03762)" de 2017, que puedes consultar si necesitas más información. 
- Una implementación sencilla del [modelo BERT][pybert] [<i class="fas fa-file"></i>][pybert].

A estas alturas, ya deberías estar dedicando tiempo a estudiar las implementaciones de los modelos, pero recuerda empezar por las más sencillas de bloques anteriores.

[pytr]: https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/transformer.py
[pybert]: https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/bert.py


## Ejercicios de repaso

1. Argumenta la verdad o falsedad de la siguiente afirmación: si la consulta de un token $$q_i$$ es igual a su clave $$k_i$$, entonces el embedding computado por el mecanismo de autoatención para dicho token coincide con su valor $$v_i$$.

2. La siguiente fórmula para el cálculo de la auto-atención en el descodificador de un transformer es ligeramente distinta a la habitual, ya que se ha añadido explícitamente la máscara que impide que, como hemos visto, la atención calculada para un token durante el entrenamiento tenga en cuenta los tokens que se encuentran posteriormente en la frase:

   $$
   \text{Atención}(Q,K,V,M) = \text{softmax} \left( \frac{Q K^T + M}{\sqrt{d_k}} \right) \, V
   $$

   Indica qué forma tiene la matriz de máscara $$M$$. Busca las operaciones de PyTorch que te permiten inicializar dicha matriz. Si necesitas usar valores de infinito en el código, razona si es suficiente con utilizar un número grande como $$10^9$$.

3. Razona cómo se usaría una máscara $$M$$ similar a la de la fórmula anterior para enmascarar los tokens de relleno de un mini-batch de frases de entrenamiento.

4. Dada una secuencia de tokens de entrada, el codificador (*encoder*) de un transformer permite obtener un conjunto de embeddings para cada token de entrada. Estos embeddings son contextuales en todas sus capas menos en una. ¿En qué capa los embeddings no son contextuales? ¿Cómo se obtienen los valores finales de estos embeddings? ¿Y los valores iniciales?

5. Supongamos que el embedding no contextual almacenado en la tabla de embeddings del codificador de un transformer para un determinado token viene definido por el vector $$\mathbf{e} = \mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$$. Consideremos el caso en el que dicho token aparece en dos posiciones diferentes de una frase de entrada. Si no se utilizaran embeddings posicionales, ¿cuál sería el valor del coseno entre los vectores de los dos embeddings no contextuales usados para el token? Llamemos ahora $$\mathbf{p}$$ y $$\mathbf{p}’$$ a los embeddings posicionales que se utilizan en cada una de las dos apariciones del token. El coseno del ángulo entre los vectores resultantes de sumar al embedding e los embeddings posicionales sería:

   $$
   \cos(\alpha)=\frac{\sum_{i=1}^d(\mathbf{e}_i+\mathbf{p}_i)(\mathbf{e}_i+\mathbf{p}'_i)}{\sqrt{\left(\sum_{j=1}^d(\mathbf{e}_j+\mathbf{p}_j)^2\right)\left(\sum_{j=1}^d(\mathbf{e}_j+\mathbf{p}_j')^2\right)}}
   $$

   ¿A qué valor se irá acercando el coseno anterior cuando la distancia que separa las dos apariciones del token en la frase vaya aumentando? Razona tu respuesta.

6. Una matriz de atención representa en cada fila el nivel de atención que se presta a los embeddings calculados para cada token de la frase en la capa anterior (representados en las columnas) cuando se va a calcular el embedding del token de dicha fila en un determinado cabezal (head) del codificador de un transformer. Un color más oscuro representa mayor atención. Considera ahora un cabezal (head) de un codificador de un transformer que presta una atención aproximadamente monótona sobre los embeddings de la capa anterior. En particular, para cada token se atiende en un elevado grado al embedding de ese mismo token en la capa anterior y con mucha menor intensidad al token inmediatamente a su derecha; el resto de embeddings de la capa anterior no reciben atención. Dibuja aproximadamente la matriz de atención resultante sobre la misma frase que la de la figura.
