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
- Por último, no es necesario que comprendas la sección 5.10 ("Advanced: Deriving the Gradient Equation") antes de pasar a capítulos posteriores, pero es muy recomendable que seas capaz de derivar por ti mismo la función de pérdida más pronto que tarde. Cuando los modelos se vayan haciendo más complejos obtener la derivada manualmente será una tarea ardua e innecesaria (porque librerías como PyTorch se encargarán de calcularla por nosotros), pero hacerlo ahora mejorará tu perspectiva del entrenamiento de redes neuronales.

Este capítulo es probablemente el más complejo de todos y el que ofrece una mayor curva de aprendizaje al aparecer en él un montón de elementos que quizás son nuevos para ti. A continuación se enfatizan algunos de los conceptos más importantes de cada apartado.

## Anotaciones al libro

#### Apartado 5.1

Se introduce el concepto de producto escalar que será una piedra angular de todo lo que está por venir. Si recuerdas cómo se hacía el producto de matrices (que aparecerá numerosas veces más adelante), observarás que este consiste en una serie de cálculos de productos escalares. El sesgo (*bias*) es importante en algunos problemas porque permite desplazar las fronteras de decisión como demostraremos más adelante. Observa que no linealidad de la exponenciación de la función sigmoide *encoge* las diferencias entre los valores de salida de la función según nos alejamos del origen, es decir, aunque 2-0 = 4 -2, se tiene que $$\sigma(2)-\sigma(0) >>> \sigma(4)-\sigma(2)$$. Por otro lado, no es necesario que hagas la demostración analítica, pero sí que observes gráficamente en $$1 - \sigma(x) = \sigma(-x)$$; esta propiedad nos permitirá simplificar algunas ecuaciones. Finalmente, observa que por ahora la función $$\sigma$$ se está aplicando a un escalar, pero más adelante se aplicará a un vector o incluso a un tensor de cualquier número de dimensiones. En este caso, la función se aplica elemento a elemento, es decir, si $$\mathbf{x}$$ es un vector, $$\sigma(\mathbf{x}) = [\sigma(x_1), \sigma(x_2), \ldots, \sigma(x_n)]$$.

#### Apartado 5.2

El ejemplo de clasificación de sentimiento de esta sección es interesante porque muestra la técnica usada hasta hace unos años para esta tarea. Se supone aquí que una persona experta en el dominio ha definido las características (*features*) que ella considera que pueden ser importantes para decidir si una frase tiene connotaciones positivas o negativas. Estas características se computarán para cada frase mediante un programa antes de poder pasarla por el regresor. Este es un proceso costoso porque requiere expertos de cada dominio y porque el criterio de lo que es relevante o no puede ser subjetivo; el número de características en este caso solía estar alrededor de unas pocas decenas. En la actualidad, los modelos neuronales, como veremos, procesan los datos *en bruto* y aprenden las características más relevantes (en cantidades de cientos o miles de ellas), aunque en la mayoría de las ocasiones estas no tienen una interpretación lógica para los expertos. 

Por otra parte, la idea de normalización puede parecer ahora poco relevante, pero jugará un papel importante en el modelo del transformer para evitar que ciertos valores intermedios se hagan demasiado grandes o pequeños. Si miras la gráfica de la función sigmoide en el apartado anterior, verás que para valores de $$x$$ muy grandes o muy pequeños no hay apenas diferencias en el valor de $$\sigma(x)$$ por lo que la función no será sensible a pequeños cambios en el valor de $$x$$. Además, en estas zonas la función es prácticamente plana, por lo que su derivada es muy pequeña lo que, como veremos más adelante, dificulta el entrenamiento. Por último, la idea de procesar varios datos de entrada a la vez es también muy importante, ya que permite reducir el tiempo de procesamiento. Puedes ver cómo empaquetando por filas una serie de vectores de entrada y con una simple multiplicación matricial seguida de la aplicación de la función sigmoide se obtiene el resultado de la clasificación de todos los vectores de entrada a la vez. Las GPUs están especializadas en poder realizar estas operaciones matriciales de forma muy eficiente, por lo que siempre intentaremos empaquetar los datos en los denominados *mini-batches* (mini-lotes, en español) para llenar la memoria de la GPU con la mayor cantidad de ellos y poder procesarlos en paralelo. Para que la operación de suma del sesgo sea consistente en tamaños, es necesario *estirar* el sesgo para obtener un vector $$b$$ con el mismo tamaño que el número de muestras procesadas a la vez. Cuando trabajemos con PyTorch veremos que esta operación se realiza automáticamente y que, gracias al mecanismo de *broadcasting*, no es necesario obtener explícitamente un vector con el valor del sesgo copiado y podremos sumar directamente el escalar o un tensor unidimensional de tamaño 1.

#### Apartado 5.3 

La función softmax es el equivalente de la función sigmoide cuando se tiene que clasificar una muestra en más de dos clases. En este caso, la función recibe un vector de valores no normalizados (es decir, sin un rango concreto) y lo transforma en un vector de probabilidades de pertenencia a cada una de las clases. Al vector no normalizado se le denomina *logits* (logit es la función inversa de la función sigmoide). Observa que no podríamos haber normalizado los valores entre 0 y 1 dividiendo cada uno por la suma de todos ellos porque hay valores negativos que anularían otros positivos. Podríamos haber considera elevar cada valor del vector de entrada al cuadrado y dividirlo por la suma de todos los cuadrados, pero la función softmax las diferencias, como hemos comentado, y penaliza más los valores más alejados del máximo:

```python
z = torch.tensor([0.6,1.1,-1.5,1.2,3.2,-1.1])
squared = z*z / sum(z*z)
softmax= torch.nn.functional.softmax(z, dim=-1)
print(z, squared, softmax)  
# z =       [0.6000, 1.1000, -1.5000, 1.2000, 3.2000, -1.1000]
# squared = [0.0215, 0.0724,  0.1346, 0.0862, 0.6128,  0.0724]
# softmax = [0.0548, 0.0904,  0.0067, 0.0999, 0.7382,  0.0100]
``` 

Observa que cuando aquí hacemos $$\hat{\mathbf{y}} = \mathrm{softmax} (\mathbf{W} \mathbf{x} + \mathbf{b})$$, el vector $$\mathbf{x}$$ corresponde a una única muestra, pero, a diferencia de apartados anteriores, $$\mathbf{W}$$ es una matriz y $$\mathbf{b}$$ es un vector con valores no necesariamente repetidos. En este caso, la matriz $$\mathbf{W}$$ de forma $$K \times f$$ transforma un vector de características de tamaño $$f$$ en un vector de logits de tamaño $$K$$, donde $$K$$ es el número de clases. Si cambiamos la forma de la matriz a $$f \times K$$, entonces una operación equivalente se realizaría con $$\hat{\mathbf{y}} = \mathrm{softmax} (\mathbf{x} \mathbf{W}  + \mathbf{b})$$, donde ahora $$\mathbf{x}$$ y $$\mathbf{b}$$ son vectores fila y no columna. Observa asimismo que en lugar de aplicar la operación a una única muestra (por ejemplo, las características de una sola frase), podemos hacerlo a un lote de estas, *apilándolas* por filas en una matriz $$\mathbf{X}$$ y haciendo $$\hat{\mathbf{y}} = \mathrm{softmax} (\mathbf{X} \mathbf{W}  + \mathbf{B})$$ o apilándolas por columnas y haciendo $$\hat{\mathbf{y}} = \mathrm{softmax} (\mathbf{W} \mathbf{X}  + \mathbf{B})$$. En ambos casos, la matriz $$\mathbf{B}$$ contendrá *repetido* $$m$$ veces el vector de sesgos $$\mathbf{b}$$. El resultado será un lote de $$m$$ vectores de logits de tamaño $$K$$, uno por cada muestra del lote.

Cuando de ahora en adelante veas una ecuación de una parte de un modelo neuronal en la que se multiplica un lote de vectores por una matriz, puedes identificar que se trata de una transformación lineal que transforma cada uno de los vectores de entrada en otro vector normalmente de tamaño diferente. Recuerda bien esto cuando estudiemos el tema de las redes neuronales hacia adelante.

En este apartado se introduce también el concepto de vector *one hot* (un vector donde todos los elementos son cero, excepto uno de ellos que vale uno) que usaremos con frecuencia para referirnos al vector con el que compararemos la salida de la red neuronal. Por ejemplo, si tenemos un problema de clasificación de imágenes de dígitos, el vector *one hot* que correspondería a la etiqueta del 3 sería $$\mathbf{y} = [0,0,0,1,0,0,0,0,0,0]$$.

#### Apartado 5.4

Se avisa de que los dos próximos apartados se centran en la entropía cruzada y el descenso por gradiente para el caso de la regresión logística binaria y que después se retomará la regresión softmax.

#### Apartado 5.5

La ecuación $$p(y \vert x) = \hat{y}^y (1−\hat{y})^{1-y}$$ es solo una forma compacta de escribir matemáticamente la idea de que, si tenemos un dato correctamente etiquetado como $$y$$ (donde $$y$$ es cero o uno), la verosimilitud que el modelo da a este dato es $$\hat{y}$$, si el dato está etiquetado como 1 y $$1−\hat{y}$$ si está etiquetado como 0. Verosimilitud y probabilidad denotan algo muy similar a efectos prácticos, pero usaremos el término *verosimilitud* para referirnos a la probabilidad de una serie de datos cuando vamos asignando distintos valores a los parámetros (o pesos) del modelo; por otro lado, si los parámetros no son una variable aleatoria, sino que tienen un valor concreto, hablaremos de la *probabilidad* que tienen los datos dados los parámetros del modelo. 

Un aspecto básico del entrenamiento de redes neuronales es el principio de estimación por máxima verosimilitud. La explicación del capítulo sobre este método puede complementarse con este [breve tutorial][tutorialmle] [<i class="fas fa-file"></i>][tutorialmle]. La idea básica es ir *probando* con diferentes valores de los parámetros intentando encontrar los que maximizan la verosimilitud de los datos. En el caso de la regresión logística, esto se traduce en encontrar los valores de los pesos $$\mathbf{W}$$ y de los sesgos $$\mathbf{b}$$ que maximizan la probabilidad de que los datos etiquetados como 1 tengan una probabilidad alta de ser 1 y los datos etiquetados como 0 tengan una probabilidad baja de ser 1.

[tutorialmle]: https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/08/19/01-Maximum-likelihood-estimation.html

Aunque en algún momento del capítulo se calcula el valor concreto de la función de pérdida $$L_{CE}(\hat{y},y)$$ para un par de datos concretos, nuestro principal interés estará en la forma analítica de la ecuación (5.23), ya que, como se ve en el siguiente apartado, es la que usaremos para calcular el gradiente de la función de pérdida con respecto a los parámetros del modelo y, por tanto, para actualizarlos en cada paso de entrenamiento. No obstante, la media de la función de error sobre un conjunto de datos (bien los datos de entrenamiento, bien los de validación) en forma de un número concreto (0.564, por ejemplo) nos será útil durante el entrenamiento para comprobar si el modelo está mejorando o no.

La discusión sobre la entropía cruzada (*cross-entropy*) se puede extender un poco para ver de dónde viene esta función y por qué minimizarla es equivalente a maximizar la verosimilitud. Más [abajo][cross] tienes una pequeña explicación de este tema.

[cross]: #entropía

Los logaritmos van a aparecer con cierta frecuencia al aprender sobre redes neuronales, por lo que es conveniente recordar algunas de sus propiedades:

- Logaritmo del producto: $$\log(xy) = \log(x) + \log(y)$$
- Logaritmo de la división: $$\log(x/y) = \log(x) - \log(y)$$
- Logaritmo de la exponenciación: $$\log(x^a) = a\log(x)$$, donde $$a$$ es una constante
- Logaritmo de uno: $$\log(1) = 0$$

#### Apartado 5.6

Para recordar fácilmente cómo afecta la derivada a la actualización de los pesos considera, para simplificar, que la función de error adopta la forma de $$x^2$$ y observa en la siguiente gráfica cómo cuando el gradiente es negativo (en el punto con la marca de la estrella) es necesario incrementar el peso para reducir el error, mientras que cuando el gradiente es positivo (en el punto con la marca del círculo) es necesario reducir el peso para reducir el error. En la gráfica se muestra también la derivada de la función de error, que es $$2x$$.

{% include figure.html path="assets/img/transformers/derivativex2.png" title="derivative of x^2" class="img-fluid rounded z-depth-1" width="512px" %}

La gráfica anterior ha sido generada con el siguiente código:

```python
import matplotlib.pyplot as plt
import numpy as np
squares = lambda x: x*x
derivative = lambda x: 2*x
x = np.linspace(-25, 25, 500)
y = np.array([squares(xi) for xi in x])
plt.plot(x, y)
plt.text(-17, 300, r'$x^2$', fontsize=12)
plt.scatter(-15, squares(-15), marker="*", color="blue")
plt.arrow(-14, squares(15), 10, 0, head_width=20, head_length=2, width=1, color="green")
plt.scatter(18, squares(18), marker="o", color="blue")
plt.arrow(17, squares(18), -10, 0, head_width=20, head_length=2, width=1, color="green")
y = np.array([derivative(xi) for xi in x])
plt.plot(x, y)
plt.text(20, 72, r'$2x$', fontsize=12)
plt.xlabel(r'$w$')
plt.ylabel('loss')
plt.grid()
plt.savefig("derivativex2.png")
```

#### Apartado 5.7

Puedes saltar este apartado sobre la regularización, ya que no es fundamental en estos momentos para entender el funcionamiento de las redes neuronales.

#### Apartado 5.8

Cuando se usa la entropía cruzada como función de error en la regresión multinomial, su forma es muy sencilla ya que es igual al logaritmo de la probabilidad de la clase correcta. No obstante, la complejidad de la derivada dependerá de la complejidad de todo el modelo subyacente. En el caso de la regresión logística binaria, la función de error es la entropía cruzada binaria, que ya vimos que toma la siguiente forma:

$$L_{CE}(\hat{y},y) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

#### Apartado 5.9

Puedes saltar este apartado.

#### Apartado 5.10

Se calcula aquí paso a paso el gradiente de la entropía cruzada binaria respecto a cada uno de los parámetros del regresor logístico binario. Estas son las reglas de derivación que se necesitan para derivar la mayor parte de las funciones de error que se usan en redes neuronales:

- Derivada con un exponente: $$\frac{d}{dx}(x^a) = a x^{a-1}$$
- Derivada con el producto de una constante: $$\frac{d}{dx}(cx) = c$$, donde $$c$$ es una constante
- Derivada de una constante: $$\frac{d}{dx}(c) = 0$$
- Derivada de la suma: $$\frac{d}{dx}(x+y) = \frac{d}{dx}(x) + \frac{d}{dx}(y)$$
- Derivada del logaritmo: $$\frac{d}{dx}(\log(x)) = \frac{1}{x}$$
- Derivada del producto: $$\frac{d}{dx}(xy) = y\frac{d}{dx}(x) + x\frac{d}{dx}(y)$$

Dado que la función de error será una función compuesta de múltiples funciones, la regla de la cadena nos será de suma utilidad:

$$\displaystyle \frac{\displaystyle d f(g(x))}{\displaystyle dx} = f'(g(x))\cdot g'(x)$$, 

donde $$f'$$ y $$g'$$ representan las derivadas de $$f$$ y $$g$$ respectivamente.

Sería interesante que calcularas la derivada de la entropía cruzada binaria respecto al umbral y que te animaras a calcular también el gradiente para el caso de la regresión logística multinomial.

## Regresores implementados en PyTorch

Estas son algunos implementaciones en PyTorch de los regresores que hemos estudiado en unas pocas decenas de líneas de código (normalmente menos de 100). Asegúrate de que terminas entendiendo el código lo suficiente como para sentirte con ánimo de poder modificarlo para adaptarlo a otras necesidades.

- Un [regresor logístico][pylog] [<i class="fas fa-file"></i>][pylog] que clasifica muestras bidimensionales sintéticas en dos clases. Se usan solo los elementos más básicos de PyTorch para poder tener una implementación lo más detallada posible. Como ejercicio, puedes hacer una traza y analizar qué tamaños tienen los tensores. Puedes jugar también con el número de pasos de entrenamiento  y la tasa de aprendizaje para ver cómo evoluciona el entrenamiento. Explora diversas posiciones de los centros de las clases y de la dispersión de los datos alrededor de estos y observa cómo cambia la frontera de decisión. Elimina el sesgo (*bias*) de las ecuaciones y observa cómo se restringe la forma de la frontera de decisión al obligar a esta a pasar por el origen de coordenadas.
- Un [regresor softmax para clasificar imágenes de dígitos][pysoft] [<i class="fas fa-file"></i>][pysoft]. Las imágenes y etiquetas de los dígitos se toman de un conjunto de datos muy conocido llamado MNIST. Como ejercicio, puedes simplificar este código para que realice una tarea de clasificación de sentimiento sobre un conjunto de datos sintéticos muy pequeño que se defina explícitamente en el propio programa; puedes inspirarte en el código de los siguientes ejemplos para ello.

Si no lo has hecho ya, puedes empezar a aprender Python y PyTorch siguiendo el [capítulo][cappy] correspondiente de esta serie.

[cappy]: ../python
[pylog]: https://github.com/jaspock/me/blob/master/assets/code/transformers/logistic-regressor.py
[pysoft]: https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/softmax-regressor.py

**Actividad práctica.** Utiliza las herramientas de depuración como se explicaron [aquí][debug] para explorar paso a paso el código de los regresores. Analiza el valor y tipo de las variables, así como la forma de cada tensor y asegúrate de que entiendes qué representa cada dimensión.

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

Puedes ver, por lo tanto, que la fórmula (5.21) del libro coincide con la ecuación anterior: maximizar la verosimilitud respecto a los parámetros del modelo es equivalente a minimizar la entropía cruzada $$H(y,\hat{y})$$.


## Ejercicios de repaso

Estos ejercicios te permitirán repasar los conceptos más importantes de este capítulo.

1. La disposición de los elementos en matrices y vectores puede ser diferente a la utilizada en la sección 5.2.3. Lo realmente importante es que se realice el productor escalar de cada una de las $$m$$ muestras con el vector de pesos $$\mathbf{w}$$. Indica qué tamaños deberían tener las matrices y vectores si en lugar de una ecuación como la 5.14, usamos una de la forma $$\mathbf{y} = \mathbf{w} \mathbf{X} + \mathbf{b}$$.
2. Calcula la derivada de la función de coste respecto al umbral $$b$$. Si te basas en la derivada de la función de coste respecto a los pesos $$w$$, que está calculada en el libro, llegarás rápido a la solución.
3. Tras entrenar un regresor logístico, le aplicamos una entrada $$\mathbf{x}$$ y calculamos la derivada $$\partial \hat{y} / \partial \mathbf{x}_i$$ para un cierto $$i$$. ¿Qué mide esta derivada? Piensa en el concepto básico de la derivada y en cómo mide la *sensibilidad* del valor de un función respecto a un cambio en una de sus variables.
