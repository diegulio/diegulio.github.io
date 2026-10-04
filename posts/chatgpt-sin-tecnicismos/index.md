---
title: ChatGPT Sin Tecnicismos
subtitle: Explicamos como funciona ChatGPT de forma intuitiva
author: Diegulio
date: 2026-01-13
categories: [llm, explained]
image: posts/chatgpt-sin-tecnicismos/chatgpt.png
---

# 🤖 Introducción

En el año 2022 OpenAI lanzó ChatGPT con un simple tweet, un chatbot que buscaba ser un robot capaz de responder preguntas genéricas y mantener una conversación con los usuarios. Irónicamente, lanzaron esta herramienta como una versión preliminar, buscando que los usuarios la probaran y así pudiesen recibir retroalimentación de la comunidad. OpenAI nunca pensó que causaría tal revolución y que alcanzaría el millón de usuarios en 5 días.

Hoy en día, diversas empresas desplegaron su propio LLM. LLM (Large Language Model) es como se les llama a estos modelos, debido a que son **modelos** de **lenguaje** con una **gran** cantidad de parámetros. En otras palabras, son modelos que procesan texto y que tienen una gran cantidad de engranajes que ajustar para hacerlo.

> A partir de ahora nos referiremos a los modelos como ChatGPT con su nombre de pila: **LLM**

Básicamente, estas compañías se vieron forzadas a crear sus propios modelos y así comenzar a formar parte en la carrera por la búsqueda de la Inteligencia Artificial General. Hoy en día esta es una carrera constante, en donde cada ciertos meses una de estas empresas lanza una nueva versión de su LLM asegurando que es la mejor del mercado.

Actualmente, los LLMs son ampliamente utilizados en tópicos tan importantes como la salud o el trabajo; muchas veces sin tener conciencia de que hay detrás, y por consiguiente, que estos modelos suelen equivocarse o derechamente mentir (término conocido como **alucinación**).

Este post busca desmitificar lo que sucede dentro de ChatGPT, democratizando así el conocimiento y buscando que cualquier persona que esté leyendo esto entienda como estos modelos funcionan y responden a nuestras preguntas.

---

# 🪄 As simple as next token prediction

Comencemos con lo principal, la forma en la que interactuamos con ChatGPT desde nuestro punto de vista:

1. Hacemos una pregunta (a esto usualmente se le llama *prompt*)
2. ChatGPT nos responde.
3. Vuelve al paso 1.

¿Cómo este robot entiende lo que le estoy preguntando? ¿Cómo responde acorde a eso? ¿Está pensando? ¿Tiene un cerebro?

La respuesta es mucho más simple de lo que pensamos:

> ***ChatGPT tiene como único propósito predecir la siguiente palabra.***

Pero ¿cómo? si me responde párrafos enormes, ¿solo va a predecir la siguiente palabra?

En realidad es un proceso iterativo en donde va palabra por palabra, prediciendo una a la vez. No es coincidencia que cuando ChatGPT nos responde lo hace con una animación progresiva de escritura.

ChatGPT elige la siguiente palabra basado en el contexto. ChatGPT posee un conjunto finito de palabras, llamado **vocabulario**. A cada palabra del vocabulario se le asigna una probabilidad de ser la siguiente según las palabras que la preceden (contexto). La manera más fácil de escoger qué palabra es la siguiente, es tomando aquella con mayor probabilidad.

### Aclaración: Tokens

Por motivos de eficiencia y generalización, los LLMs en realidad no predicen la siguiente palabra, sino que predicen el siguiente *token*. Un token es una pieza de texto, puede ser una letra, sub-palabra, palabra u otro tamaño. Por simplicidad, en todo el post asumiremos que un token es una palabra.

---

# 🧠 A cerebro abierto: del prompt a la respuesta

El proceso es el siguiente:

El texto inicial *"Como me preparo para una triatlón?"* es dividido en palabras (tokens) por un proceso llamado *Tokenización*; cada palabra es transformada a números mediante el uso de *Embeddings*; las palabras, ahora de forma numérica entran al cerebro de ChatGPT, el cual utiliza una arquitectura llamada *Transformer*, que utiliza mecanismos de *atención* para así asignarle probabilidades a cada palabra de su vocabulario y elegir la siguiente.

## 1. Tokenización

La Tokenización es el proceso en donde se divide la entrada del modelo en tokens. Sea cual sea el nivel (letra, sub-palabra, palabra), a estos pedacitos se les llama *tokens*.

## 2. Embeddings

Los modelos de Inteligencia Artificial sólo aceptan números. Para transformar cada palabra (token) a un elemento numérico, se utilizan *Embeddings*. Los Embeddings son un conjunto de números que para un humano puede no significar nada, pero de donde los modelos pueden extraer patrones e información oculta.

Los LLM utilizan actualmente **miles** de dimensiones para describir una palabra.

## 3. Transformer

Esta arquitectura fue propuesta por Google en el año 2017 y fue un punto de inflexión en la Inteligencia artificial moderna. OpenAI utilizó esta arquitectura para crear el primer ChatGPT. De hecho, la **"T"** en GPT es de **Transformer**.

El título del paper que publicó Google cuando presentó esta arquitectura fue: *"Attention is All you need"* → Atención es todo lo que necesitas.

### 3.1 Mecanismo de Atención

El mecanismo de atención es bastante simple pero poderoso. Se trata de identificar a qué palabras del contexto actual debemos darle mayor importancia.

Imaginemos el contexto *"… Para la triatlón, debes saber"*, el mecanismo de atención toma cada palabra y le asigna un peso de importancia. Las palabras "triatlón" y "saber" parecen ser importantes para predecir la siguiente palabra, por lo que el modelo le da mayor atención.

Luego, se crea una especie de "contexto unificado", promediando los embeddings dándole mayor peso a aquellos con una mayor atención. Este contexto unificado finalmente se utiliza para asignarle probabilidades a cada palabra del vocabulario.

---

# 🏋🏽‍♂️ No Pain no gain: El entrenamiento de ChatGPT

ChatGPT no "nació" sabiendo predecir la siguiente palabra, fue entrenado para eso.

ChatGPT tiene dentro unas pequeñas tuercas llamadas **parámetros**, podemos pensar en estas como las neuronas del cerebro humano. En el proceso de entrenamiento, ChatGPT debe ajustar esas tuercas hasta encontrar la combinación que tenga mayor éxito en la tarea de predecir la siguiente palabra. Para esto, le damos una gran cantidad de ejemplos y lo penalizamos si responde mal, o recompensamos si lo hace bien.

Mientras más tuercas tengamos para ajustar, mayor capacidad tendrá nuestro modelo, pero a la vez más ejemplos necesitaremos. Los modelos actuales más grandes son entrenados con trillones de tuercas en trillones de datos, de acá el "Large" en **Large Language Models**.

Como dato, **Llama 3.1** de Meta:
- **405 billones** de parámetros
- **15.6 billones** de tokens de entrenamiento
- Utilizó **16,000 tarjetas gráficas**
- Se demoró **2 a 3 meses** en entrenarse
- Costo estimado: entre **93 y 123 millones de dólares**

---

# 👩🏽‍🏫 Apliquemos lo aprendido: Los Si/No de ChatGPT

## ¿Cuántas 'r' tiene la palabra Strawberry?

**Respuesta**: ❌ NO

**Explicación**: Como vimos, ChatGPT no lee a nivel de letras, sino a nivel de *tokens*. La palabra "Strawberry" se divide en tokens como ("Str" + "aw" + "berry"). Por lo tanto, al no ver a nivel de letra, ChatGPT inventa una respuesta.

## ¿Cuánto es 9.943 multiplicado por 583.254?

**Respuesta**: ❌ NO

**Explicación**: La misión de los LLMs es predecir la siguiente palabra, no son calculadoras. Lo que ChatGPT va a hacer es intentar predecir qué número suele ir después según los textos que ha leído. En resumen, no calcula, sino que responde algo que "parezca" una respuesta matemática.

## ¿Quiero que me resumas el final de Harry Potter?

**Respuesta**: ✅ SÍ

**Explicación**: Resumir es una de las tareas más adecuadas para ChatGPT. Esta tarea está totalmente alineada con su misión de predecir la siguiente palabra y además gracias al *mecanismo de atención*, puede capturar los puntos principales del contexto.

---

# 🎬 Conclusión y palabras finales

El objetivo de este post es capacitar al lector/a para entender el funcionamiento tras bambalinas de ChatGPT de forma intuitiva y motivante.

Vimos los pasos principales por los que nuestra pregunta pasa cuando dialogamos con ChatGPT: **Tokenización → Embeddings → Atención**, y cómo estos modelos son entrenados para tener un comportamiento humano y un espectacular conocimiento general.

Si bien, de cierta forma hicimos hincapié en el hecho de que los modelos **sólo** predicen la siguiente palabra, la evidencia sugiere que en realidad son capaces de planificar a futuro. Actualmente, los LLMs se utilizan para generación de imágenes, audio y video — y siguen mejorando a pasos agigantados.

Personalmente, creo que aún falta un largo camino para llegar a algo que realmente se sienta como Inteligencia Artificial General. Día a día se investigan nuevas arquitecturas, se mejoran los datos y los niveles de procesamiento. Mientras tanto, nos queda aprovechar las nuevas oportunidades que se están dando con estas herramientas. 🚀
