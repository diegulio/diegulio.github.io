---
title: "Iniciando en LLM: Crea tu primera aplicación con LangChain y ChatGPT"
subtitle: "Cocina tu comida favorita con la ayuda de LLM"
author: "Diegulio"
date: "2023-07-05"
categories: [python, llm, langchain]
image: "posts/llm-recipe/portada.jpeg"
---
# LLM Recipe

[![](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/diegulio/llm-recipe)

> [!NOTE]
>
> Antes de comenzar a leer esto, recuerda que yo estoy aprendiendo junto contigo. Si tienes alguna duda, sugerencia, correción o comentario, no dudes en escribirme a mi [LinkedIn](https://www.linkedin.com/in/dieguliomachado/).

# 📚 Tópico: Large Language Models

Estoy seguro que alguna vez haz escuchado el término ChatGPT. Yep! ese robot 🤖 que te hace hasta la tesis!  ChatGPT es un LLM (Large Language Model), lo que en otras palabras significa que es un modelo matemático que se alimenta de grandes volumenes de texto y es capaz de generar lenguaje humano de manera muuy avanzada. Y si que lo hemos visto en acción, yo lo ocupo en mi dia a dia y me facilita un montón mi trabajo.

Ahora ¿ Que tal si dejamos que un LLM nos describa que es un LLM? 

> [!NOTE] 🤖
>
> Un large language model (modelo de lenguaje amplio) es un tipo de sistema de inteligencia artificial diseñado para comprender y generar lenguaje humano de manera avanzada. Estos modelos están entrenados en grandes cantidades de texto y utilizan técnicas de aprendizaje automático para aprender patrones y estructuras lingüísticas. Un large language model, como GPT-3.5, puede responder preguntas, redactar textos, generar código, traducir idiomas y realizar una variedad de tareas relacionadas con el lenguaje natural. Estos modelos son capaces de capturar la complejidad y sutileza del lenguaje humano, y pueden adaptarse a diferentes contextos y estilos de escritura.

En este post **no** hablaremos de las técnicas utilizadas para construir un LLM, ya que en internet hay muy buenas fuentes para aprender sobre esto, acá te dejo una: 

[LLM University (LLMU) | Cohere](https://docs.cohere.com/docs/llmu)

Lo que haremos en este post, será utilizar los LLM en el bien de la humanidad ! 🦸🏽‍♀️ o quizá sólo en nuestro bien. 

# 🛒 Motivación: De compras con LLM

El otro dia queria cocinar lasagna, pero no sabia si tenia los ingredientes en casa. En realidad, no queria gastar ningún ingrediente en casa, por lo que me metí a la website de mi supermercado favorito para comprar los ingredientes. El problema es que nunca he cocinado lasagna por lo que no sé que ingredientes lleva! me dió tanta flojera googlear una receta, extraer los ingredientes y ponerlos uno por uno en el carrito de compras que terminé comiendo cereales con leche 🥣. 

Ahí fue cuando pensé que sería bueno que el supermercado tuviese integrado un algoritmo que agregue automáticamente los ingredientes de una comida específica en el carrito de compras! 🛍️

🧠 **Solución: Utilizar LLM para obtener recetas y extraer los ingredientes de forma estructurada.** 

La verdad esto fue una excusa para comenzar a aprender a utilizar los LLM, el mundo se está moviendo en torno a esto y no pienso quedarme atrás.

# 🔨 Tool Path: Que utilizaremos

A continuación les dejo las herramientas que utilizaremos en este post:

1. **ChatGPT API**: Para poder acceder a los poderes de ChatGPT
2. **LangChain**: Para poder comunicarme de manera fácil con la API de ChatGPT, además de aprovechar un montón de los facilitadores que tiene para construir herramientas basadas en LLM
3. **Python**: Lenguaje de programación
4. **Gradio**: Para crear una interfaz simple de uso 

# 💭 Concept Path: Que aprenderemos

A continuación algunos de los conceptos que aprenderemos: 

1. LLM: Ya hablamos un poco sobre esto
2. Prompt Engineering
3. Prompt Templates
4. Chains
5. Environments
6. Framework

La verdad esto es un primer paso para aprender LLM, hay muchos conceptos más allá que aún queda por explorar, y lo peor es que esto sigue avanzando a pasos agigantados. 🦶🏽

# ♟️ Estrategia: Como abordamos

La solución es bastante directa, pedirle a algún LLM que nos entregue los ingredientes de una comida especifica

## Prompt:

Imaginemos queremos saber los ingredientes para cocinar una lasagna, entonces escribiremos algo del estilo:

> [!NOTE] 👩🏼‍🔬
>
> ¿Que necesito para cocinar una Lasagna?

Esta pregunta, que es la entrada de un LLM se le denomina `prompt.` Un término bastante conocido hasta ahora, que de hecho se asocia a un rol, es `Prompt Engineering`. Podemos entender este término como el “arte” de escribir el mejor prompt para obtener la respuesta deseada. 

“Cuida la forma en la que pides las cosas” me decía mi mamá cuando niño al pedirle un favor a alguien. Las mamás siempre tienen la razón, y esta no es la excepción.  

Imaginemos que la respuesta de la LLM es algo como:

> [!NOTE] 🤖
>
> Para cocinar una lasagna necesitas un horno, un cuchillo, una cocina y los ingredientes.

No es la respuesta que esperabamos! nosotros en realidad queríamos saber los ingredientes, pero nos expresamos mal. Es por esto que comúnmente se suele iterar el prompt hasta conseguir la respuesta deseada, imaginemos que luego de iterar un poco llegamos al prompt final:

> [!NOTE] 👩🏼‍🔬
>
> ¿Cuales son los ingredientes que necesito para cocinar una lasagna?

Probablemente con un prompt así obtengamos lo que buscamos. No hay un prompt óptimo, pero si existen muchos prompt que nos conseguirán la respuesta que buscamos.

Más adelante veremos algunas técnicas de prompt engineering. Por ahora nos basta con saber que el prompt será elemento importante de nuestra solución.

## Output:

Algo que nos debemos cuestionar es: ¿Como necesitamos el resultado? algunas de las opciones son: Una lista de ingredientes, un json, un DataFrame, etc. 

En este caso lo que decidí fue obtener un json, el cual luego convertiría a un DataFrame. Los elementos que tendrá la respuesta son:

- **Ingredient**: Nombre del ingrediente
- **Quantity**: Cantidad necesaria
- **Optional**: “Yes” si el ingrediente es opcional, “No” si es obligatorio
- **Estimated Price**: Un precio estimado en dólares del ingrediente (así podremos calcular algún valor aproximado de la receta)
- **Available**: Una simulación de disponibilidad del ingrediente en el supermercado. “Yes” si está disponible, “No” si no lo está.

Acá tenemos un ejemplo:

```json
{
  "Spaguetthi With Meat":
  [
    {
      "ingredient": "Spaguetti",
      "optional": "No",
      "quantity": "200g",
      "estimated_price": "5.00",
      "available": "No"
    },
    {
      "ingredient": "Meat",
      "optional": "No",
      "quantity": "1kg",
      "estimated_price": "10.00",
      "available": "Yes"
    },
    {
      "ingredient": "Pepper",
      "optional": "Yes",
      "quantity": "at ease",
      "estimated_price": "1.00",
      "available": "No"
    }

  ]
}
```

Una pregunta importante es, ¿Cómo lograremos que el LLM nos estructure el output como lo requerimos?

**SPOILER: Prompt Engineering**

# 🔨 Tools

En este post no busco explicar a fondo las herramientas que utilizaré, si no que mostrar su uso. Para entender más sobre ellas te dejaré enlaces a sus propias documentaciones (probablemente mucho mejor explicado como lo haría yo).

A continuación mencionaré un poco sobre las herramientas que utilizaremos en las siguientes secciones:

- **Langchain**: LangChain es un framework diseñado para simplificar la creación de aplicaciones utilizando modelos de lenguaje grandes (LLM). Como framework de integración del modelo de lenguaje, los casos de uso de LangChain se superponen en gran medida con los de los modelos de lenguaje en general, incluidos el análisis y resumen de documentos, los chatbots y el análisis de código.
- **ChatGPT API**: ChatGPT API es una interfaz de programación de aplicaciones (API) que permite a los desarrolladores interactuar con el modelo de lenguaje ChatGPT de OpenAI. Esta API permite enviar solicitudes a través de una conexión de red para obtener respuestas generadas por el modelo en tiempo real. Al utilizar la API, los desarrolladores pueden integrar fácilmente la funcionalidad de ChatGPT en sus propias aplicaciones, productos o servicios. Proporciona una manera conveniente de aprovechar la potencia de ChatGPT para tareas como conversación, generación de texto y respuestas a preguntas en una amplia gama de aplicaciones y escenarios.

# 🧠 Prototyping

La forma final en la que plantearemos la solución será:

1. Usar un LLM que obtenga la receta según la comida
2. Usar un LLM que obtenga los ingredientes de la receta obtenida del paso 1
3. Crear la cadena final de LLM. Esto es, unir los resultados de los pasos 1 y 2.
4. Parsear los resultados. Esto es, estructurarlos.

> [!NOTE]
>
> 👀 Inicialmente habia pensado en que un LLM directamente me entregue los ingredientes desde una comida especificada, pero luego de unas cuantas iteraciones llegué a que de la forma en base a chains conseguía mejores resultados

## 1. Obtener receta mediante LLM

### Prompt Templates

En este punto vale la pena preguntarnos, cómo esperamos que el usuario interactúe con nuestra aplicación? Queremos que el usuario haga la pregunta completa? Ahora sabemos que esto no es una buena idea por varias razones:

1. El usuario podria ingresar incluso preguntas que no estén relacionadas con la aplicación (comida)
2. El usuario puede preguntar de forma ineficiente
3. Obtener la estructura json que deseamos sería imposible
4. El usuario no sabe de `Prompt Engineering`

La idea es que el usuario sólo ingrese el nombre de la comida, y por detrás nuestra aplicación haga el resto. Para esto, Langchain cuenta con una herramienta llamada Prompt Templates, que como su nombre lo indica es una plantilla del prompt. 

Esto es, imaginemos nuestra plantilla es: “¿Cuales son los ingredientes que necesito para cocinar {COMIDA}?” Entonces si la entrada del usuario es “Lasagna”, el prompt quedará  “¿Cuales son los ingredientes que necesito para cocinar *Lasagna*?”. 

### 👨🏾‍💻 Code time!

Para crear un template, primero debemos definir la estructura:

```python
template_string = """
Give me a list of ingredients to cook {food}.
"""
```

Notemos que el input en este caso es “food”. Luego usamos Langchain

```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(template_string)
```

Luego simplemente obtenemos el prompt final para entregarle al modelo de la siguiente forma:

```python
user_input = 'Lasagna'
final_prompt = prompt.format(food=user_input)
```

Así de simple! Ahora quiero que nos compliquemos un poco más la vida. ChatGPT hizo unos cambios en su API, por lo que en Langchain ahora aparece un nuevo elemento llamado `ChatPromptTemplate`

La idea de este template, que si bien también acepta entradas como las vistas anteriormente, ahora permite ingresar mensajes con `roles`. Existen tres tipos de roles: System, Human, AI. Bien brevemente te explico que deberian ser:

- System Message: Las instrucciones que se le quiere entregar al modelo
- Human Message: Las entradas del usuario
- AI Message: Alguna respuesta por parte de el modelo

Acá te dejo un post del por qué de esta implementación, que viene de la mano con `ChatModels`: [https://blog.langchain.dev/chat-models/](https://blog.langchain.dev/chat-models/)

Debido a que sólo el paso 1 utiliza input de usuario, sólo lo haremos así en este paso:

```python
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# System template
first_system_template_str = """
    You are a good chef, users need you to bring them recipes from given food.
"""
first_system_template = SystemMessagePromptTemplate.from_template(first_system_template_str)

# Human Template
first_human_template_str = "{food}"
first_human_template = HumanMessagePromptTemplate.from_template(first_human_template_str)

# First Prompt Template
first_prompt = ChatPromptTemplate.from_messages([first_system_template, first_human_template])
```

### LLM Model

Ahora que ya tenemos una entrada para nuestro modelo, necesitamos llamarlo!

Para poder hacer uso de un modelo de LLM (En este caso ChatOpenAI) primero debemos crearnos una cuenta en [openai](https://openai.com/blog/openai-api) y crear una nueva API Key. Luego, debemos crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```bash
OPENAI_API_KEY = "<tu api key>"
```

> [!WARNING]
>
> No debes dejar que nadie vea tu API KEY, asi que no subas tu .env a ningún lugar público!

Si no, puedes agregarlo directamente utilizando la libreria openai o Langchain.Luego de obtener y entregar tu API KEY, en Langchain basta con hacer algo como:

```python
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.0)
prompt = "Porfavor hazme la tesis!"

result = chat(prompt)
```

En este caso, estaremos utilizando dos LLM, en donde la salida de una es la entrada de otra. Langchain ya tiene algo preparado para esto! se les llama Chains. En este caso, tenemos una cadena simple (Dos LLM secuenciales), pero existen otros tipos de arquitecturas mucho más complejas ☠️

Para poder crear nuestra primera Chain no es tanto más complejo que el ejemplo anterior, sólo debemos agregar unos pasos extras:

```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# LLM Definition
llm = ChatOpenAI(temperature=0)

# Chain Step 1
recipe_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="recipe")

```

En el código anterior estamos utilizando la LLM ChatOpenAI, y creando la primera parte de la cadena. El parámetro `output_key` nos servirá más adelante para comunicarle a la segunda parte cual es el nombre del primer output. 

## 2. Obtener ingredientes de una receta mediante LLM

Ya sabemos que la salida del paso anterior será una receta, a la cual deberemos extraer los ingredientes. Además, éste es el último paso, por lo que deberemos preocuparnos de que el output que salga del LLM sea adecuado y simple de  “parsear”.

Para el output, lo que haremos será aplicar un poco del ya conocido `prompt engineering`. 

Además, en este caso, ya que no existen tantos riesgos de `prompt injection` , es que usaremos templates sencillos (no haremos uso de los roles en los mensajes)

 

```python
second_step_template = """I need you to bring me the ingredients contained in the following recipe: \
recipe: {recipe}
{format_instructions}
{example_instructions}"""
```

Vemos que tenemos tres entradas:

- recipe: Este input será el resultado del paso anterior.
- format_instructions: Acá le comunicaremos al LLM como queremos el formato del output.
- example_instructions: Acá aplicamos un poco de lo que se llama `few-shot` , que es básicamente darle un par de ejemplos al LLM para que entienda como esperamos el resultado.

```python
# Format Instructions
custom_format_instructions = """
The output should be in a json format, formatted in the following schema:
{
  "food": List // List of ingredients
  [
    {
      "ingredient": string // Name of one ingredient
      "quantity": string  // Quantity of the ingredient 
      "optional": string  // Whether or not that ingredient is optional to cook the food. "Yes" if the ingredient is not indispensable to cook, "No" if is the ingredient is indispensable.
      "estimated_price": string  // The ingredient's estimated price in dolars
      "available": string // Random "Yes" or "No"
    }
  ]
}
"""
```

```python
	example_instructions = """
Follow the schema of this example:
{
  "food":
  [
    {
      "ingredient": "Spaguetti",
      "optional": "No",
      "quantity": "200g",
      "estimated_price": "5.00",
      "available": "No"
    },
    {
      "ingredient": "Meat",
      "optional": "No",
      "quantity": "1kg",
      "estimated_price": "10.00",
      "available": "Yes"
    },
    {
      "ingredient": "Pepper",
      "optional": "Yes",
      "quantity": "at ease",
      "estimated_price": "1.00",
      "available": "No"
    }

  ]
}
"""
```

> [!WARNING]
>
> ✋🏽 Es importante que sepas que langchain cuenta con un módulo de Output parser que crea por detrás el format_instructions e incluso cuenta con funciones que transforman la salida del LLM(string) en el formato que deseabamos (json, lista, Pydantic, etc).
> La razón de porqué yo no ocupé esto fue que no funciona para json nesteados. Pero me basé en sus instrucciones para crear `custom_format_instructions`

Luego seguimos como lo vimos anteriormente:

```python
second_prompt = ChatPromptTemplate.from_template(second_step_template)
ingredient_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="ingredients")
```

## 3. Cadena Final

Luego necesitamos orquestar las cadenas que creamos anteriormente para obtener la cadena final. En Langchain se hace de la siguiente forma:

```python
overall_simple_chain = SequentialChain(chains=[recipe_chain, ingredient_chain], verbose=True,
                                       input_variables=["food", "format_instructions", "example_instructions"],
                                       output_variables=["recipe", "ingredients"])

```

## 4. Output Parser

Finalmente, necesitamos parsear el resultado obtenido desde la LLM. Esto es ***string → json.***

Debido a la forma en que le pedimos el resultado al LLM, es que se nos hace muy sencillo:

```python
import json
import pandas as pd

result = overall_simple_chain(
        {
            "food": food, # Esto lo entrega el usuario
            "format_instructions": custom_format_instructions, # Esto lo entregamos nosotros
            "example_instructions": example_instructions, # Esto lo entregamos nosotros
        }
    )

# String to json(dict en python)
dict_response = json.loads(result['ingredients']) 

# Dict to df
output_df = pd.DataFrame(data = dict_response['food'])

```

Finalmente, para la entrada  “Lasagna”, podemos obtener algo asi:

![Output 1 as DataFrame](assets/Untitled.png)

# 🧐 Front-End

Nuestra aplicación no puede quedarse sólo en código, es por esto que creamos un pequeño front-end para los usuarios. A continuación te dejo una imagen, pero lo mejor es que corras el código por ti mismo y los pruebes! Te invito a mejorar el algoritmo! 

![Input and Output 0](assets/Untitled%201.png)

![Output 1](assets/Untitled%202.png)

Notar que en el front-end se le entrega al usuario tanto la receta como el resumen de los ingredientes. Algo interesante de gradio es el botón `Avisar`. Este botón sirve para recibir feedback de los usuarios en caso de que el resultado no haya sido satisfactorio, y así puedes mejorar tu producto.

# 🚀 Próximos Pasos

Esto es sólo una solución inicial, podemos mejorarla de muchas formas, algunas que se me ocurren por ahora son:

- Prompt Optimization: Mejorar más los prompt. Quizá agregarle el hecho de que una persona puede ingresar algo que no es una comida. Podemos simplemente decirle “Si piensas que no es una comida, entrega este resultado..”
- Fine-Tuning: Podriamos entrenar un LLM con datos de recetarios
- Document: Podriamos hacer que el LLM considere un libro de recetas en particular a la hora de crear su respuesta. Quizá también agregarle productos propios del supermercado para incentivar su compra. Esto es desafiante debido a que un documento tiene muchos caracteres y los LLM tienen un limite llamado `context_length`. Afortunadamente ya existen metodologías para sobrellevar esto.

# 🥳 Conclusión

En este blog exploramos el mundo de los Large Language Models (LLM) y su aplicación en la vida cotidiana. El LLM más conocido, ChatGPT, es un modelo matemático capaz de comprender y generar lenguaje humano de manera avanzada. Descubrimos cómo utilizar los LLM para obtener recetas y extraer los ingredientes de forma estructurada, utilizando técnicas de Prompt Engineering.

Mediante el uso de herramientas como LangChain, Python y Gradio, pudimos construir una solución que permite al usuario ingresar el nombre de una comida y obtener automáticamente los ingredientes necesarios para cocinarla. Utilizando Prompt Templates y Chains, logramos interactuar con el modelo de manera eficiente y obtener respuestas precisas.

Aunque esta solución es solo un primer paso en el mundo de los LLM, demuestra el potencial de estos modelos para simplificar tareas y mejorar nuestra vida diaria. A medida que la tecnología avance, seguiremos explorando nuevas aplicaciones y técnicas para aprovechar al máximo los Large Language Models en beneficio de la humanidad.

Recuerda que en github podrás encontrar el código utilizado en este proyecto.
