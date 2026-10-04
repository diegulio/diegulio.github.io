# diegulio.github.io

Sitio estático sin build (HTML/CSS/JS vanilla). Ya no se usa Quarto. Para previsualizar: `python3 -m http.server 4188 --bind 127.0.0.1`.

- `index.html` + `home.css` + `night.css` + `life-scene.js/css`: portada.
- `app.js`: catálogo desde `posts/posts.json` (filtros y búsqueda).
- `post.html?slug=<slug>`: plantilla única; renderiza `posts/<slug>/index.md` con marked.js (frontmatter YAML > posts.json).
- `style.css`: estilos de artículos.
- `resume.html` + `trayectoria.js/css`: la trayectoria como libro pop-up pixel art en Three.js (CDN jsdelivr). Cada `<li data-scene>` es un capítulo; su escena se dibuja en `SCENES` de `trayectoria.js`. Sin WebGL queda como lista legible. Sigue `data-theme` (día/noche) y el atardecer del home. `CV.pdf`.

## Flujo: pasar un post de Notion al sitio

Hacer esto cuando el usuario pida traspasar una página de Notion (pasa una URL o un título).

1. **Obtener el contenido.** Con el MCP de Notion (`API-retrieve-page-markdown`, o `API-get-block-children` si falla). Si el usuario entrega un export `.md` + carpeta de imágenes, usar eso.
2. **Slug**: minúsculas, con guiones, sin acentos (ej. `chatgpt-sin-tecnicismos`). Crear `posts/<slug>/index.md`.
3. **Frontmatter** al inicio de `index.md`:
   ```
   ---
   title: "..."
   subtitle: "..."
   author: "Diegulio"
   date: "YYYY-MM-DD"
   categories: [llm, explained]
   image: "posts/<slug>/portada.png"
   ---
   ```
   Si el título se repite como `# Título` en el cuerpo, dejarlo solo si los demás posts lo hacen. Revisar `posts/mcp/index.md` como referencia.
4. **Imágenes**: las URLs de Notion/S3 expiran. Descargar cada imagen a `posts/<slug>/` (o `posts/<slug>/assets/`) y referenciarla con ruta relativa a esa carpeta (`![](assets/x.png)`); `post.html` antepone `posts/<slug>/` solo. Nombres sin espacios si se puede. Nunca dejar URLs `notion.so` ni `amazonaws.com`.
5. **Portada**: la primera imagen o una que indique el usuario, guardada como `posts/<slug>/portada.*`.
6. **Contenido a vigilar**: callouts (`> [!NOTE]`), bloques de código con lenguaje (```python), ecuaciones (verificar que rendericen), tablas, toggles y embeds (convertir a HTML/links), videos (`.mov` pesados → gif/mp4 comprimido).
7. **Registrar en `posts/posts.json`**: añadir la entrada al principio, con `slug, title, subtitle, date, author, categories, image, "type": "local"`. Categorías ya usadas: llm, python, pytorch, nlp, mcp, explained, paper, langchain, kaggle, application, library, lightning. Si hace falta una nueva, añadirla al mapa `labels` en `app.js`.
8. **Verificar**: levantar el servidor local y abrir `post.html?slug=<slug>` y la portada; confirmar que cargan imágenes y que no hay errores de consola.
9. **No hacer commit ni push sin que el usuario lo pida.**

## Despliegue

GitHub Pages sirve este repo. Antes se publicaba la rama `gh-pages` (Quarto). Ahora el sitio vive en la raíz de `main`: en Settings → Pages hay que poner Source = "Deploy from a branch", rama `main`, carpeta `/ (root)`. Añadir `.nojekyll` si algún archivo o carpeta empieza con `_`.
