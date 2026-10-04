/* Keep the existing resume navigation working with this shared script. */
const legacyNavbar = document.getElementById('navbar');
if (legacyNavbar) {
  const updateNavbar = () => legacyNavbar.classList.toggle('scrolled', window.scrollY > 16);
  window.addEventListener('scroll', updateNavbar, {passive:true});
  updateNavbar();
}

/* The index stays usable independently of the WebGL scene. */
if (document.getElementById('posts-grid')) {
const grid = document.getElementById('posts-grid');
const filters = document.getElementById('filter-bar');
const search = document.getElementById('post-search');
let posts = [];
let activeFilter = 'all';
document.getElementById('year').textContent = new Date().getFullYear();
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const normalize = value => value.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
const postURL = post => post.type === 'local' ? `post.html?slug=${encodeURIComponent(post.slug)}` : post.url;
const dateLabel = value => new Date(`${value}T12:00:00`).toLocaleDateString('es', {month:'short',year:'numeric'}).replace('.', '');
const labels = {llm:'LLMs',python:'Python',pytorch:'PyTorch',nlp:'NLP',mcp:'MCP',explained:'Conceptos',paper:'Paper → código',langchain:'LangChain',kaggle:'Kaggle',application:'Aplicaciones',library:'Open source',lightning:'Lightning'};
function media(post, className, eager = false) {
  return `<div class="${className}${post.image ? '' : ' media-failed'}">${post.image ? `<img src="${escapeHTML(post.image)}" alt="${escapeHTML(post.title)}" loading="${eager ? 'eager' : 'lazy'}" decoding="async" />` : ''}</div>`;
}
function bindImageFallbacks(root) {
  root.querySelectorAll('img').forEach(img => {
    const fallback = () => img.parentElement.classList.add('media-failed');
    img.addEventListener('error', fallback, {once:true});
    if (img.complete && !img.naturalWidth) fallback();
  });
}
function renderFeatured(post) {
  const target = document.getElementById('featured-post');
  target.innerHTML = `<a class="featured" href="${escapeHTML(postURL(post))}">
    ${media(post, 'feature-media', true)}<span class="feature-label">↗ EMPIEZA POR AQUÍ</span>
    <div class="feature-copy"><div class="post-kicker"><span class="post-tag">${escapeHTML(labels[post.categories[0]] || post.categories[0])}</span><span>${escapeHTML(dateLabel(post.date))} / ÚLTIMA PUBLICACIÓN</span></div>
    <h3>${escapeHTML(post.title)}</h3><p>${escapeHTML(post.subtitle)}</p><div class="feature-link">Abrir el cuaderno <span aria-hidden="true">↗</span></div></div></a>`;
  bindImageFallbacks(target);
}
function renderPosts() {
  const query = normalize(search.value.trim());
  const matches = posts.filter(post => (activeFilter === 'all' || post.categories.includes(activeFilter)) && normalize(`${post.title} ${post.subtitle} ${post.categories.join(' ')}`).includes(query));
  document.getElementById('post-count').textContent = String(matches.length).padStart(2,'0');
  document.getElementById('results-status').textContent = matches.length === 1 ? '1 artículo encontrado' : `${matches.length} artículos encontrados`;
  grid.innerHTML = matches.length ? matches.map(post => `<article class="post-card"><a class="post-card-link" href="${escapeHTML(postURL(post))}">
    ${media(post, 'post-media')}<span class="post-index">EXP. ${String(posts.indexOf(post) + 1).padStart(2,'0')}</span>
    <div class="post-kicker"><span>${escapeHTML(dateLabel(post.date))}</span><span>—</span><span>${escapeHTML(post.categories.slice(0,2).map(cat => labels[cat] || cat).join(' / '))}</span></div>
    <h4>${escapeHTML(post.title)}</h4><p>${escapeHTML(post.subtitle)}</p><div class="card-bottom"><span>EXPLORAR ARTÍCULO</span><span aria-hidden="true">↗</span></div></a></article>`).join('') : '<div class="empty-results">No hay artículos con esa combinación. Probemos otra idea.<button id="reset-search" type="button">Ver todos los artículos</button></div>';
  bindImageFallbacks(grid);
  document.getElementById('reset-search')?.addEventListener('click', () => { search.value = ''; setFilter('all'); search.focus(); });
}
function setFilter(filter) {
  activeFilter = filter;
  filters.querySelectorAll('button').forEach(button => {
    const selected = button.dataset.filter === filter;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  renderPosts();
}
async function initPosts() {
  try {
    const response = await fetch('posts/posts.json');
    if (!response.ok) throw new Error(`Posts: ${response.status}`);
    posts = (await response.json()).sort((a,b) => b.date.localeCompare(a.date));
    document.getElementById('nav-count').textContent = String(posts.length).padStart(2,'0');
    if (posts.length) renderFeatured(posts[0]);
    const categories = ['all', ...new Set(posts.flatMap(post => post.categories))];
    const preferred = ['all','llm','python','pytorch','nlp','mcp','kaggle'];
    filters.innerHTML = preferred.filter(cat => categories.includes(cat)).map(cat => `<button class="filter-button${cat === activeFilter ? ' active' : ''}" type="button" data-filter="${cat}" aria-pressed="${cat === activeFilter}">${cat === 'all' ? 'Todos los experimentos' : labels[cat]}</button>`).join('');
    renderPosts();
  } catch (error) {
    console.error(error);
    grid.innerHTML = '<p class="loading-note">No pudimos abrir los artículos.<button id="retry-posts" type="button">Volver a intentar</button></p>';
    document.getElementById('retry-posts').addEventListener('click', initPosts);
  }
}
filters.addEventListener('click', event => { const button = event.target.closest('[data-filter]'); if (button) setFilter(button.dataset.filter); });
search.addEventListener('input', renderPosts);
initPosts();
}
