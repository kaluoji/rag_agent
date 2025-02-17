import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async def scrape_all_links(base_url, max_pages=10):
    async with AsyncWebCrawler() as crawler:
        all_links = set()
        current_url = base_url

        for page in range(1, max_pages + 1):
            print(f"Scrapeando página {page}: {current_url}")

            # Configuración del crawler
            config = CrawlerRunConfig(
                js_code="""
                    // Buscar enlaces con clase page-link que no sean la página actual
                    const nextPageLinks = Array.from(document.querySelectorAll('a.page-link[href*="?page="]'));
                    const currentPageLink = nextPageLinks.find(link => link.classList.contains('page-item is-active'));
                    const nextPageLink = nextPageLinks.find(link => !link.classList.contains('page-item is-active'));
                    if (nextPageLink) {
                        nextPageLink.click();
                    }
                """,
                wait_for=".pagination.js-pager__items",  # Esperamos al contenedor de paginación
                process_iframes=True,
                exclude_external_links=False,
            )

            try:
                # Ejecuta el crawler
                result = await crawler.arun(url=current_url, config=config)

                # Agrega los enlaces encontrados al conjunto
                if result and result.links:
                    # Filtramos enlaces que no sean de navegación
                    content_links = {link for link in result.links 
                                   if isinstance(link, str) and 
                                   not link in ['external', 'internal'] and
                                   '/databases-library/esma-library' in link}
                    all_links.update(content_links)
                    print(f"Enlaces encontrados en página {page}: {len(content_links)}")

                # Construye la URL de la siguiente página
                next_page = page + 1
                next_page_url = f"{base_url}?page={next_page}"
                
                # Verifica si la siguiente página existe en los enlaces
                page_exists = False
                if result and result.links:
                    for link in result.links:
                        if isinstance(link, str) and f"page={next_page}" in link:
                            page_exists = True
                            current_url = link
                            break

                if not page_exists:
                    # Intentar una estrategia de fallback
                    print("No se encontró un enlace a la siguiente página. Intentando estrategia de fallback.")
                    next_page_url = None
                    for link in result.links:
                        if isinstance(link, str) and "page=" in link:
                            next_page_url = link
                            current_url = link
                            page_exists = True
                            break
                    
                    if not next_page_url:
                        print("No se pudo encontrar un enlace a la siguiente página. Fin del scraping.")
                        break

            except Exception as e:
                print(f"Error al procesar la página {page}: {str(e)}")
                break

            # Pequeña pausa entre páginas
            await asyncio.sleep(2)

        return all_links

if __name__ == "__main__":
    base_url = "https://www.eiopa.europa.eu/document-library/guidelines_en"
    max_pages = 10

    # Ejecuta el scraper
    links = asyncio.run(scrape_all_links(base_url, max_pages))

    # Imprime todos los enlaces encontrados
    print(f"\nTotal de enlaces encontrados: {len(links)}")
    for link in sorted(links):
        print(link)