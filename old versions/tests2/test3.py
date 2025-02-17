import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async def scrape_all_links(base_url, max_pages=10):
    async with AsyncWebCrawler() as crawler:
        all_links = set()

        for page in range(1, max_pages + 1):
            page_url = f"{base_url}?page={page}"
            print(f"Scrapeando página {page}: {page_url}")

            # Configuración del crawler
            config = CrawlerRunConfig(
                process_iframes=True,
                exclude_external_links=False,
            )

            try:
                # Ejecuta el crawler
                result = await crawler.arun(url=page_url, config=config)

                # Agrega los enlaces encontrados al conjunto
                if result and result.links:
                    # Filtramos enlaces que no sean de navegación
                    content_links = {link for link in result.links 
                                   if isinstance(link, str) and 
                                   not link in ['external', 'internal'] and
                                   '/databases-library/esma-library' in link}
                    all_links.update(content_links)
                    print(f"Enlaces encontrados en página {page}: {len(content_links)}")

                    # Mostrar un fragmento del contenido de la página
                    print(f"Fragmento del contenido de la página {page}:")
                    print(result.content[:500] + "...")


            except Exception as e:
                print(f"Error al procesar la página {page}: {str(e)}")
                break

            # Pequeña pausa entre páginas
            await asyncio.sleep(2)

        return all_links

if __name__ == "__main__":
    base_url = "https://www.esma.europa.eu/databases-library/esma-library"
    max_pages = 10

    # Ejecuta el scraper
    links = asyncio.run(scrape_all_links(base_url, max_pages))

    # Imprime todos los enlaces encontrados
    print(f"\nTotal de enlaces encontrados: {len(links)}")
    for link in sorted(links):
        print(link)