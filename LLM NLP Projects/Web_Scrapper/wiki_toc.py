import csv
import requests
from bs4 import BeautifulSoup

def get_sections(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find the main content element
            content = soup.find("div", id="mw-content-text")
            if content:
                # Find all subsections within the content
                subsections = content.find_all(["h2", "h3", "h4"])
                if subsections:
                    sections = []
                    for subsection in subsections:
                        # Extract section title and remove any citation numbers
                        section_title = subsection.get_text(strip=True).split("[")[0]
                        sections.append(section_title)
                    return sections
                else:
                    print("No subsections found in the main content.")
            else:
                print("Main content element not found on the page.")
        else:
            print("Failed to retrieve page:", url)
    except Exception as e:
        print("An error occurred:", e)
    return []

def export_sections(sections, file_name):
    try:
        with open(file_name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Section Title"])
            writer.writerows([[section] for section in sections])
        print(f"Sections exported to '{file_name}' successfully.")
    except Exception as e:
        print("An error occurred while writing to CSV:", e)

def main():
    urls = [
        ("https://en.wikipedia.org/wiki/Python_(programming_language)", "python_sections.csv"),
        ("https://en.wikipedia.org/wiki/Web_scraping", "web_scraping_sections.csv")
    ]
    for url, file_name in urls:
        print(f"Extracting sections from: {url}")
        sections = get_sections(url)
        if sections:
            export_sections(sections, file_name)
        else:
            print(f"No sections found for {url}")

if __name__ == '__main__':
    main()
