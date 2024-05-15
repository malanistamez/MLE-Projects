
# Python Web Scraping Tutorial

## Introduction

This tutorial provides a step-by-step guide to web scraping in Python using the `requests` library for fetching web pages and `BeautifulSoup` for parsing HTML content. We'll cover the basics of web scraping, essential Python libraries, and techniques to extract data from web pages.

## Prerequisites

Before diving into web scraping, ensure you have the following prerequisites installed:

- Python 3.x
- `requests` library
- `BeautifulSoup` library

You can install these libraries using pip:

```bash
pip install requests beautifulsoup4
```

## Getting Started

To begin, let's understand the basic components of web scraping:

1. **Fetching HTML**: Use the `requests` library to retrieve the HTML content of a web page.
2. **Parsing HTML**: Utilize `BeautifulSoup` to parse the HTML content and extract relevant data.
3. **Data Extraction**: Extract specific data from the parsed HTML using BeautifulSoup methods.
4. **Data Storage**: Store the extracted data in a suitable format, such as CSV or JSON.

## Code Overview

Here's a breakdown of the main components in our Python web scraping code:

- **`get_data(url)`:** This function fetches the HTML content of the specified URL using the `requests` library and parses it using BeautifulSoup. It then extracts the table of contents from the page and returns a list of dictionaries containing heading numbers and text.
- **`export_data(data, file_name)`:** This function exports the extracted data to a CSV file specified by `file_name`.
- **`main()`:** The main function orchestrates the entire scraping process. It defines the URLs to scrape, calls the `get_data` function for each URL, and exports the data to CSV files.

## Running the Code

To run the web scraping code, execute the `main()` function. Ensure that you have an active internet connection, as the code fetches data from web pages. After execution, you'll find CSV files containing the extracted data for each URL.

## Conclusion

Web scraping is a powerful technique for extracting data from websites. With Python and libraries like `requests` and `BeautifulSoup`, you can automate the process of gathering information from web pages efficiently. Use this tutorial as a foundation to explore more advanced web scraping techniques and build robust data extraction pipelines.
```