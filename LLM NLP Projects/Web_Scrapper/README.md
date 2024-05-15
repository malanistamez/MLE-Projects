# Step-By-Step Tutorial on Python Web Scraping

@malanistamez

[![Oxylabs promo code](https://user-images.githubusercontent.com/129506779/250792357-8289e25e-9c36-4dc0-a5e2-2706db797bb5.png)](https://oxylabs.go2cloud.org/aff_c?offer_id=7&aff_id=877&url_id=112)

[![](https://dcbadge.vercel.app/api/server/eWsVUJrnG5)](https://discord.gg/GbxmdGhZjq)

## Table of Contents

- [Web Scraping in 5 Lines of Code](#web-scraping-in-5-lines-of-code)
- [Components of Web Scraping with Python Code](#components-of-web-scraping-with-python-code)
    - [Python Libraries](#python-libraries)
    - [Python Web Scraping: Working with Requests](#python-web-scraping-working-with-requests)
- [BeautifulSoup](#beautifulsoup)
- [Find Methods in BeautifulSoup4](#find-methods-in-beautifulsoup4)
    - [Finding Multiple Elements](#finding-multiple-elements)
    - [Finding Nested Elements](#finding-nested-elements)
    - [Exporting the data](#exporting-the-data)
- [Other Tools](#other-tools)

This tutorial delves into Python web scraping, providing a comprehensive guide from simple examples to more intricate processes.

Python's simplicity and extensive open-source libraries make it an optimal choice for web scraping tasks. These libraries facilitate data extraction and transformation into various formats, such as CSV, JSON, or direct database storage.

## Web Scraping in 5 Lines of Code

Execute the following five lines in a text editor, save them as a `.py` file, and run them with Python. Ensure that the required libraries are installed, as detailed later in this tutorial.

```python
import requests
from bs4 import BeautifulSoup
response = requests.get("https://en.wikipedia.org/wiki/Web_scraping")
bs = BeautifulSoup(response.text,"lxml")
print(bs.find("p").text)
```

This code snippet retrieves the first paragraph from the Wikipedia page on web scraping, showcasing Python's power and simplicity. You'll find this code in the `webscraping_5lines.py` file.

## Components of Web Scraping with Python Code

The fundamental components of any web scraping code are:

1. Obtaining HTML
2. Parsing HTML into Python objects
3. Saving the extracted data

Typically, browser loading of auxiliary files like images, CSS, and JavaScript is unnecessary for web scraping, as the focus is on data extraction. Python simplifies browser interaction when required.

## Python Libraries

Python's extensive library ecosystem significantly aids web scraping tasks. This tutorial focuses on three key libraries – requests, BeautifulSoup, and CSV.

- The [Requests](https://docs.python-requests.org/en/master/) library retrieves HTML files, eliminating the need for browser interaction.
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) parses raw HTML into Python objects. Version 4, known as `bs4` or `BeautifulSoup4`, is utilized here.
- The [CSV](https://docs.python.org/3/library/csv.html) library, part of Python's standard installation, facilitates data storage.

To install these libraries, use the following command in your terminal or command prompt:

```sh
pip install requests BeautifulSoup4 lxml
```

You might need to use `pip3` or include the `--user` switch depending on your OS and settings.

## Python Web Scraping: Working with Requests

The requests library negates browser loading by directly fetching webpage HTML. For instance:

```python
import requests

url_to_parse = "https://en.wikipedia.org/wiki/Large_language_model"
response = requests.get(url_to_parse)
print(response)
```

The output, such as `<Response (200)>`, signifies a successful response (status code 200). HTML extraction from the response is achieved using the `.text` attribute.

```python
print(response.text)
```

This yields the HTML string, ready for parsing.

## BeautifulSoup

Beautiful Soup simplifies HTML navigation, search, and modification, handling encoding automatically. It supports various Python parsers like lxml and html5lib.

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(response.text, 'lxml')
```

Accessing specific elements, such as the page title, becomes straightforward:

```python
print(soup.title)
print(soup.title.text)
```

## Find Methods in BeautifulSoup4

`find()` and `find_all()` are commonly used methods. For instance, to extract the table of contents from a Wikipedia page:

```python
table_of_contents = soup.find("div", id="toc")
```

Using `find_all()` retrieves multiple elements, such as all headings:

```python
headings = table_of_contents.find_all("li")
```

## Finding Multiple Elements

Utilize `find_all()` to extract multiple elements based on specific criteria. For example, to gather all heading text:

```python
headings_text = soup.find_all("span", class_="toctext")
```

Similarly, retrieve heading numbers:

```python
headings_number = soup.find_all("span", class_="tocnumber")
```

## Finding Nested Elements

Navigate through nested elements with ease. For instance, each heading number and text lies within an `li` tag:

```python
headings = table_of_contents.find_all("li")
```

Loop through these elements to create a structured dataset:

```python
data = []
for heading in headings:
    heading_text = heading.find("span", class_="toctext").text
    heading_number = heading.find("span", class_="tocnumber").text
    data.append({
        'heading_number': heading_number,
        'heading_text': heading_text,
    })
```

## Exporting the data

Export the structured data to a CSV file using the CSV module:

```python
import csv

with open("toc.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=['heading_number', 'heading_text'])
    writer.writeheader()
    writer.writerows(data)
```

This concludes the process, yielding a CSV file containing the structured data.

## Other Tools

Some websites employ JavaScript to load data dynamically, necessitating browser interaction. Selenium is an excellent solution for such scenarios. Refer to [this detailed guide on Selenium](https://en.wikipedia.org/wiki/Web_scraping) for further information.