import os
import requests
from bs4 import BeautifulSoup


class BibEntry:
    def __init__(self, filename):
        self.filename = filename
        with open(os.path.join('/home/ak/Documents', filename), 'r') as file:
            self.entry = file.read()

    def _format_author(self, author):
        return ' '.join([name.capitalize() for name in author.split()])

    def _format_publisher(self, publisher):
        return ' '.join([word.capitalize() for word in publisher.split()])

    def _get_publisher_address(self, publisher):
        query = '+'.join(publisher.split())
        url = f"https://www.google.com/search?q={query}&oq={query}&aqs=chrome.0.35i39l2j0l4j46j69i60.6299j1j7&sourceid=chrome&ie=UTF-8"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        try:
            result = soup.find("div", class_="BNeawe s3v9rd AP7Wnd").get_text()
            return result.split(',')[0]
        except:
            return None

    def _modify_entry(self):
        for i, line in enumerate(self.entry.split('\n')):
            if '=' not in line:
                continue
            key, value = [elem.strip() for elem in line.split('=')]
            if key == 'author':
                authors = [self._format_author(author.strip()) for author in value.split(' and ')]
                self.entry = self.entry.replace(line, f'{key}={{' + ' and '.join(authors) + '}},')
            elif key == 'journal':
                journal_name = value.lower()
                journal_name = ' '.join(
                    [word.capitalize() if i == 0 else word.lower() for i, word in enumerate(journal_name.split())])
                self.entry = self.entry.replace(line, f'{key}={{' + journal_name + '}},')
            elif key == 'publisher':
                publisher_name = self._format_publisher(value)
                publisher_address = self._get_publisher_address(publisher_name)
                if publisher_address:
                    self.entry = self.entry.replace(line,
                                                    f'{key}={{' + publisher_name + '}},\naddress={{' + publisher_address + '}},')
                else:
                    print(f"Could not find address for publisher: {publisher_name}")
            elif key == 'doi':
                if not value:
                    print("Could not find DOI for this entry")
                else:
                    print(f"DOI for this entry: {value}")
            else:
                print(f"Unsupported key: {key}")


def main(filename):
    with open(os.path.join('/home/ak/Documents', filename), 'r') as file:
        entry = file.read()
    bib_entry = BibEntry(entry)
    bib_entry._modify_entry()
    bib_entry.save_modified(filename)

if __name__ == '__main__':
    filename = 'stylized_modified_v3.bib'
    main(filename)