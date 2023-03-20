import os
import re
import requests


class BibFileModifier:
    def __init__(self, filename):
        self.filename = filename

    def format_author(self, author):
        name_parts = [part.capitalize() for part in author.split()]
        formatted_name = ' '.join(name_parts)
        return formatted_name

    def format_publisher(self, publisher):
        parts = [part.capitalize() for part in publisher.split()]
        formatted_publisher = ' '.join(parts)
        return formatted_publisher

    def get_publisher_address(self, publisher):
        url = f'https://api.crossref.org/prefixes/{publisher}/works'
        response = requests.get(url)
        data = response.json()
        if 'message' in data and 'items' in data['message']:
            item = data['message']['items'][0]
            if 'publisher' in item and 'location' in item['publisher']:
                return item['publisher']['location']
        return None

    def modify_file(self):
        version = 1
        new_filename = f"{self.filename[:-4]}_modified_{version}.bib"
        while os.path.exists(os.path.join('/home/ak/Documents', new_filename)):
            version += 1
            new_filename = f"{self.filename[:-4]}_modified_{version}.bib"

        modified_lines = []
        with open(os.path.join('/home/ak/Documents', self.filename), 'r') as file:
            for line in file:
                if line.startswith('author'):
                    authors = [self.format_author(author.strip()) for author in line[8:-2].split(' and ')]
                    new_line = f'author = {{{" and ".join(authors)}}},\n'
                    print(f'Original line: {line.strip()}')
                    print(f'Modified line : {new_line.strip()}\n')
                    modified_lines.append(new_line)
                elif line.startswith('journal'):
                    journal_name = line[9:-2]
                    if not journal_name:
                        continue
                    journal_name = journal_name.lower()
                    journal_name = ' '.join(
                        [word.capitalize() if i == 0 else word.lower() for i, word in enumerate(journal_name.split())])
                    new_line = f'journal = {{{journal_name}}},\n'
                    print(f'Original line: {line.strip()}')
                    print(f'Modified line : {new_line.strip()}\n')
                    modified_lines.append(new_line)
                elif line.startswith('publisher'):
                    publisher_name = line[11:-2]
                    if not publisher_name:
                        continue
                    publisher_name = self.format_publisher(publisher_name)
                    publisher_address = self.get_publisher_address(publisher_name)
                    if publisher_address:
                        new_line = f'publisher = {{{publisher_name}}},\n'
                        modified_lines.append(new_line)
                        new_line = f'address = {{{publisher_address}}},\n'
                        print(f'Original line: {line.strip()}')
                        print(f'Modified line : {new_line.strip()}\n')
                        modified_lines.append(new_line)
                    else:
                        print(f'Could not find address for publisher: {publisher_name}\n')
                elif line.startswith('booktitle'):
                    booktitle = line[11:-2]
                    if not booktitle:
                        continue
                    booktitle = booktitle.lower()
                    booktitle = ' '.join(
                        [word.capitalize() if i == 0 else word.lower() for i, word in enumerate(booktitle.split())])
                    new_line = f'booktitle = {{{booktitle}}},\n'
                    print(f'Original line: {line.strip()}')
                    print(f'Modified line : {new_line.strip()}\n')
                    modified_lines.append(new_line)
                elif line.startswith('doi'):
                    dois = [doi.strip() for doi in line[5:-2].split(',')]
                    new_line = f'doi = {{{",".join(dois)}}},\n'
                    print(f'Original line: {line.strip()}')
                    print(f'Modified line : {new_line.strip()}\n')
                    modified_lines.append(new_line)
                else:
                    modified_lines.append(line)

        with open(os.path.join('/home/ak/Documents', new_filename), 'w') as new_file:
            new_file.writelines(modified_lines)

        print(f'Modified file written to {new_filename}')


def main(filename):
    bib_modifier = BibFileModifier(filename)
    bib_modifier.modify_file()


if __name__ == '__main__':
    filename = 'stylized_modified_v3.bib'
    main(filename)
