import os

from requests_html import HTMLSession
import PyPDF2
import requests
from tqdm import tqdm


def extract_links_pdf(fname):
    PDFFile = open(fname, 'rb')

    PDF = PyPDF2.PdfFileReader(PDFFile)
    pages = PDF.getNumPages()
    key = '/Annots'
    uri = '/URI'
    ank = '/A'

    urls = []
    print(f'Parse PDF: {fname}')
    for page in range(pages):
        # print("Current Page: {}".format(page))
        pageSliced = PDF.getPage(page)
        pageObject = pageSliced.getObject()
        if key in pageObject.keys():
            ann = pageObject[key]
            for a in ann:
                u = a.getObject()
                if uri in u[ank].keys():
                    link = u[ank][uri]
                    # print(f'\r{link}...', end='')
                    urls.append(link)

    print(f'Found {len(urls)} links.')
    return urls


def check_integrity(fname):
    PDFFile = open(fname, 'rb')
    try:
        PDF = PyPDF2.PdfFileReader(PDFFile)
        pages = PDF.getNumPages()
        if pages > 0:
            return True
        else:
            return False
    except:
        return False


# def download_file(url, dest_fname):
#     local_filename = dest_fname
#     # NOTE the stream=True parameter below
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 if chunk:  # filter out keep-alive new chunks
#                     f.write(chunk)
#                     # f.flush()
#     return local_filename

def download_file(url, dest, chunck=2000):
    fileb = requests.get(url, stream=True)
    chunk_size = chunck  # bytes
    with open(dest, 'wb') as f:
        for chunk in tqdm(fileb.iter_content(chunk_size)):
            f.write(chunk)


def run():
    pdf = 'springer-ebook.pdf'
    urls = extract_links_pdf(pdf)

    session = HTMLSession()
    # key = 'isbn='
    key = '10.1007%2F'
    for i, url in enumerate(urls):
        print(f'Url: {i}/{len(urls)}')
        # url = 'https://link.springer.com/book/10.1007%2F978-0-387-21736-9'

        r = session.get(url)
        r_tmp = requests.get(url)

        # file_url_name = url.split('/')[-1]
        # key = 'isbn='
        # file_url_name = file_url_name[file_url_name.index(key)+len(key):]

        try:
            display_name = r.html.text[:r.html.text.index('|')].strip()
        except ValueError:
            print('NAME NOT FOUND')
            display_name = r.html.text[:50] if len(r.html.text) > 50 else r.html.text

        # OLD METHOD
        # parsing links
        # links = r.html.links
        # link_pdf = [l for l in links if l.endswith(f'{file_url_name}.pdf')]
        # all_links_pdf = [l for l in links if l.endswith('.pdf')]
        # Hypothesis: shortest link has no chapter, i.e. is the file
        # m = float('inf')
        # best = all_links_pdf[0]
        # for l in all_links_pdf:
        #     if len(l) < m:
        #         best = l
        #         m = len(l)
        # link_pdf = best
        # link_pdf = f'https://link.springer.com{link_pdf}'

        # ok method
        try:
            redirect = r_tmp.url
            tiny_url = redirect[redirect.index(key) + len(key):]

            link_pdf = f'https://link.springer.com/content/pdf/10.1007%2F{tiny_url}'

            print(f'Downloading \"{display_name}\" from {link_pdf}')
            display_name = display_name.replace(' ', '_')
            target_file = f'pdfs/{display_name}.pdf'
            print(f'Writing content to \"{display_name}\"..')
            if not os.path.exists(target_file):
                download_file(link_pdf, target_file)
            else:
                print('Target file exists checking for corruption...', end='')
                # TODO: check if pdf is correctly downloaded
                # this is damaged, check "Business_Statistics_for_Competitive_Advantage_with_Excel_2016.pdf"
                is_ok = check_integrity(target_file)
                print('done.')

                if not is_ok:
                    print('File requires to be downloaded again')
                    download_file(link_pdf, target_file)

            # with open(f'pdfs/{display_name}.pdf', 'wb') as f:
            #     f.write(fileb.content)

        except Exception:
            import uuid

            if len(display_name) > 0:
                display_name = display_name.strip().replace(' ', '_')
            else:
                display_name = uuid.uuid4().__str__()

            print(f'Unable to retrieve link: downloading all pdfs available into folder {display_name}')
            dest = f'pdfs/{display_name}'
            os.makedirs(dest, exist_ok=True)

            links = r.html.links
            link_pdf = [l for l in links if l.endswith(f'.pdf')]
            for link in link_pdf:
                id = uuid.uuid4().__str__()
                fname = dest + f'/{id}.pdf'
                link = f"https://link.springer.com{link}"
                try:
                    download_file(link, fname)
                except requests.exceptions.MissingSchema:
                    print(f'Invalid url: {link}')

            # TODO: remove the least in size. (the most heavy should be the full file)
            continue


if __name__ == '__main__':
    run()
