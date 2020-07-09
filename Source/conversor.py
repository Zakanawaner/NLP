import PyPDF2
from os import listdir


def get_text():
    files = [f for f in listdir('PDF/fromws')]
    text = ''
    for f in files:
        if f.find(".pdf") != -1:
            pdf_file_object = open('PDF/fromws/' + f, 'rb')
            pdf_reader = PyPDF2.PdfFileReader(pdf_file_object)
            pages = pdf_reader.numPages
            book = ''
            for p in range(pages):
                page_object = pdf_reader.getPage(p)
                book += page_object.extractText()
            pdf_file_object.close()
            beginning = book.find("www.luarna.com")
            book = book[beginning+15:]
            with open('TXT/' + f + '.txt', 'w+') as file:
                file.write(book)
            text += book

    with open('TXT/text.txt', 'w+') as file:
        file.write(text)


if __name__ == '__main__':
    get_text()
