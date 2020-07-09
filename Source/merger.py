import PyPDF2
from os import listdir


def merger():
    files = [f for f in listdir('TXT/fromws')]
    text = ''
    for f in files:
        if f.find(".txt") != -1:
            with open('TXT/fromws/' + f, 'r') as file:
                book = file.read()
            text += book

    with open('TXT/merged.txt', 'w+') as file:
        file.write(text)


if __name__ == '__main__':
    merger()
