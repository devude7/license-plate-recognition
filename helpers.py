# filter out characters that do not occur on licence plates
def filter_ocr(text):
    new = ''
    for ch in text:
        if ch.isalnum():
            if ch.isupper() or ch.isdigit():
                new += ch
    return new

