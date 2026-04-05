import docx
import sys

doc = docx.Document(sys.argv[1])
for para in doc.paragraphs:
    print(para.text)
