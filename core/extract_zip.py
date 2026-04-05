import zipfile
import xml.etree.ElementTree as ET
import sys
import re

try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        xml_content = z.read('word/document.xml').decode('utf-8')
        # simple regex to strip all XML tags
        text = re.sub('<[^<]+>', ' ', xml_content)
        # replace multiple spaces
        text = re.sub(' +', ' ', text)
        with open('resume_zip.txt', 'w', encoding='utf-8') as f:
            f.write(text)
except zipfile.BadZipFile:
    print("Not a zip file, maybe it's a .doc file, or corrupted.")
    try:
        with open(sys.argv[1], 'rb') as f:
            content = f.read(1000)
            print("Magic bytes:", content[:10])
    except Exception as e:
        print("Error reading file:", e)
except Exception as e:
    print(f"Other error: {e}")
