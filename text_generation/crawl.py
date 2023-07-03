import requests
from bs4 import BeautifulSoup
import http.client


req = requests.get("https://voh.com.vn/song-dep/tho-kieu-435851.html")
soup = BeautifulSoup(req.text, 'html.parser')

div_1= soup.find("div", class_ ="news-detail-content_detail-content__goYFs")
text = div_1.find_all('p', style = "text-align: center;")

with open("truyen_kieu.txt", 'w', encoding='utf-8') as f:
    f.write(str(text))

f.close()

