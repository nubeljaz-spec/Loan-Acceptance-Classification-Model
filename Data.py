import requests
import pandas as pd
from io import BytesIO

url = "https://data.mendeley.com/public-files/datasets/tx2v248cx4/files/935a1c7c-11a6-4da6-b2f8-e02cc3a20eac/file_downloaded"

# Add a "User-Agent" header to look like a browser
headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(url, headers=headers)
resp.raise_for_status()  # will raise if not 200 OK

df = pd.read_csv(BytesIO(resp.content))
df.head()
