import requests

# url = 'https://api.catastrophe.world/natural-disasters/3392'
url = 'https://api.catastrophe.world/states/TX'

data = requests.get(url=url)

# print(data.content)
print(data.json())

data.close()