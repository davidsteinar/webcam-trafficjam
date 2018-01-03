import pprint
import pycurl
from io import BytesIO
import json

path = "/data"
data = BytesIO()
c = pycurl.Curl()
c.setopt(c.URL, 'https://webcamstravel.p.mashape.com/webcams/list/category=traffic/limit=10,1/country=SE?lang=en&show=webcams%3Aimage')
c.setopt(c.WRITEFUNCTION, data.write)
c.setopt(c.HTTPHEADER, ['X-Mashape-Key: MXoQyRzvhvmshzJOk2Oe9IweEpLfp1W1feUjsnLBt5yv7NJoVs'])
c.perform()
c.close()

body = data.getvalue()
dictionary = json.loads(body)
theList = dictionary.get("result",{}).get("webcams",{});

for item in dictionary.get("result",{}).get("webcams",{}):
    res = item.get("image", {}).get("current", {}).get("thumbnail", {})
    name = res.split('/')[-1]
    print(res)

    c = pycurl.Curl()
    c.setopt(c.URL, res)
    with open('data/' + name,'wb') as f:
    	c.setopt(c.WRITEFUNCTION, f.write)
    	c.perform()
