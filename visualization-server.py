from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import urlparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load all data

file_name = sys.argv[2]
count = int(sys.argv[3])

descriptions = []
with open(file_name) as f:
    i = 0
    for line in f:
        i += 1
        if i > count:
            break
        try:
            descriptions.append(json.loads(line.replace('nan,', 'NaN,').replace('nan]', 'NaN]')))
        except Exception as e:
            print('An exception occurred while parsing:')
            print(e)
            continue

print("Loaded %d descriptions" % (len(descriptions),))

def color(activation):
    r = 0
    b = 0
    g = 0
    a = 0
    if activation > 0:
        r = 255
        a = 1 - 0.5 ** activation
    elif activation < 0:
        b = 255
        a = 1 - 0.5 ** -activation

    return 'rgba(%d, %d, %d, %f);' % (r, g, b, a)

class ListServer(BaseHTTPRequestHandler):
    def _set_headers(self, content_type = 'text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        print(url)

        if url.path == '/get-neuron'
            query = url.query
            query_components = dict(qc.split('=') for qc in query.split('&'))
            neuron_index = int(query_components['neuron'])


            self._set_headers()
            self.wfile.write(body)

httpd = HTTPServer(
    ('', int(sys.argv[1])),
    ListServer
)

print('Running server on 8080')

httpd.serve_forever()
