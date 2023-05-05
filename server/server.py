import cgi
import threading
from http.server import BaseHTTPRequestHandler

from inferer_road import class_road
from new_picture import routing_calculation

ready_to_serve = False


class Server(BaseHTTPRequestHandler):
    with open('server/not_found.html', 'r') as file:
        not_found = file.read().rstrip()
        file.close()
    with open('server/not_yet.html', 'r') as file:
        not_yet = file.read().rstrip()
        file.close()
    with open('server/index.html', 'r') as file:
        index_page = file.read().rstrip()
        file.close()
    with open('server/calculation.html', 'r') as file:
        calculation_page = file.read().rstrip()
        file.close()
    with open('server/styles.css', 'r') as file:
        css_page = file.read().rstrip()
        file.close()
    with open('server/favicon.ico', 'rb') as file:
        favicon = file.read()
        file.close()

    def handle_get(self):
        if self.path == "/":
            self.handle_get_index()
        elif self.path.endswith("/styles.css"):
            self.handle_get_stylesheet()
        elif self.path.endswith("/favicon.ico"):
            self.handle_get_favicon()
        elif self.path == "/calc":
            self.handle_get_calc()
        elif self.path == "/route":
            self.handle_get_route()

    def handle_get_stylesheet(self):
        self.send_response(200)
        self.send_header("Content-type", "text/css")
        self.end_headers()
        self.wfile.write(bytes(self.css_page, encoding='utf8'))

    def handle_get_favicon(self):
        self.send_response(200)
        self.send_header("Content-type", "image/x-icon")
        self.end_headers()
        self.wfile.write(self.favicon)

    def handle_get_index(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.index_page, encoding='utf8'))

    def handle_get_not_yet(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.not_yet, encoding='utf8'))

    def handle_get_not_found(self):
        self.send_response(404)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(self.not_found, encoding='utf8'))

    def handle_get_calc(self):
        self.send_response(200)
        # cookie = SimpleCookie()
        # session_id = str(random.randint(0, 4294967295))
        # cookie['session-id'] = session_id
        # self.ready_to_serve[session_id] = False
        self.send_header("Content-type", "text/html")
        # for morsel in cookie.values():
        #     self.send_header("Set-Cookie", morsel.OutputString())
        self.end_headers()
        self.wfile.write(bytes(self.calculation_page, encoding='utf8'))
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        if ctype == 'multipart/form-data':
            fields = cgi.parse_multipart(self.rfile, pdict)
            image_map = fields['upload-image'][0]
            with open('server/image_map.png', 'wb') as file:
                file.write(image_map)
                file.close()
            image_heights = fields['upload-image-height'][0]
            with open('server/image_height.png', 'wb') as file:
                file.write(image_heights)
                file.close()
            start = fields['start'][0]
            end = fields['end'][0]
            liststart = start.split(',')
            startx = int(liststart[0])
            starty = int(liststart[1])
            listend = end.split(',')
            endx = int(listend[0])
            endy = int(listend[1])
            start_t = (startx, starty)
            end_t = (endx, endy)
        return start_t, end_t

    def do_GET(self):
        self.handle_get()

    def result(self, a, b, c, d, e):
        routing_calculation(a, b, c, d, e)
        global ready_to_serve
        ready_to_serve = True

    def classification(self, a, b, c):
        class_road(a, b, c)
        global ready_to_serve
        ready_to_serve = True

    def do_POST(self):
        if self.path == "/calc":
            startp, endp = self.handle_get_calc()
            # thread = threading.Thread(target=self.classification,
            #                          args=[".\\outputs\\test\\cfg.yaml", "server\\image_map.png", ".\\server\\result.png"])
            thread = threading.Thread(target=self.result,
                                      args=["server\\image_map.png", "server\\image_height.png",
                                            ".\\server\\result.png", startp, endp])
            thread.start()
        else:
            self.handle_get_not_found()

    def handle_get_route(self):
        global ready_to_serve
        if ready_to_serve:
            with open('server/result.png', 'rb') as file:
                image = file.read()
                file.close()
            self.send_response(200)
            self.send_header("Content-type", "image/png")
            self.end_headers()
            self.wfile.write(image)
        else:
            self.handle_get_not_yet()
