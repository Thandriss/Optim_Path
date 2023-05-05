from http.server import HTTPServer
from server.server import Server

if __name__ == "__main__":
    port = 8080
    hostname = "0.0.0.0"
    server = HTTPServer((hostname, port), Server)
    print("Server started http://%s:%s" % (hostname, port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    print("Server stopped.")
