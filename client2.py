
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 8089)
client.connect(server_address)

i = 0
while True:
    
    i+= 1
    num = str(i)+'\r\n'
    client.send(num.encode())