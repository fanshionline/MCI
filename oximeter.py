from socket import *
import select
import threading
from time import sleep


class oximeter:
    end = False
    id = 0
    hr = 0

    def __init__(self, port_oximeter=10001):
        oxi = threading.Thread(name='oximeter', target=self.oximeter, args=(port_oximeter,))
        oxi.start()

    def oximeter(self, port_oximeter):
        tcpCliSock3 = socket(AF_INET, SOCK_STREAM)
        tcpCliSock3.setblocking(False)
        tcpCliSock3.bind(('', port_oximeter))
        print('oximeter connecting')
        tcpCliSock3.listen()
        print('oximeter is ready.')
        inputs = [tcpCliSock3, ]
        while not self.end:
            r_list, w_list, e_list = select.select(inputs, [], [], 0.005)
            for event in r_list:
                if event == tcpCliSock3:
                    new_sock, addr = event.accept()
                    # print('oximeter connected')
                    inputs = [tcpCliSock3, new_sock, ]
                else:
                    data = event.recv(1024)
                    # print(data)
                    # logger.info(data)
                    if data != b'' and data != b'socket connected':
                        # logger.info(data)
                        oximeter_result = data.split()
                        try:
                            self.id = bytes.decode(oximeter_result[-1])[-1]  # 1,2
                            # if oximeter_list[oximeter_tag_num - 1] in result.keys():
                            self.hr = int(oximeter_result[-2])
                            # logger.info('oximeter wrong')
                        except:
                            print('oximeter wrong!')

    def reset(self):
        self.end = True

    def __del__(self):
        self.end = True


if __name__ == '__main__':
    oxi = oximeter()
    t = 30
    while t > 0:
        print(oxi.hr)
        t -= 1
        sleep(1)

    oxi.reset()
