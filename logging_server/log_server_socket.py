import sys
import pickle
import struct
import logging
import logging.handlers
import socketserver


class LogServerStreamHandler(socketserver.StreamRequestHandler):
    '''
    Handler for a streaming logging request.
    '''

    def handle(self):
        '''
        Waits for packets to be sent via the oppen socket connection
        and logs them once they have been received
        '''
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0] # '>L' format stands for big-endian (>) unsigned long (L)
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = pickle.loads(chunk)
            record = logging.makeLogRecord(obj)
            self.handleRecord(record)

    def handleRecord(self, record):
        '''
        Logs the :param: record according the logging handlers specified upon initialization
        :param record: LogRecord received from open socket from a distant logger
        '''
        logger = logging.getLogger('ServerLogger')
        logger.handle(record)


def initialize_root_log_server_socket(log_path):
    '''
    IMPORTANT: the logger MUST be the root logger, otherwise
    the handlers attached to it will be ignored

    Initializes root logger to print to standard output
    and to a log file located in :param: log_path
    :param log_path: Path where log file will be saved
    '''
    logger          = logging.getLogger('')
    console_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler    = logging.FileHandler(filename=log_path)
    log_format      = logging.Formatter(fmt='[%(asctime)s]%(levelname)s:%(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def serve_logging_server_forever(log_path):
    '''
    Starts a TCPServer that spawns a new thread for each connection
    (analogous to creating a new connection per logger).

    TODO: figure out a way of shutting the server down upon completion
    :param log_path: Path where log file will be saved
    '''
    initialize_root_log_server_socket(log_path)
    tcpserver = socketserver.ThreadingTCPServer(('localhost', logging.handlers.DEFAULT_TCP_LOGGING_PORT),
                                                LogServerStreamHandler)
    tcpserver.serve_forever()
