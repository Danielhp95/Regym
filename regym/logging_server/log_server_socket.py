import sys
import pickle
import struct
from multiprocessing import Process
import logging
import logging.handlers
import socketserver
import socket
import select

SERVER_SHUTDOWN_MESSAGE = 'LOG_SERVER_SHUTDOWN_REQUEST'


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
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
        self.server.logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    '''
    TCP Server which serves until shutdown message arrives
    from queue.
    '''

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler,
                 log_path=None, Queue=None):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.timeout = 1
        self.logger = self.initialize_root_log_server_socket(log_path)

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

    def initialize_root_log_server_socket(self, log_path):
        '''
        IMPORTANT: the logger MUST be the root logger, otherwise
        the handlers attached to it will be ignored

        Initializes root logger to print to standard output
        and to a log file located in :param: log_path
        :param log_path: Path where log file will be saved
        '''
        logger          = logging.getLogger('ServerLogger')
        console_handler = logging.StreamHandler(stream=sys.stdout)
        file_handler    = logging.FileHandler(filename=log_path)
        log_format      = logging.Formatter(fmt='[%(asctime)s]:%(levelname)s:%(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')

        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def serve_until_stopped(self):
        while True:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()


def serve_logging_server_forever(log_path: str):
    '''
    Starts a TCPServer that spawns a new process for each connection
    (analogous to creating a new connection per logger).
    TODO: figure out a way of shutting the server down upon completion
    :param log_path: Path where log file will be saved
    '''
    tcpserver = LogRecordSocketReceiver(host='localhost',
                                        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                                        handler=LogRecordStreamHandler,
                                        log_path=log_path)
    tcpserver.serve_until_stopped()


def create_logging_server_process(log_path: str) -> Process:
    '''
    :param log_path: Path to file where server will dump logs
    :returns: Started process hosting logging server.
    '''
    p = Process(target=serve_logging_server_forever,
               args=(log_path,),
               daemon=True)
    p.start()
    return p


def initialize_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    '''
    Creates a logger with :param: name and :param: level
    that connects to regym's logging server. This logger features
    a handler that forwards all logging calls to server.

    :param name: Name to be given to logger
    :param level: Logging level at which logs will be sent over to server
    :returns: logger with added server handler
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    socketHandler = logging.handlers.SocketHandler(
        host='localhost',
        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    logger.addHandler(socketHandler)
    return logger
