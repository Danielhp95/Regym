import sys
import pickle
import struct
import logging
import logging.handlers
import socketserver

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
            if record.getMessage() == SERVER_SHUTDOWN_MESSAGE:
                self.server.shutdown()
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
        self.shutdown_is_requested = False
        self.timeout = 1
        self.logger = self.initialize_root_log_server_socket(log_path)

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

    def shutdown(self):
        self.shutdown_is_requested = True

    def serve_until_stopped(self):
        import select
        while not self.shutdown_is_requested:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
        self.logger.warning('Server shutting down, no further logs will be saved')


def serve_logging_server_forever(log_path):
    '''
    Starts a TCPServer that spawns a new thread for each connection
    (analogous to creating a new connection per logger).
    TODO: figure out a way of shutting the server down upon completion :param log_path: Path where log file will be saved
    '''
    tcpserver = LogRecordSocketReceiver(host='localhost',
                                        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                                        handler=LogRecordStreamHandler,
                                        log_path=log_path)
    tcpserver.serve_until_stopped()


if __name__ == '__main__':
    serve_logging_server_forever('logs')
