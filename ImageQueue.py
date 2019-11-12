import Queue
from PyQt4 import QtCore
class ImageQueue:
    def __init__(self):
        self.mutex = QtCore.QMutex()
        self.queue = Queue.Queue()

    def enqueue(self, image):
        self.mutex.lock()
        self.queue.put(image)
        self.mutex.unlock()

    def dequeue(self):
        return self.queue.get()

    def isEmpty(self):
        return self.queue.empty()

    def getSize(self):
        return self.queue.qsize()