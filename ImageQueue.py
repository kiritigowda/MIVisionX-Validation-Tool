import Queue

class ImageQueue:
    def __init__(self):
        print 'init'
        self.queue = Queue.Queue()

    def __del__(self):
        print 'del'

    def enqueue(self, image):
        self.queue.put(image)

    def dequeue(self):
        return self.queue.get()

    def isEmpty(self):
        return self.queue.empty()

    def getSize(self):
        return self.queue.qsize()

if __name__ == '__main__':
    queue = ImageQueue()
    queue.enqueue(1)
    print queue.getSize()
    print queue.dequeue()
    print queue.isEmpty()