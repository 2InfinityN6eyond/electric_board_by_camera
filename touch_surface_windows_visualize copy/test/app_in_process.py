import time, sys
from multiprocessing import Process, Queue, Pipe, Value
from PyQt5 import QtWidgets, QtCore, QtGui



class MainWindow(QtWidgets.QMainWindow) :
    def __init__(self) :
        super(MainWindow, self).__init__()

        self.show()
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

class MainWindowProcess(Process) :
    def __init__(self) :
        super(MainWindowProcess, self).__init__()

    def run(self) :
        app = QtWidgets.QApplication(sys.argv)
        main_window = MainWindow()
        sys.exit(app.exec_())

if __name__ == "__main__" :
    prcs = MainWindowProcess()
    prcs.start()

    prcs.join()