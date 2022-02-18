        # -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:51:29 2020

@author: aylin
"""
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from uygulamakod import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()