"""This file contains classes and functions for controller objects.
    Author: Sean Dai
"""
from PyQt4 import QtGui
from PyQt4.QtGui import QFileDialog

class ViewGenerator(object):
    def open_file_dialog(self, app, filter):
        """
        Opens a file dialog for the user to select the desired file.
        """
        frame = QtGui.QWidget()
        path = QFileDialog.getOpenFileName(parent=frame, caption="Open File",
                                           filter=filter)
        frame.destroy()
        app.closeAllWindows()
        return str(path)

