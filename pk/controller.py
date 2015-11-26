"""This file contains classes and functions for controller objects.
    Author: Sean Dai
"""
from PyQt4 import QtGui
from PyQt4.QtGui import QFileDialog
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import QDialogButtonBox

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

    def get_preprocess_options(self, app):
        """
        Opens a new window for selecting preprocessing options for the dataset.
        """
        class PreprocessFrame(QtGui.QWidget):
            def __init__(self):
                QtGui.QWidget.__init__(self)
                layout = QVBoxLayout(self)
                self.setWindowTitle("Preprocessing")
                self.cbox1 = QCheckBox("Normalize")
                self.cbox2 = QCheckBox("Standardize")
                self.cbox3 = QCheckBox("Remove examples containing:")
                text_area = QLineEdit(self.cbox3)
                text_area.setText("?")
                dbx = QDialogButtonBox(self)
                dbx.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                # self.connect(dbx, SIGNAL("accepted()"), dbx, SLOT("accept()"))
                # self.connect(dbx, SIGNAL("rejected()"), dbx, SLOT("reject()"))
                layout.addWidget(self.cbox1)
                layout.addWidget(self.cbox2)
                layout.addWidget(self.cbox3)
                layout.addWidget(text_area)
                layout.addWidget(dbx, alignment=Qt.AlignCenter)
                self.setLayout(layout)

        pf = PreprocessFrame()
        pf.show()
        app.exec_()
