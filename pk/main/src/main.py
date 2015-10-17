from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from  pk.main.ui.main_gui import Ui_main_tab
import sys
from pk.utils.loading import *

class MainWindow(QtGui.QTabWidget, Ui_main_tab):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
# GUI functions goes here
class FileOpener(object):
    def __init__(self, filename):
        self.filename = filename
    @staticmethod
    def load_file(filename):
        extension = filename[filename.rfind('.'):]
        if (extension == '.csv'):
            return load_csv(filename)
        elif (extension == '.arff'):
            return load_arff(filename)
        elif (extension == '.xls' or extension == '.xlsx'):
            return load_excel(filename)



def openfile():
    filename = QFileDialog.getOpenFileName('Open File','/')
    filename = str(filename)
    X, y = FileOpener.load_file(filename)
    return X,y



# Scikit functions goes here

# main function to run the program
def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    # ex.openfile_btn.clicked.connect(openfile)
    ex.show()
    sys.exit(app.exec_())

# runs the main function
if __name__ == '__main__':
    main()