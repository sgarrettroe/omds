from scidatalib.scidata import SciData
from PyQt5.QtWidgets import *
from pprint import pprint
import sys #argh why!

class MySciData(SciData):
    def __init__(self, uid):
        super().__init__(uid)

# creating a class
# that inherits the QDialog class
class Window(QDialog):

    # constructor
    def __init__(self,input):
        super(Window, self).__init__()

        # list to hold all the input elements
        self.elements = []

        # setting window title
        self.setWindowTitle("Python")

        # setting geometry to the window
        self.setGeometry(100, 100, 500, 400)

        # creating a group box
        self.formGroupBox = QGroupBox("Form 1")

        # calling the method that creates the form
        self.createForm(input)

        # creating a dialog button for ok and cancel
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # adding action when form is accepted
        self.buttonBox.accepted.connect(lambda: self.getInfo(input))

        # adding action when form is rejected
        self.buttonBox.rejected.connect(self.reject)

        # creating a vertical layout
        mainLayout = QVBoxLayout()

        # adding form group box to the layout
        mainLayout.addWidget(self.formGroupBox)

        # adding button box to the layout
        mainLayout.addWidget(self.buttonBox)

        # setting lay out
        self.setLayout(mainLayout)

    # get info method called when form is accepted
    def getInfo(self,input):
        count = 0
        for key in input.__dict__.keys():
            input.__dict__[key] = f'{self.elements[count].text()}'
            count += 1

        # closing the window
        self.close()


    # create form method
    def createForm(self,input):
        # creating a form layout
        layout = QFormLayout()

        # adding rows
        # for name and adding input text
        for key in input.__dict__.keys():
            this_element = QLineEdit()
            this_element.setText(input.__dict__[key])
            self.elements.append(this_element)
            layout.addRow(QLabel(key), this_element)

        # setting layout
        self.formGroupBox.setLayout(layout)

class SciDataInput():
    def __init__(self):
        pass

    def input_form(self,input):
        app = QApplication(sys.argv)

        # create the instance of our Window
        window = Window(input)

        # showing the window
        window.show()
        app.exec()

class Author(SciDataInput):
    '''Class to hold author information. Can be called with keyword argument pairs to set the
    properties.

    Examples:
         a = Author(name='Robin',affiliation='Penn',orcid='')
         d = {'author':'Robin','affiliation':'Penn',orcid=''}
         a = Author(**d)

    '''
    def __init__(self, name='', affiliation='', orcid='',input_form=False):
        name: str
        affiliation: str
        orcid: str
        self.name = name
        self.affiliation = affiliation
        self.orcid = orcid

        if input_form:
            super(Author,self).input_form(self)

        self.validate()

    def validate(self):
        'Validate author information'
        return self.validate_name() & self.validate_affiliation() & self.validate_orcid()


    def validate_name(self):
        'Make sure name is not empty'
        return not self.name.strip()

    def validate_affiliation(self):
        'Make sure affiliation is not empty'
        return not self.affiliation.strip()

    def validate_orcid(self)->bool:
        'Test if last digit agrees with check digit per ISO 7064 11, 2.'
        def generate_check_digit(base_digits):
            '''Generates orcid check digit as per ISO 7064 11, 2.
            '''
            total = 0
            for i in range(len(base_digits)-1):
                digit = int(base_digits[i])
                total = (total + digit)*2

            remainder = total % 11
            result = (12 - remainder) % 11
            if result == 10:
                out = 'X'
            else:
                out = str(result)
            return out
        orcid = self.orcid.lstrip(r'http://orcid.org/').replace('-','')
        val = generate_check_digit(orcid)
        match = val == orcid[-1]
        if not match:
            raise ValueError(f'ORCiD {self.orcid} checkdigit {self.orcid[-1]} does not match expected value {val}')
        # normalize
        self.orcid = f'http://orcid.org/{orcid[0:4]}-{orcid[4:8]}-{orcid[8:12]}-{orcid[12:16]}'
        return match


class Authors:
    def __init__(self, *args):
        author_list: list
        self.author_list = []
        for arg in args:
            self.append(arg)
    def __iter__(self):
        for elem in self.author_list:
            yield elem

    def append(self,item):
        # make sure it is an author
        if isinstance(item, Author):
            # reject duplicates
            if item not in self.author_list:
                self.author_list.append(item)

    def __getitem__(self,index):
        return self.author_list[index]

#uh = MySciData(42)
#pprint(uh.meta)

d = {'name':'sean','affiliation':'pitt'}
uhh = Author(name='sean',affiliation='pitt',orcid='http://orcid.org/0000-0003-1415-9269')
uhh2 = Author(name='not sean',affiliation='not pitt',orcid='http://orcid.org/0000-0003-1415-9269')

print(uhh.name)
pprint(vars(uhh))

a = Authors()
a.append(uhh2)
a.append(uhh)
pprint(vars(a))
pprint(vars(a[0]))
pprint(vars(a[-1]))


# so I need to get a list of dictionaries
uhhh = [aa.__dict__ for aa in a]
pprint(uhhh)

b = Authors(uhh,uhh2,uhh,uhh2)
pprint(len(b.author_list))

# ok from form
a = Author(name='sean',affiliation='pitt',orcid='http://orcid.org/0000-0003-1415-9269',input_form=True)