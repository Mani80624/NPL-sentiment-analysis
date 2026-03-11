import unittest
from stopwords.stopwords import limpiar_texto

class TestMathUtils(unittest.TestCase):
   def text_1(self):
       self.assertEqual(limpiar_texto("I went to the university today and met some friends. We talked about our classes and the projects we need to finish this week. It was a normal day and nothing unusual happened."), 
                        ['i', 'went', 'university', 'today', 'met', 'friends', 
                         'we', 'talked', 'classes', 'projects', 'need', 
                         'finish', 'week', 'normal', 'day', 'nothing', 
                         'unusual', 'happened'])


if __name__ == "__main__":
   unittest.main(verbosity=2)