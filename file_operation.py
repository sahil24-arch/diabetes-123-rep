
import pickle

class FileOperations:

    """
    This class is used to perform saving and loading operation of models
    """

    try:
        def save_model(self,model,file_name):
            pickle.dump(model,open(file_name,'wb'))

        def load_model(self,filename):
            mod=pickle.load(open(filename,'rb'))
            return mod

    except Exception as ex:
        self.log_writer.log(self.file_object, 'Error occured in file operation')
        raise ex

