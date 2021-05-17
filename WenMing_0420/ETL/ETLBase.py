#-*- coding: utf-8 -*-
import os
import gc
import feather
import pandas as pd
import numpy as np


class ETLBase:
    def __init__(self, process_name, pre_request_etls=[], save=False):
        self.process_name = process_name
        self.pre_request_etls = pre_request_etls
        self.save = save

    def run(self):
        '''
        Check if the pre-request results are completed.
        If completed, load the result. Otherwise, run the pre-request ETL process.
        '''
        inputs = []
        for pre_etl in self.pre_request_etls:
            if pre_etl.is_complete():
                print(f'[LOAD] result of "{pre_etl.process_name}"')
                inputs.extend(pre_etl.load_result())

            else:
                print(f'[RUN] process of "{pre_etl.process_name}"')
                inputs.extend(pre_etl.run())

        results = self.process(inputs)
        print(f'[COMPLETE] {self.process_name}')
        del inputs
        gc.collect()
        if self.save:
            print(f'[SAVE] result of "{self.process_name}"')
            self.save_result(results)
        return results

    def is_complete(self):
        '''
        This function check if the temporary result file is already saved, to notify
        whether the process should be triggered before the next ETL is processed.
        '''
        return False

    def load_result(self):
        '''
        This function load the temporary result file saved by "save_result" function.
        Should be override if save_result is override.
        '''
        pass

    def process(self, inputs):
        '''
        This is the main process and should be override.
        input:
         - inputs: a list containing the inputs
        output:
         - outputs: a list containing the outputs
        '''
        outputs = inputs
        return outputs

    def save_result(self, results):
        '''
        Save result for the next ETL Process.
        Sould be considered overrided if re-use of processed data is considered
        '''
        pass


class ETLwithDifferentResults(ETLBase):
    def __init__(self, process_name, pre_request_etls, result_dir, save=True):
        super(ETLwithDifferentResults, self).__init__(
            process_name,
            pre_request_etls=pre_request_etls,
            save=save
        )
        if type(result_dir) == list:
            self.result_dirs = result_dir
        else:
            self.result_dirs = [result_dir]

    def is_complete(self):
        for file_dir in self.result_dirs:
            if not os.path.exists(file_dir):
                return False
        return True

    def load_result(self):
        results = []
        for file_dir in self.result_dirs:
            if '.h5' in file_dir:
                results.append(pd.read_hdf(file_dir, key=file_dir.split('.h5')[0].split('/')[-1], mode='r'))
            elif '.feather' in file_dir:
                results.append(feather.read_dataframe(file_dir))
            elif '.npy' in file_dir:
                # np.save(file_dir.split('.npy')[0], feature_mapper)
                np_result = np.load(file_dir, allow_pickle=True)
                try:
                    np_result = np_result.item()
                except:
                    pass
                results.append(np_result)
            else:
                pass
        return results

    def save_result(self, results):
        for i, file_dir in enumerate(self.result_dirs):
            # feather.write_dataframe(results[0], self.result_dir)
            if '.h5' in file_dir:
                results[i].to_hdf(file_dir, key=file_dir.split('.')[0].split('/')[-1], mode='w')
            elif '.feather' in file_dir:
                feather.write_dataframe(results[i], file_dir)
            elif '.npy' in file_dir:
                np.save(file_dir.split('.npy')[0], results[i])
            print(f' as {file_dir}')


class ETLPro:
    def __new__(cls, process_name, pre_request_etls, result_dir=None, **kwargs):
        if result_dir:
            super_class = ETLwithDifferentResults
        else:
            super_class = ETLBase
        cls = type(cls.__name__ + '+' + super_class.__name__, (cls, super_class), {})
        return super(ETLPro, cls).__new__(cls)

    def __init__(self, process_name, pre_request_etls, result_dir=None, **kwargs):
        if result_dir:
            super(ETLPro, self).__init__(process_name, pre_request_etls, result_dir, save=True)
        else:
            super(ETLPro, self).__init__(process_name, pre_request_etls=pre_request_etls, save=False)


class SelectResult(ETLPro):
    '''
    This ETL process selects particular results from previous ETLs.
    By default, it extract the first result.
    '''

    def __init__(self, process_name, pre_request_etls, selected_indices=[0], result_dir=None):
        super(SelectResult, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.selected_indices = selected_indices

    def process(self, inputs):
        assert len(self.selected_indices) < len(inputs)
        return [inputs[i] for i in self.selected_indices]
# for ETL with multiple output


'''class ETLwithFeatherResult(ETLBase):
    def __init__(self, process_name, pre_request_etls, result_dir, save=True):
        super(ETLwithFeatherResult, self).__init__(
            process_name,
            pre_request_etls=pre_request_etls,
            save=save
        )
        self.result_dir = result_dir  # a list of diractories

    def is_complete(self):
        return os.path.exists(self.result_dir)

    def load_result(self):
        return [feather.read_dataframe(self.result_dir)]

    def save_result(self, results):
        feather.write_dataframe(results[0], self.result_dir)
        print(f' as {self.result_dir}')
'''

'''class ETLwithH5Result(ETLBase):
    def __init__(self, process_name, pre_request_etls, result_dir, save=True):
        super(ETLwithH5Result, self).__init__(
            process_name,
            pre_request_etls=pre_request_etls,
            save=save
        )
        self.result_dir = result_dir

    def is_complete(self):
        return os.path.exists(self.result_dir)

    def load_result(self):
        return [pd.read_hdf(self.result_dir, key=self.result_dir.split('.'), mode='r')]

    def save_result(self, results):
        # feather.write_dataframe(results[0], self.result_dir)
        results[0].to_hdf(self.result_dir, key=self.result_dir.split('.')[0], mode='w')
        print(f' as {self.result_dir}')'''

'''
class ETLPro(ETLSelectiveInherent):
    def __new__(cls, process_name, pre_request_etls, result_dir=None, **kwargs):
        super_class = ETLSelectiveInherent
        cls = type(cls.__name__ + '+' + super_class.__name__, (cls, super_class), {})
        return super(ETLPro, cls).__new__(cls, process_name, pre_request_etls, result_dir=result_dir)

    def __init__(self, process_name, pre_request_etls, result_dir=None, **kwargs):
        super(ETLPro, self).__init__(process_name, pre_request_etls, result_dir=result_dir)'''
