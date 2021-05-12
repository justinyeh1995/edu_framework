#-*- coding: utf-8 -*-
import gc
import os
import feather


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
                print(f'[LOAD] {pre_etl.process_name} result')
                inputs.extend(pre_etl.load_result())

            else:
                print(f'[RUN] {pre_etl.process_name} process')
                inputs.extend(pre_etl.run())

        results = self.process(inputs)
        print(f'[COMPLETE] {self.process_name}')
        del inputs
        gc.collect()
        if self.save:
            print(f'[SAVE] {self.process_name} result')
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


class ETLwithFeatherResult(ETLBase):
    def __init__(self, process_name, pre_request_etls, result_dir, save=True):
        super(ETLwithFeatherResult, self).__init__(
            process_name,
            pre_request_etls=pre_request_etls,
            save=save
        )
        self.result_dir = result_dir

    def is_complete(self):
        return os.path.exists(self.result_dir)

    def load_result(self):
        return [feather.read_dataframe(self.result_dir)]

    def save_result(self, results):
        feather.write_dataframe(results[0], self.result_dir)
        print(f' as {self.result_dir}')


class ETLwithH5Result(ETLBase):
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
        print(f' as {self.result_dir}')


class ETLPro:
    def __new__(cls, process_name, pre_request_etls, result_dir=None, **kwargs):
        if result_dir:
            if '.h5' in result_dir:
                super_class = ETLwithH5Result
            elif '.feather' in result_dir:
                super_class = ETLwithFeatherResult
        else:
            super_class = ETLBase
        cls = type(cls.__name__ + '+' + super_class.__name__, (cls, super_class), {})
        return super(ETLPro, cls).__new__(cls)

    def __init__(self, process_name, pre_request_etls, result_dir=None, **kwargs):
        if result_dir:
            super(ETLPro, self).__init__(process_name, pre_request_etls, result_dir, save=True)
        else:
            super(ETLPro, self).__init__(process_name, pre_request_etls=pre_request_etls, save=False)


'''
class ETLPro(ETLSelectiveInherent):
    def __new__(cls, process_name, pre_request_etls, result_dir=None, **kwargs):
        super_class = ETLSelectiveInherent
        cls = type(cls.__name__ + '+' + super_class.__name__, (cls, super_class), {})
        return super(ETLPro, cls).__new__(cls, process_name, pre_request_etls, result_dir=result_dir)

    def __init__(self, process_name, pre_request_etls, result_dir=None, **kwargs):
        super(ETLPro, self).__init__(process_name, pre_request_etls, result_dir=result_dir)'''
