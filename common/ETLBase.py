#-*- coding: utf-8 -*-
import os
import gc
import feather
import pandas as pd
import numpy as np
from pyflow import GraphBuilder

class PipelineBuilder():
    def __init__(self):
        self.current_process = None 
        self.current_process_name = None
        self.pyflow_GB = GraphBuilder()
        
    def add(self, process, method_alias=None, output_alias=None, result_dir=None, 
            n_out=1, rank=None, color='lightblue', shape=None, fontsize=None):
        self.current_process = process
        self.current_process_name = process.__name__
        self.n_out = n_out
        
        if type(result_dir) == list: 
            assert len(result_dir) == n_out
            for out, path in zip(output_alias, result_dir):
                assert out == path.split("/")[-1].split(".")[0]
        if result_dir:
            color = 'pink'
        self.result_dir = result_dir
        
        self.pyflow_GB.add(process, 
                           method_alias = method_alias, 
                           output_alias = output_alias,
                           n_out = n_out,
                           rank = rank,
                           color = color,
                           shape = shape,
                           fontsize = fontsize
                          )
        return self
    def __call__(self, *args, **kwargs):
        assert self.current_process
        assert self.current_process_name
        
        
        pf_input = [arg.pf_output_node for arg in args]
        
        pf_kwargs = dict([(key, self._to_pf_out_node(arg)) for key, arg in kwargs.items()])

        pf_output = self.pyflow_GB(*pf_input, **pf_kwargs)
        
        process_module = DataNode(
            self.current_process_name,
            list(args),
            result_dir = self.result_dir,
            **kwargs 
        )
        
        process_module.set_process(self.current_process)
        process_module.set_n_out(self.n_out)
        
        if self.n_out > 1:
            
            outs = []
            for i, pf_output_node in zip(range(self.n_out), pf_output):
                
                out = SelectResult(
                self.current_process_name + f'[{i}]',
                [process_module],
                selected_indices=[i])
                out.set_pf_output_node(pf_output_node)
                
                outs.append(out)
                
            return outs
            
        else:
            process_module.set_pf_output_node(pf_output)
            return process_module 
    def _to_pf_out_node(self, arg):
        if type(arg).__name__ == 'DataNode+ETLBase':
            return arg.pf_output_node
        if type(arg).__name__ == 'SelectResult+ETLBase':
            return arg.pf_output_node
        return arg
    def view(self, *args, **kargs):
        return self.pyflow_GB.view(
            *args, 
            **kargs
        )
    def view_dependency(self, *args, **kargs):
        return self.pyflow_GB.view_dependency(*args, **kargs)


class ETLBase:
    def __init__(self, process_name, pre_request_etls=[], save=False):
        self.process_name = process_name
        self.pre_request_etls = pre_request_etls
        self.save = save

    def run(self, verbose=False):
        '''
        Check if the pre-request results are completed.
        If completed, load the result. Otherwise, run the pre-request ETL process.
        '''
        inputs = []
        for pre_etl in self.pre_request_etls:
            if pre_etl.is_complete():
                if verbose:
                    print(f'[LOAD] result of "{pre_etl.process_name}"')
                inputs.extend(pre_etl.load_result())
            else:
                if verbose:
                    print(f'[RUN] process of "{pre_etl.process_name}"')
                inputs.extend(pre_etl.run())
        results = self.process(inputs)
        if verbose:
            print(f'[COMPLETE] {self.process_name}')
        del inputs
        gc.collect()
                            
        if self.save:
            if verbose:
                print(f'[SAVE] result of "{self.process_name}"')
            self.save_result(results)
        return results

    def is_complete(self):
        '''
        This function check if the temporary result file is already saved, to notify
        whether the process should be triggered before the next ETL is processed.
        '''
        return False

    def load_result(self, verbose=False):
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

    def save_result(self, results, verbose=False):
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
        self._create_data_folder(self.result_dirs)
    def _create_data_folder(self, result_dirs, verbose=True):
        # create folders on initialization
        for result_path in result_dirs:
            if ".." in result_path:
                break # Note: do not create folder outside current directory
            folder_path = ""
            for folder_name in result_path.split('/')[:-1]:
                folder_path += f"{folder_name}/"
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                    if verbose: 
                        print(f"You have created directory: {folder_path}")

    def is_complete(self):
        for file_dir in self.result_dirs:
            if not os.path.exists(file_dir):
                return False
        return True

    def load_result(self, verbose=False):
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
            if verbose:
                print(f' from {file_dir}')
        return results

    def save_result(self, results, verbose=False):
        for i, file_dir in enumerate(self.result_dirs):
            # feather.write_dataframe(results[0], self.result_dir)
            if '.h5' in file_dir:
                results[i].to_hdf(file_dir, key=file_dir.split('.')[0].split('/')[-1], mode='w')
            elif '.feather' in file_dir:
                feather.write_dataframe(results[i], file_dir)
            elif '.npy' in file_dir:
                np.save(file_dir.split('.npy')[0], results[i])
            if verbose:
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
    def set_pf_output_node(self, pf_output_node):
        self.pf_output_node = pf_output_node


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
    def get(self, verbose=False):
        assert len(self.selected_indices) == 1
        return self.run(verbose=verbose)[0]
    
    
class DataNode(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, **kwargs):
        super(DataNode, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.kwargs = dict([(key, self._get_real_var(arg)) for key, arg in kwargs.items()])
        self.n_out = 1
        
    def _get_real_var(self, arg):
        '''    
        Some of the args in kwargs is not real data but ETLPro Objects.
        To solve the problem, convert them all to data using .get.  
        '''
        if type(arg).__name__ == 'DataNode+ETLBase':
            return arg.get()
        if type(arg).__name__ == 'SelectResult+ETLBase':
            return arg.get()
        return arg
    
    def set_process(self, current_process):
        self.current_process = current_process 
        
    def set_n_out(self, n_out):
        self.n_out = n_out 
        
    def process(self, inputs):
        
        result = self.current_process(*inputs, **self.kwargs)
        if self.n_out == 1:
            return [result]
        else:
            return result
    
    def get(self, verbose=False):
        assert self.n_out == 1
        return self.run(verbose=verbose)[0]
        
