#-*- coding: utf-8 -*-
import os
import sys 
import gc
import feather
import pandas as pd
from tabulate import tabulate
import numpy as np
from pyflow import GraphBuilder
import pprint 
from common.utils import Str2CodeAdaptor

_pp = pprint.PrettyPrinter(width=10, compact=True, indent=1)
class PipeConfigBuilder:
    def __init__(self):
        self.pyflow_GB = GraphBuilder()
        self.graph_dict = dict()
        
    def add(self, var_name, value, rank=None, color='gray', shape='cylinder', fontsize=None):
        def current_process():
            return value 
        current_process_name = f'{var_name}={value}'
        # method_alias = current_process_name 
        n_out = 1
        
        self.pyflow_GB.add(current_process, 
                           method_alias = var_name, 
                           output_alias = _pp.pformat(value),
                           n_out = n_out,
                           rank = rank,
                           color = color,
                           shape = shape,
                           fontsize = fontsize
                          )
        # no call function here, directly call the method in add. 
        pf_output = self.pyflow_GB()
        config_module = DataNode(current_process_name, [])
        config_module.set_process(current_process)
        config_module.set_n_out(n_out)
        config_module.set_pf_output_node(pf_output)
        self.graph_dict[pf_output.get_node_uid()] = config_module
        return config_module
    
    def view(self, *args, **kargs):
        return self.pyflow_GB.view(
            *args, 
            **kargs
        )
    def view_dependency(self, *args, **kargs):
        return self.pyflow_GB.view_dependency(*args, **kargs)
    
    def setups(self, env = None, **kargs):
        for var_name, value in kargs.items():
            config_module = self.add(var_name, value)
            if env: # not None, can be globals() or an object. 
                Str2CodeAdaptor.add_var(env, var_name, config_module)
            else:
                Str2CodeAdaptor.add_var(self, var_name, config_module)
    
class PipelineBuilder():
    # - [ ] load the function only if the function does not exist 
    # - [ ] load the function in the beginning if source_func is provided 
    def __init__(self, pipe=None, func_source=None):
        '''
        Input: 
            - pipe: the pre-request pipe or a config object from PipeConfigBuilder. 
            - func_source: the name of the module/package holding the functions used in this pipeline or 
                the globals() dictionary if the functions is defined in main function. 
        '''
        self.current_process = None 
        self.current_process_name = None
        self.pipe = pipe
        self.func_source = func_source
        if pipe:
            self.pyflow_GB = pipe.pyflow_GB
        else:
            self.pyflow_GB = GraphBuilder()
            
        self.graph_dict = dict()
        self.data_info_holder = _DataInfoHolder()
        self._current_mode = 'arg_only'

        
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
        # print("In __call__, args, kargs:", args, kwargs)
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
                self.graph_dict[pf_output_node.get_node_uid()] = out
                outs.append(out)
                
            return outs
            
        else:
            process_module.set_pf_output_node(pf_output)
            self.graph_dict[pf_output.get_node_uid()] = process_module
            return process_module 
    def _to_pf_out_node(self, arg):
        if type(arg).__name__ == 'DataNode+ETLBase':
            return arg.pf_output_node
        if type(arg).__name__ == 'SelectResult+ETLBase':
            return arg.pf_output_node
        return arg
    def view(self, *args, option = 'arg_only', **kargs):
        if kargs['summary'] == False:
            self._assign_datanode_info(option = option)
        return self.pyflow_GB.view(
            *args, 
            **kargs
        )
    def view_dependency(self, *args, option = 'arg_only', **kargs):
        if kargs['summary'] == False:
            self._assign_datanode_info(option = option)
        return self.pyflow_GB.view_dependency(*args, **kargs)
    
    def _assign_datanode_info(self, option = 'arg_only'):
        assert option == 'all' or option == 'light' or option == 'arg_only'
        if option == 'arg_only':
            for node_id, data_node in self.graph_dict.items():
                graph_node = self.pyflow_GB.graph_dict[node_id]
                graph_node['alias'] = "_".join(node_id.split("_")[:-1])
            return 
        if self._current_mode == option:
            return 
        else:
            if option == 'all' or option == 'light':
                for node_id, data_node in self.graph_dict.items():
                    graph_node = self.pyflow_GB.graph_dict[node_id]
                    graph_node['alias'] = self._obtain_datanode_info(node_id, data_node, option = option)
                    graph_node['attributes']['fontsize']=1
    def _obtain_datanode_info(self, node_id, data_node, option = 'light'):
        assert option == 'all' or option == 'light'
        if option == 'all':
            data_name = "_".join(node_id.split("_")[:-1])
            _size_ = self.data_info_holder.get_size(node_id, data_node)
            _type_ = self.data_info_holder.get_type(node_id, data_node)
            _data_content_ = self.data_info_holder.get_data_content(node_id, data_node)
            result_string = f'{data_name}\ntype:{_type_}\nsize: {_size_} bytes\n{_data_content_}'
        else:
            data_name = "_".join(node_id.split("_")[:-1])
            _size_ = self.data_info_holder.get_size(node_id, data_node)
            _type_ = self.data_info_holder.get_type(node_id, data_node)
            result_string = f'{data_name}\ntype:{_type_}\nsize: {_size_} bytes'
        self.data_info_holder.clean_up_result()
        return result_string
    
    def is_leaf(self, arg_name):
        info = self.pyflow_GB.graph_dict[arg_name]
        if 'type' not in info or 'children' not in info:
            return False 
        else:
            return info['type'] == 'data' and len(info['children']) == 0
        
    def get_all_ancestor_pf_data_nodes(self, graph_dict, node):
        data_nodes = dict()
        def recursive(node):
            try:
                node_id = node.pf_output_node.get_node_uid()
                data_nodes[node_id] = (graph_dict[node_id], node)
            except:
                pass 
            finally:
                for parent_node in node.pre_request_etls:
                    recursive(parent_node)
        recursive(node)
        return data_nodes
    
    def _func_(self, _func):
        # decorator for adding a function as a member of the class.  
        exec(f'self._{_func.__name__} = _func')
        return _func
    
    def _rep_func_(self, func_name):
        '''
        Inputs: 
            - func_name: the name of the function that should be replace 
            - new_func: the new function (type: method) 
        '''
        matched_operations = self._select_operation_by_name(func_name)
        output_nodes_list = self._get_output_nodes_list(matched_operations)
        def decorator(new_func):
            self._assign_new_function(output_nodes_list, new_func)
        return decorator
        
    def _select_operation_by_name(self, func_name):
        '''
        Input: 
            - func_name: the name of the function 
        Output:
            - matched_operations: a list of operation nodes in pyflow graph_dict 
        '''
        def is_selected_operation(node_id, func_name):
            node = self.pyflow_GB.graph_dict[node_id]
            if 'type' in node and node['type'] == 'operation' and node['alias'] == func_name:
                return True
            else:
                return False
        matched_operations = list(filter(
            lambda node_id: is_selected_operation(node_id, func_name), 
            self.pyflow_GB.graph_dict.keys()
        ))
        return matched_operations
    def _get_output_nodes_list(self, operations):
        '''
        Inputs:
            - matched_operations: a list of operation nodes in pyflow graph_dict. 
        Outputs:
            - output_nodes_list: a list of children list where each element is a output data node 
                of an operation. 
        '''
        output_nodes_list = [self.pyflow_GB.graph_dict[operation]['children'] for operation in operations]
        return output_nodes_list

    def _assign_new_function(self, output_nodes_list, new_func):
        '''
        Given a list of output nodes where the new function should be assigned as their process. 
        Note: the new function is injected from one of the output nodes of the operation. 
        '''
        for output_nodes in output_nodes_list:
            if len(output_nodes) > 1:
                assert type(self.graph_dict[output_nodes[0]]).__name__ == 'SelectResult+ETLBase'
                self.graph_dict[output_nodes[0]].pre_request_etls[0].set_process(new_func)
            else:
                assert type(self.graph_dict[output_nodes[0]]).__name__.split('+')[0] == 'DataNode'
                self.graph_dict[output_nodes[0]].set_process(new_func)
    
    def setup_connection(self, code_str, env = None, result_dir = None, func = None):
        '''
        This function allows self.add to be simplify with simple python function call command. 
        Input:
            - obj: determine where to put the output variables of the function call. 
                - if `obj` == globals(): put into global environment. 
                - if `obj` == None: put into self 
                - otherwise: put into the object represented by `obj`
        '''
        if env: # not None, can be globals() or an object. 
            pass 
        else:
            env = self
            
        out_vars, func_str, args_str_list, kwargs_str_dict = Str2CodeAdaptor.breakdown_function_call(code_str)
        
        private_func_str = f"self._{func_str.replace('.','_')}"
        
        if func:
            exec(f"{private_func_str} = func")
        else: # If no custom function 
            # insert function from global 
            try:
                eval(private_func_str)
            except:
                if type(self.func_source) == dict:
                    tmp = self.func_source[func_str]
                    exec(f"{private_func_str} = tmp")
                if type(self.func_source) == str:
                    if '.' in func_str:
                        fn = func_str.split(".")[0]
                        exec(f'from {self.func_source} import {fn}')
                        exec(f"{private_func_str} = {func_str}")
                    else:
                        exec(f'from {self.func_source} import {func_str}')
                        exec(f"{private_func_str} = {func_str}")
        
        func_added_pipe = self.add(
            eval(private_func_str), 
            method_alias = func_str,
            n_out = len(out_vars), 
            output_alias = out_vars, 
            result_dir = result_dir
        )
        # print('func_added_pipe:', func_added_pipe)
        # TODO: evaluate values in input_dict : 
        #  [V] if env == globals(), get global_variable from global. 
        #  [V] if env == None, use self as the prefix of object values. 
        #  [V] if env == an object, use this object as the prefix (meaning that the pre-request objects are from this object.) 
        
        args = [Str2CodeAdaptor.get_var(env, var) for var in args_str_list]
        
        def force_eval(value_str):
            try:
                try:
                    return eval(value_str)
                except:
                    return Str2CodeAdaptor.get_var(env, value_str)
            except: 
                if type(env) != dict:
                    # if cannot find in this pipe, how about try to find it from the requested pipe connect to this pipe (i.e., env.pipe)
                    return Str2CodeAdaptor.get_var(env.pipe, value_str)
        kargs = dict([(var, force_eval(value_str)) for var, value_str in kwargs_str_dict.items()])
        # print("In setup connection, args, kargs:",args, kargs)
        out_nodes = func_added_pipe(*args, **kargs)# eval(f'func_added_pipe{input_str}')

        if len(out_vars) > 1:
            for out_name, node in zip(out_vars, out_nodes):
                Str2CodeAdaptor.add_var(env, out_name, node)
        else:
            Str2CodeAdaptor.add_var(env, out_vars[0], out_nodes)

class _DataInfoHolder:
    def __init__(self):
        self.print_content_dict = dict()
        self.size_dict = dict()
        self.type_dict = dict()
        self._result = None 
    def get_size(self, node_id, data_node):
        if node_id in self.size_dict:
            return self.size_dict[node_id]
        else:
            if self._result is not None:
                self.size_dict[node_id] = sys.getsizeof(self._result)
            else:
                self._result = data_node.get()
                self.size_dict[node_id] = sys.getsizeof(self._result)
            return self.size_dict[node_id]
    def get_type(self, node_id, data_node):
        if node_id in self.type_dict:
            return self.type_dict[node_id]
        else:
            if self._result is not None:
                self.type_dict[node_id] = type(self._result)
            else:
                self._result = data_node.get()
                self.type_dict[node_id] = type(self._result)
            return self.type_dict[node_id]
    def get_data_content(self, node_id, data_node):
        if node_id in self.print_content_dict:
            return self.print_content_dict[node_id]
        else:
            if self._result is not None:
                self.print_content_dict[node_id] = self._get_print_result(self._result)
            else:
                self._result = data_node.get()
                self.print_content_dict[node_id] = self._get_print_result(self._result)
            return self.print_content_dict[node_id]
    def _get_print_result(self, result):
        if type(result) == pd.core.frame.DataFrame:
            print_result = tabulate(result.head(1).T, headers='keys', tablefmt='psql')
        else:
            print_result = _pp.pformat(result)
        if print_result.count('\n')>20:
            print_result = "\n".join(print_result.split('\n')[:8] + ['...'] + print_result.split('\n')[-4:])
        return print_result
    def clean_up_result(self):
        self._result = None 
        gc.collect()
        
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
        
# TODO:
# - [V] change temp object name to obj rather than the function name
# - [V] make the ETLBase allow save when a result directory is assigned
# - [V] allow saving of multiple files (ETL might have multiple results)
# - [V] allow single input and single output (not list)
# - [V] allow subset selection from previous ETL process
# - [X] allow the input to be a dictionary and the output to be a dictionary, too
# - [V] make the ETL Process object allows two step construction like nn.module. 1. first initialized with configuration . 2. Be called to assign inputs and obtain outputs later
# - [ ] incorporate google drive download as the first step of ETL
# - [V] allows zero input ETL if the ETL does not have previous ETL
