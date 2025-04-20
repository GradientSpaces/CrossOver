"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=logging-fstring-interpolation
import logging
import re
import multiprocessing as mp
from abc import abstractmethod
from typing import List, Dict, Tuple
import multiprocessing
import yaml
import json
import copy
import inspect
import logging
import numpy as np
import os

class DictRecursive(object):
    def __init__(self):
        pass

    def load(self, kargs: dict, shared_dict=None):
        """
        Launch args of class from a dict. All subclass of DictRecursive will call this function automatically. Supported
            types includes int, float, list, str and DictRecursive

        Args:
            kargs: a dict saved the pairs of name/value of attributions
            shared_dict: a shared item used by all other items
        """
        if shared_dict is None:
            shared_dict = {}
        for cls_arg_name in self.__dict__.keys():
            arg_value = None
            if kargs is not None:
                arg_value = kargs[cls_arg_name] if cls_arg_name in kargs.keys() else None
            if shared_dict is not None:
                arg_value = shared_dict[cls_arg_name] if cls_arg_name in shared_dict.keys() else arg_value
            cls_arg = self.__dict__[cls_arg_name]
            self.__dict__[cls_arg_name] = self.parse_single_arg(cls_arg, arg_value, shared_dict)
        return self

    def save(self):
        save_dict = {}
        for cls_arg_name in self.__dict__.keys():
            save_dict[cls_arg_name] = self.inverse_single_arg(self.__dict__[cls_arg_name])
        return save_dict

    def load_from_yaml(self, yaml_path, shared_scope=''):
        with open(yaml_path, 'r', encoding='utf-8') as fp:
            cfg_cxt = yaml.load(fp.read(), Loader=yaml.FullLoader)
            self.load(cfg_cxt, cfg_cxt[shared_scope] if shared_scope in cfg_cxt.keys() else dict())

    def load_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as fp:
            self.load(json.load(fp))

    def save_to_json(self, json_path):
        with open(json_path, 'w') as fp:
            save_meta = self.save()
            json.dump(self.save(), fp)

    @staticmethod
    def inverse_single_arg(arg_value):
        if issubclass(type(arg_value), DictRecursive):
            return arg_value.save()
        elif isinstance(arg_value, list):
            list_arg_value = list()
            for a_v in arg_value:
                list_arg_value.append(DictRecursive.inverse_single_arg(a_v))
            return list_arg_value
        elif isinstance(arg_value, np.ndarray):
            return arg_value.tolist()
        else:
            return arg_value

    @staticmethod
    def parse_single_arg(cls_arg, arg_value, shared_dict=None):
        if isinstance(cls_arg, int):
            cls_arg_value = int(arg_value) if arg_value is not None else cls_arg
        elif isinstance(cls_arg, str):
            cls_arg_value = str(arg_value) if arg_value is not None else cls_arg
        elif isinstance(cls_arg, float):
            cls_arg_value = float(arg_value) if arg_value is not None else cls_arg
        elif isinstance(cls_arg, list):
            cls_arg_value = list()
            cls_arg_e = str() if not cls_arg else cls_arg[0]
            if arg_value is not None:
                for a_v in arg_value:
                    cls_arg_value.append(DictRecursive.parse_single_arg(cls_arg_e, a_v, shared_dict))
        elif isinstance(cls_arg, dict):
            if arg_value is not None:
                cls_arg_value = dict()
                for a_v in arg_value:
                    cls_arg_value[a_v] = arg_value[a_v]
            else:
                cls_arg_value = cls_arg
        elif isinstance(cls_arg, np.ndarray):
            if arg_value is not None:
                cls_arg_value = np.asarray(arg_value, cls_arg.dtype)
            else:
                cls_arg_value = cls_arg
        elif issubclass(type(cls_arg), DictRecursive):
            cls_arg_value = type(cls_arg)()
            cls_arg_value.load(arg_value, shared_dict)
        else:
            raise NotImplementedError
        return cls_arg_value

    def match_function_args(self, external_dict, target_func):
        args_dict = copy.deepcopy(external_dict)
        for func_key in inspect.signature(target_func).parameters.keys():
            if func_key not in self.__dict__.keys():
                continue
            if func_key in args_dict.keys():
                continue
            args_dict[func_key] = self.__dict__[func_key]
        return args_dict


class ProcessUnit(DictRecursive):
    """
    Pipeline units
    """
    def __init__(self):
        super().__init__()
        self.assemble_function = str()
        self.name = str()
        self.stride = 1
        self.attrs = dict()
        self.in_paths = list()
        self.out_paths = list()


class EntryConfig(DictRecursive):
    """
    Main entry of each task
    """
    def __init__(self):
        super().__init__()
        self.assemble_class = str()
        self.process_pipelines = list([ProcessUnit()])


class EnvsConfig(DictRecursive):
    """
    Global environments
    """
    def __init__(self):
        super().__init__()
        self.in_data_root = str()
        self.out_data_root = str()
        self.io_paths: Dict[str, str] = dict()

    def get_env_path(self, env_name: str):
        """
        Get the absolute folder path by the env name
        """
        if 'in_data_root' not in self.io_paths:
            self.io_paths['in_data_root'] = self.in_data_root
            self.io_paths['out_data_root'] = self.out_data_root
        self.io_paths = {
            "points": os.path.join(self.out_data_root, "points"),
            "semantic_mask": os.path.join(self.out_data_root, "semantic_mask"),
            "instance": os.path.join(self.out_data_root, "instance"),
            "annotations": os.path.join(self.out_data_root, "annotations"),
        }
        rel_path = self.io_paths[env_name]
        while True:
            regex_pattern = r'\$.*\$'
            patterns = re.findall(regex_pattern, rel_path)
            if not patterns:
                break
            rel_path = rel_path.replace(patterns[0], self.io_paths[patterns[0][1:-1]])
        return rel_path


class StreamingTasks(DictRecursive):
    """
    Main entry of streaming tasks
    """
    def __init__(self):
        super().__init__()
        self.envs = EnvsConfig()
        self.streaming_lines = list([EntryConfig()])


class EntryBase:
    """
    The basic config of entry
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        self.proc_units = proc_units
        self.envs = envs

    def execute_pipeline(self):
        """
        execute the data processing pipeline
        """
        for proc_unit in self.proc_units:
            proc_func = getattr(self, proc_unit.assemble_function)
            proc_func(proc_unit)


class MPEntryBase(EntryBase):
    """
    The multi-process config of entry
    """
    def __init__(self, proc_units: List[ProcessUnit], envs: EnvsConfig) -> None:
        super().__init__(proc_units, envs)
        self._enable_mp = True
        self._num_worker = 8

    @abstractmethod
    def _sample_list(self):
        """
        Return the list of samples to be processed
        """

    def _execute_proc_unit(self, sample: str, proc_unit: ProcessUnit, shared_vars: Dict):
        proc_func = getattr(self, proc_unit.assemble_function)
        proc_func(sample, proc_unit, shared_vars)

    def _merged_cross_processing(self, ipc_vars):
        """
        Merge all shared list information cross all processors
        """

    def _merged_within_processing(self, shared_vars, ipc_vars):
        """
        Merge all information within a processor
        """

    def _mp_execute_pipeline(self, samples, ipc_vars: List, worker_offset=0, worker_id=0):
        del worker_offset
        logging.info(f'worker {worker_id} begin...')
        shared_vars = dict()
        for s_idx, sample in enumerate(samples):
            for proc_unit in self.proc_units:
                if s_idx % proc_unit.stride != 0:
                    continue
                self._execute_proc_unit(sample, proc_unit, shared_vars)
        self._merged_within_processing(shared_vars, ipc_vars)
    
    def multiple_processor(func, samples: List, workers, args: Tuple):
        samples_per_worker = int((len(samples) - 1) / workers + 1)
        processes = list()
        for w in range(workers):
            start_index = w * samples_per_worker
            end_index = min((w + 1) * samples_per_worker, len(samples))
            f_args = (samples[start_index: end_index], ) + args + (start_index, w)
            t = multiprocessing.Process(target=func, args=f_args)
            processes.append(t)
            t.start()
        for p in processes:
            p.join()
            
    def execute_pipeline(self):
        logging.info(f'- Start to execute pipeline {self.__class__.__name__}')
        samples = self._sample_list()
        ipc_vars = mp.Manager().list()
        if self._enable_mp:
            __class__.multiple_processor(self._mp_execute_pipeline, samples, workers=8, \
                args=(ipc_vars, ))
        else:
            self._mp_execute_pipeline(samples, ipc_vars)
        self._merged_cross_processing(list(ipc_vars))
        logging.info(f'- Finished to execute pipeline {self.__class__.__name__}')