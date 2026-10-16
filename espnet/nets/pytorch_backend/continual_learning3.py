# Copyright 2022 Steven Vander Eeckt

""" Continual Learning extensions for End-to-End ASR (ESPnet2) """
import time
import numpy as np
import torch
import logging
from abc import ABC, abstractmethod
import os
from espnet2.fileio.read_text import load_num_sequence_text
import torch.nn.functional as func
import copy
import typing
import random

from espnet2.layers.create_adapter_utils import (
    get_submodules,
    replace_module,
)
from espnet2.torch_utils.model_summary import model_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################
########################### HELP FUNCTIONS ##########################################################################
#####################################################################################################################


def distillation_loss(
        teacher_output: torch.tensor,
        student_output: torch.tensor,
        dim: int = 2,
        neg: bool = True,
        teacher_softmax: bool = True,
        divide_by_batch_size: bool = True,
        per_sample: bool = False
):
    """
        Distillation loss between teacher and student

        :param torch.Tensor teacher_output: 3D (2D) logits of teacher
        :param torch.Tensor student_output: 3D (2D) logits of student
        :param int dim: dimension along which to apply softmax
        :param bool neg: if True, -loss is returned, else loss
        :param bool teacher_softmax: if False, teacher_output is already softmaxed
        :param bool divide_by_batch_size: if True, loss is divded by batch size
        :param bool per_sample: if True, loss is returned per mini-batch
    """
    sm_t, lsm_s = torch.nn.Softmax(dim=dim), torch.nn.LogSoftmax(dim=dim)
    prob_t = sm_t(teacher_output) if teacher_softmax else teacher_output
    log_s = lsm_s(student_output)
    if not per_sample:
        loss = (prob_t * log_s).sum(dim=dim).sum()
    else:
        loss = (prob_t * log_s).sum(dim=dim).sum(dim=-1)
    # check if there is a batch size
    if divide_by_batch_size and len(teacher_output.size()) > 2:
        # if there is, divide loss by it
        loss = loss / teacher_output.size(0)
    return -loss if neg else loss


def _recursive_to(
        xs,
        device
):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple) or isinstance(xs, list):
        return tuple(_recursive_to(x, device) for x in xs)
    if isinstance(xs, dict):
        return {n: _recursive_to(p, device) for n, p in xs.items()}
    return xs

def clone_module(module):
    # Create a new instance of the same class as the original module
    cloned_module = type(module)()

    # Copy the parameters from the original module to the new module
    for original_param, cloned_param in zip(module.parameters(), cloned_module.parameters()):
        cloned_param.data = original_param.data.clone()

    return cloned_module

def copy_model(
        model: torch.nn.Module,
        device: str,
        train: bool = False,
        requires_grad: bool = False,
):
    """
        Copy the model and make it untrainable

        :param torch.nn.Module model: the model
        :param str device:  cpu or gpu device to map model to
        :param bool train: if False, model is set to eval mode
        :param bool requires_grad:
    """
    # for batch norm and dropout layers
    old_model = copy.deepcopy(model).train(train)
    for name, param in old_model.named_parameters():
        param.requires_grad = requires_grad
    return old_model.to(device)


def item(t):
    try:
        return t.item()
    except:
        return t


def cuda_overview(
        message: str = "",
):
    """
        Displays CUDA memory information

        :param str message: additional message to display (e.g. location in code)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory * 10.0 ** (-9)
        r = torch.cuda.memory_reserved(0) * 10.0 ** (-9)
        a = torch.cuda.memory_allocated(0) * 10.0 ** (-9)
        f = r - a  # free inside reserved
        logging.info("%s: CUDA memory - "
                     "total: %.4f; reserved: %.4f; allocated: %.4f; available: %.4f" % (message, t, r, a, f))
    else:
        pass


class timer:
    """
        Timer class to debug and optimize code
    """
    def __init__(
            self,
            logging: bool = True
    ):
        self.last = 0
        self.logging = logging

    def time(
            self,
            name: str = ""
    ):
        t = time.time()
        if name:
            if self.logging:
                logging.info("::timing:: %s = %ss" % (name, str(t-self.last)))
            else:
                print("::timing:: %s = %ss" % (name, str(t-self.last)))
        self.last = t


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0  # Sum of squares of differences from the current mean

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        if self.n < 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)

    def get_std_dev(self):
        return self.get_variance() ** 0.5


#####################################################################################################################
########################### MEMORY CLASSES ##########################################################################
#####################################################################################################################


class Memory(ABC):

    @abstractmethod
    def update(
            self,
            xs,
    ):
        pass

    @abstractmethod
    def next(
            self
    ):
        pass

    @abstractmethod
    def save(
            self,
            outdir: str,
    ):
        pass

    @staticmethod
    def load(
            filename: str,
    ):
        pass

    @staticmethod
    def get_unique_key(
            names: typing.List[str],
    ):
        """
            Generates unique key for batch with utterances in names

            :param List[str] names: list of utterance IDs of batch
        """
        return "_".join(sorted(names))
    
    @abstractmethod
    def from_original_memory(
            self,
            names,
    ):
        pass


class ReservoirBuffer(Memory):
    def __init__(
            self,
            iter = None,
            seen_data: int = 0,  # expressed in samples
            size: int = 500,
            save_name: str = 'reservoir_buffer',
    ):
        self.iter = iter
        self.seen_data = seen_data
        self.size = size
        self.current_size = 0
        data = []
        self.memory_batch_sizes = []
        self.original_memory = []  # store names of samples that were in the original memory
        if iter is not None:
            logging.info("Adding samples from iter to memory...")
            for (names, xs) in self.iter:
                data.append(_recursive_to(xs, 'cpu'))
                self.original_memory.append(self.get_unique_key(names))
                self.current_size += xs['speech'].size(0)
                self.memory_batch_sizes.append(xs['speech'].size(0))
                logging.info("Memory contains: %d mini-batch - %d samples" % (len(data), self.current_size))
        self.memory_batch_sizes = np.cumsum(np.array(self.memory_batch_sizes))
        self.data = data
        self.save_name = save_name + '%d' % self.size
        logging.info("Buffer has size: %d (official: %d)" % (self.current_size, self.size))
        logging.info("..... seen_data = %d" % (seen_data))

    def replace_sample(
            self,
            b: int,  # batch
            pos: int,  # position in batch
            xs,  # new batch
            j: int,  # sample in new batch
    ):
        logging.info("Replacing sample.. (batch = %d, pos = %d)" % (b, pos))
        logging.info("... speech length of: batch = %d, new sample = %d" %
                     (self.data[b]['speech'].size(1), xs['speech_lengths'][j]))
        logging.info("... text length of: batch = %d, new sample = %d" %
                     (self.data[b]['text'].size(1), xs['text_lengths'][j]))
        # 1. the speech
        # option a. the sample fits in the mini-batch
        if xs['speech_lengths'][j] < self.data[b]['speech'].size(1):
            logging.info("... --> Speech fits in the mini-batch")
            x = torch.zeros_like(self.data[b]['speech'][pos])
            x[:xs['speech_lengths'][j]] = xs['speech'][j, :xs['speech_lengths'][j]]
            self.data[b]['speech'][pos] = x
        # option b. sample does not fit in the mini-batch
        else:
            logging.info("... --> Speech DOES NOT fit in the mini-batch")
            batch = torch.zeros([self.data[b]['speech'].size(0), xs['speech_lengths'][j]])
            batch[:, :self.data[b]['speech'].size(1)] = self.data[b]['speech']
            batch[pos, :] = xs['speech'][j, :xs['speech_lengths'][j]]
            self.data[b]['speech'] = batch
        # 2. the speech lengths
        self.data[b]['speech_lengths'][pos] = xs['speech_lengths'][j]
        # 3. the text
        # option a. text fits in the mini-batch
        if xs['text_lengths'][j] < self.data[b]['text'].size(1):
            logging.info("... --> Text fits in the mini-batch")
            x = torch.zeros_like(self.data[b]['text'][pos])
            x[:xs['text_lengths'][j]] = xs['text'][j, :xs['text_lengths'][j]]
            self.data[b]['text'][pos] = x
        # option b. text does not fit in the mini-batch
        else:
            logging.info("... --> Text DOES NOT fit in the mini-batch")
            batch = -torch.ones([self.data[b]['text'].size(0), xs['text_lengths'][j]])
            batch[:, :self.data[b]['text'].size(1)] = self.data[b]['text']
            batch[pos, :] = xs['text'][j, :xs['text_lengths'][j]]
            self.data[b]['text'] = batch.long()
        # 4. the text lengths
        self.data[b]['text_lengths'][pos] = xs['text_lengths'][j]
        return

    def update(
            self,
            xs,
    ):
        xs = _recursive_to(xs, 'cpu')
        batch_size = xs['speech'].size(0)
        if self.current_size >= self.size:
            # with prob size / seen_data: add sample
            # generate p for all samples in mini-batch
            p = np.random.rand(batch_size)
            logging.info("(p <= %5f) = %s" % (self.size / self.seen_data, str(p <= self.size / self.seen_data)))
            # for all samples where p <= 1.0 / seen_data:
            for i in np.where(p <= self.size / self.seen_data)[0]:
                # randomly remove item
                j = random.randint(0, self.current_size - 1)
                # find batch of sample j
                b = np.where(self.memory_batch_sizes > j)[0][0]
                # find position in mini-batch
                a = j - (self.memory_batch_sizes[b-1] if b > 0 else 0)
                logging.info("Removing sample %d --> batch = %d, sample = %d" % (j, b, a))
                self.replace_sample(b, a, xs, i)
                logging.info("Adding sample to buffer - removing sample %d" % j)
        self.seen_data += xs['speech'].size(0)

    def next(
            self
    ):
        i = random.randint(0, len(self.data) - 1)
        return "", self.data[i]

    def save(
            self,
            outdir: str,
    ):
        memory = {'data': self.data, 'size': self.size, 'seen_data': self.seen_data, 'save_name': self.save_name}
        logging.info("storing memory at %s" % str(outdir))
        torch.save(memory, outdir + '/' + self.save_name)

    @staticmethod
    def load(
            filename: str,
    ):
        memory = torch.load(filename)
        rb = ReservoirBuffer(seen_data=memory['seen_data'], size=memory['size'], save_name=memory['save_name'])
        rb.data = memory['data']
        logging.info("Loading the memory.. name = %s" % str(filename))
        logging.info("Buffer has size: %d" % len(rb.data))
        return rb
    
    def from_original_memory(
            self,
            names,
    ):
        return self.get_unique_key(names) in self.original_memory
    

class NamedReservoirBuffer(ReservoirBuffer):
    def __init__(
            self,
            iter = None,
            seen_data: int = 0,  # expressed in samples
            size: int = 500,
            save_name: str = 'reservoir_buffer',
    ):
        self.iter = iter
        self.seen_data = seen_data
        self.size = size
        self.current_size = 0
        data, data_names = [], []
        self.memory_batch_sizes = []
        self.original_memory = []  # store names of samples that were in the original memory
        if iter is not None:
            logging.info("Adding samples from iter to memory...")
            for (names, xs) in self.iter:
                data.append(_recursive_to(xs, 'cpu'))
                data_names.append(names)
                self.original_memory += names
                self.current_size += xs['speech'].size(0)
                self.memory_batch_sizes.append(xs['speech'].size(0))
                logging.info("Memory contains: %d mini-batch - %d samples" % (len(data), self.current_size))
        self.memory_batch_sizes = np.cumsum(np.array(self.memory_batch_sizes))
        self.data = data
        self.data_names = data_names
        self.save_name = save_name + '%d' % self.size
        logging.info("Buffer has size: %d (official: %d)" % (self.current_size, self.size))
        logging.info("..... seen_data = %d" % (seen_data))

    def replace_sample(
            self,
            b: int,  # batch
            pos: int,  # position in batch
            xs,  # new batch
            j: int,  # sample in new batch
    ):
        logging.info("Replacing sample.. (batch = %d, pos = %d)" % (b, pos))
        logging.info("... speech length of: batch = %d, new sample = %d" %
                     (self.data[b]['speech'].size(1), xs['speech_lengths'][j]))
        logging.info("... text length of: batch = %d, new sample = %d" %
                     (self.data[b]['text'].size(1), xs['text_lengths'][j]))
        # 1. the speech
        # option a. the sample fits in the mini-batch
        if xs['speech_lengths'][j] < self.data[b]['speech'].size(1):
            logging.info("... --> Speech fits in the mini-batch")
            x = torch.zeros_like(self.data[b]['speech'][pos])
            x[:xs['speech_lengths'][j]] = xs['speech'][j, :xs['speech_lengths'][j]]
            self.data[b]['speech'][pos] = x
            self.data_names[b][pos] = ""
        # option b. sample does not fit in the mini-batch
        else:
            logging.info("... --> Speech DOES NOT fit in the mini-batch")
            batch = torch.zeros([self.data[b]['speech'].size(0), xs['speech_lengths'][j]])
            batch[:, :self.data[b]['speech'].size(1)] = self.data[b]['speech']
            batch[pos, :] = xs['speech'][j, :xs['speech_lengths'][j]]
            self.data[b]['speech'] = batch
            self.data_names[b][pos] = ""
        # 2. the speech lengths
        self.data[b]['speech_lengths'][pos] = xs['speech_lengths'][j]
        # 3. the text
        # option a. text fits in the mini-batch
        if xs['text_lengths'][j] < self.data[b]['text'].size(1):
            logging.info("... --> Text fits in the mini-batch")
            x = torch.zeros_like(self.data[b]['text'][pos])
            x[:xs['text_lengths'][j]] = xs['text'][j, :xs['text_lengths'][j]]
            self.data[b]['text'][pos] = x
        # option b. text does not fit in the mini-batch
        else:
            logging.info("... --> Text DOES NOT fit in the mini-batch")
            batch = -torch.ones([self.data[b]['text'].size(0), xs['text_lengths'][j]])
            batch[:, :self.data[b]['text'].size(1)] = self.data[b]['text']
            batch[pos, :] = xs['text'][j, :xs['text_lengths'][j]]
            self.data[b]['text'] = batch.long()
        # 4. the text lengths
        self.data[b]['text_lengths'][pos] = xs['text_lengths'][j]
        return

    def update(
            self,
            xs,
    ):
        xs = _recursive_to(xs, 'cpu')
        batch_size = xs['speech'].size(0)
        if self.current_size >= self.size:
            # with prob size / seen_data: add sample
            # generate p for all samples in mini-batch
            p = np.random.rand(batch_size)
            logging.info("(p <= %5f) = %s" % (self.size / self.seen_data, str(p <= self.size / self.seen_data)))
            # for all samples where p <= 1.0 / seen_data:
            for i in np.where(p <= self.size / self.seen_data)[0]:
                # randomly remove item
                j = random.randint(0, self.current_size - 1)
                # find batch of sample j
                b = np.where(self.memory_batch_sizes > j)[0][0]
                # find position in mini-batch
                a = j - (self.memory_batch_sizes[b-1] if b > 0 else 0)
                logging.info("Removing sample %d --> batch = %d, sample = %d" % (j, b, a))
                self.replace_sample(b, a, xs, i)
                logging.info("Adding sample to buffer - removing sample %d" % j)
        self.seen_data += xs['speech'].size(0)

    def next(
            self
    ):
        i = random.randint(0, len(self.data) - 1)
        return self.data_names[i], self.data[i]

    def save(
            self,
            outdir: str,
    ):
        memory = {'data': self.data, 'size': self.size, 'seen_data': self.seen_data, 'save_name': self.save_name}
        logging.info("storing memory at %s" % str(outdir))
        torch.save(memory, outdir + '/' + self.save_name)

    @staticmethod
    def load(
            filename: str,
    ):
        memory = torch.load(filename)
        rb = ReservoirBuffer(seen_data=memory['seen_data'], size=memory['size'], save_name=memory['save_name'])
        rb.data = memory['data']
        logging.info("Loading the memory.. name = %s" % str(filename))
        logging.info("Buffer has size: %d" % len(rb.data))
        return rb
    
    def from_original_memory(
            self,
            names,
    ):
        return [name in self.original_memory for name in names]

        
class FIFOBuffer(Memory):
    """
        Very simple buffer that stores a certain number of batches,
        whenever a new batch is encountered the oldest is replaced.

    """
    def __init__(
            self,
            iter = None,
            size: int = 500,
            max_batches: int = 10,
            save_name: str = 'fifo_buffer',
            replay_oldest = True
    ):
        self.iter = iter
        self.size = size
        self.current_size = 0
        self.replay_oldest = replay_oldest
        logging.info("replay_oldest = %s" % str(replay_oldest))
        self.max_batches = max_batches
        data = []
        if iter is not None:
            logging.info("Adding samples from iter to memory...")
            while True:
                try:
                    names, xs = self.iter.next()
                    data.append(_recursive_to(xs, 'cpu'))
                    self.current_size += xs['speech'].size(0)
                except StopIteration as e:
                    break
                logging.info("Memory contains: %d mini-batch - %d samples" % (len(data), self.current_size))
        self.data = data
        self.save_name = save_name + '%d' % self.size
        logging.info("Buffer has size: %d (official: %d) - contains %d batches (max: %d)" %
                     (self.current_size, self.size, len(self.data), self.max_batches))

    def replace_sample(
            self,
            xs,  # new batch
    ):
        self.data.append(xs)
        self.current_size += xs['speech'].size(0)
        if self.current_size > self.size or len(self.data) > self.max_batches:
            logging.info("Buffer overflow: samples = %d (max: %d) - batches = %d (max: %d)" %
                     (self.current_size, self.size, len(self.data), self.max_batches))
            self.current_size -= self.data[0]['speech'].size(0)
            self.data = self.data[1:]
            logging.info("Removed oldest batch! Size: samples = %d (max: %d) - batches = %d (max: %d)" %
                     (self.current_size, self.size, len(self.data), self.max_batches))
        return

    def update(
            self,
            xs,
    ):
        xs = _recursive_to(xs, 'cpu')
        self.replace_sample(xs)

    def next(
            self
    ):
        if self.replay_oldest:
            i = 0
        else:
            i = random.randint(0, len(self.data) - 1)
        return "", self.data[i]

    def save(
            self,
            outdir: str,
    ):
        memory = {'data': self.data, 'size': self.size, 'save_name': self.save_name}
        logging.info("storing memory at %s" % str(outdir))
        torch.save(memory, outdir + '/' + self.save_name)

    @staticmethod
    def load(
            filename: str,
    ):
        memory = torch.load(filename)
        rb = FIFOBuffer(size=memory['size'], save_name=memory['save_name'])
        rb.data = memory['data']
        logging.info("Loading the memory.. name = %s" % str(filename))
        logging.info("Buffer has size: %d" % len(rb.data))
        return rb


#####################################################################################################################
########################### ABSTRACT CLASSES ########################################################################
#####################################################################################################################


"""
ABSTRACT CLASSES FOR CL METHODS: HIERARCHY
    -- CL_Method
        -- LossBasedCL: "Add a regularization term to the ASR loss"
        
        -- GradBasedCL: "Modifies the gradient before using the gradient to update the parameters"
        
        -- PartialCLMethod: "Indicates that the method must know which parameters to regularize"

        -- WeightBased: "Includes methods such as averaging of model parameters" 

        -- OnlineCL: "For method that follow Online CL paradigm"
            -- OnlineWeightBased: "WeightBased CL for Online CL"
            -- OnlineRehearsal: "Rehearsal-based CL (see below) for Online CL"

        -- UnsupervisedCL: "For CL without labels to train new tasks, assume method to generate pseudo-labels"
            
        -- Rehearsal:       "Use a memory of samples from previous tasks"
            -- KnowledgeDistillationBased:   "Uses KD on a memory to distill knowledge from old to new model"
        -- Regularization:  "EWC-like regularization-based methods, with importance weights"
        -- DataFocused:     "LWF-like regularization-based methods, using the new task's mini-batch"
        NOTE: A CL method should not inherit Rehearsal, Regularization or DataFocused at the same time
            
"""


class CLMethod(ABC):
    """
        CL Method Abstract Class
    """
    pass


class OnlineCL(CLMethod):
    """
        Online CL Abstract class

        :method update_model: method called after training one epoch
    """
    @abstractmethod
    def update_model(
            self,
            model: torch.nn.Module,
    ):
        """
            This method is called after each epoch. Since in Online CL, one task lasts one epoch,
            update_model allows the CLMethod to store its 'states', i.e. all the necessary information to proceed
            training. Since it is Online CL, in which no task boundaries are known, nothing but storing information
            is allowed in the body of this method.
        """
        pass


class LossBasedCL(CLMethod):
    """
        Loss-Based CL Abstract Class

        Includes all methods, regularization and rehearsal, which enable CL by means of a regularizing loss

        :method compute_loss: each class inheriting LossBasedCL must have a compute_loss(model) method
    """
    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the loss given the model
            :param torch.nn.Module model:
        """
        pass


class GradBasedCL(CLMethod):
    """
        Grad-Based CL Abstract Class
        Includes all methods, regularization and rehearsal, which manipulate the model's gradients to enable CL

        :method grad_fn: each class inheriting GradBasedCL must have a grad_fn(model) method
    """
    @abstractmethod
    def grad_fn(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the loss given the model
            :param torch.nn.Module model:
        """
        pass

class WeightBasedCL(CLMethod):
    """
        Weight-based CL Abstract Class

        Includes all methods, regularization and rehearsal, which update the weights (after an epoch or iteration)

        :method update_model
    """
    @abstractmethod
    def update_model(
            self,
            model: torch.nn.Module
    ):
        """
            Called after each epoch to update the weights of the model
        """
        pass

    @staticmethod
    def to_param(
            x: torch.tensor,
            sizes,
    ):
        """
            Turns a vector into a state_dict
            :param torch.tensor x: the tensor as 1D vector
            :param dict sizes: dict with key the layer and value a dict with p.numel() and p.size()

        """
        sd, k = {}, 0
        for n, p in sizes.items():
            sd[n] = x[k:k + p['numel']].view(p['size'])
            k += p['numel']
        return sd


class OnlineWeightBasedCL(OnlineCL, WeightBasedCL):
    """
        Online Weight-based CL Class

        :method
    """
    @abstractmethod
    def consolidate_update(
            self,
            model: torch.nn.Module,
    ):
        """
            Called after each SGD update
        """
        pass


class SVDBasedCL(CLMethod):
    def __init__(
            self,
            init_param: str,
            outdir: str,
            gamma: float = 10.0,
            alpha: float = 1.0,
    ):
        if isinstance(init_param, list) or isinstance(init_param, tuple):
            init_param = init_param[0]
        # load state dict of old model
        try:
            logging.info(f"Loading old model: {init_param}")
            self.old_model = torch.load(init_param)
        except FileNotFoundError as fnfe:
            logging.warning(fnfe)
            self.old_model = {}
        self.outdir = outdir
        self.gamma = gamma
        self.alpha = alpha

    @staticmethod
    def set_old_bias(
            current_bias: torch.nn.Parameter,
            old_bias: torch.Tensor,
    ):
        """
        Set the current bias parameter based on the old bias.

        Args:
            current_bias (torch.nn.Parameter): The current bias parameter to be updated.
            old_bias (torch.Tensor): The old bias tensor to use for updating.
        """
        # Ensure shapes match
        assert current_bias.size() == old_bias.size(), "Size mismatch between current_bias and old_bias"
        avg_bias = (current_bias.data + old_bias.to(current_bias.device)) / 2
        current_bias.data.copy_(avg_bias)

    """
        SVD-Based CL - writes weight matrices of linear layers as U, S, Vh = W and trains only S.
    """
    def set_model(
            self,
            model: torch.nn.Module,
            optimizers: typing.List[torch.optim.Optimizer],
    ):
        """
          Sets the old model

          :param torch.nn.Module model: the old model
        """
        # do nothing if eval mode
        if not model.train:
            return
        # new SVDLinear module
        class SVDLinear(torch.nn.Linear):
            def __init__(
                    self,
                    in_features: int,
                    out_features: int,
                    gamma: float = 1.0,
                    **kwargs
            ):
                torch.nn.Linear.__init__(self, in_features, out_features, **kwargs)
                self.U, self.S, self.Vh, self.alpha, self.M = None, None, None, None, None
                self.sigmoid = None
                self.gamma = gamma

            def reset_parameters(self):
                torch.nn.Linear.reset_parameters(self)

            def prepare_svd(self, old_weight):
                weight_diff = self.weight - old_weight.to(self.weight.device)
                U, S, Vh = torch.linalg.svd(weight_diff, full_matrices=False)
                # set old_weight as self.weight
                self.weight.data.copy_(old_weight)
                # freeze weight
                self.weight.requires_grad = False
                if self.bias is not None:
                    self.bias.requires_grad = False
                # only train S
                self.U = torch.nn.Parameter(U, requires_grad=False)
                self.S = torch.nn.Parameter(S, requires_grad=False)
                self.Vh = torch.nn.Parameter(Vh, requires_grad=False)
                # introduce learnable parameter alpha
                # alpha == 0.0 at the start
                self.alpha = torch.nn.Parameter(torch.ones_like(S) * -10.00 / self.gamma, requires_grad=True)
                self.sigmoid = lambda x: func.sigmoid(self.gamma * x)

            def extra_repr(self):
                s = f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}; \n'
                """Extra representation of the module to include adapter information."""
                k = min(self.in_features, self.out_features)
                s += f'   U = ({self.in_features}, {k}), S = ({k}, {k}), alpha = ({k}), Vh = ({k}, {self.out_features})'
                return s

            def train(self, mode: bool = True):
                torch.nn.Linear.train(self, mode)

            def eval(self):
                torch.nn.Linear.eval(self)

            def state_dict(self, destination=None, prefix='', keep_vars=False):
                """Override state_dict to only store the reconstructed weight and bias."""
                state = super().state_dict(destination, prefix, keep_vars)
                # Reconstruct the full weight matrix
                full_weight = self.weight + self.U @ torch.diag(self.sigmoid(self.alpha) * self.S) @ self.Vh
                # Replace entries in state_dict
                state[prefix + 'weight'] = full_weight
                # Remove U, S, Vh from state_dict
                state.pop(prefix + 'U', None)
                state.pop(prefix + 'S', None)
                state.pop(prefix + 'Vh', None)
                state.pop(prefix + 'alpha', None)
                return state

            def forward(self, x: torch.Tensor):
                assert self.U is not None or self.S is not None or self.Vh is not None
                result = torch.nn.functional.linear(x, self.weight +
                                                    self.U @ torch.diag(self.sigmoid(self.alpha) * self.S) @ self.Vh,
                                                    bias=self.bias)
                return result

        # keep track of replaced layers
        replaced_layers = set()
        # replace every linear layer with an SVDLinear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # create new module
                new_module = SVDLinear(module.in_features, module.out_features,
                                       gamma=self.gamma, bias=module.bias is not None)
                parent_module, target_name, target_module = get_submodules(model, name)
                replace_module(parent_module, target_name, module, new_module, copy_weight=True)
                old_weight = self.old_model[name + ".weight"]
                new_module.prepare_svd(old_weight)
                replaced_layers.add(name)
            # BatchNorm and LayerNorm Handling
            elif (isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm))
                  and module.weight.requires_grad):
                # Fetch old model parameters
                old_weight = self.old_model[name + ".weight"]
                old_bias = self.old_model[name + ".bias"]
                module.weight.data.copy_(old_weight.to(module.weight.device))
                if module.bias is not None:
                    module.bias.data.copy_(old_bias.to(module.weight.device))
        # Update parameters for layers that were **not** replaced
        for name, param in model.named_parameters():
            # Extract the module name by removing the parameter suffix
            module_name = ".".join(name.split(".")[:-1])
            # If the layer was not replaced, apply averaging
            if module_name not in replaced_layers:
                old_param = self.old_model[name]
                param.data.copy_((old_param.to(param.device) + param) / 2)
                logging.info(f"Averaging {name}...")
            else:
                logging.info(f"NOT Averaging {name}...")
        # update param groups of optimizers[0]
        alphas = [p for p in model.parameters() if p.requires_grad and p is not None]
        optimizers[0].param_groups.clear()
        optimizers[0].add_param_group({'params': alphas})
        logging.info(model_summary(model))

    def update_model(self, model: torch.nn.Module):
        alphas, sing = {}, {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'alpha'):
                alpha = module.sigmoid(module.alpha)
                alphas[name] = alpha
        torch.save(alphas, f"{self.outdir}/alphas.pth")


class Rehearsal(CLMethod):
    """
        Rehearsal Abstract Class
        Generic class to be inherited by Rehearsal-based CL methods

        :param str device:
        :param str task_file : task file to map utterances to tasks (task IDs)
        :param str task_order : sets the task IDs in the right order (e.g. '1 2 0 3' means that task with task ID = 1 is
                                the first task, task with task ID = 2 the second, and so on)
    """
    def __init__(
            self,
            device: str,
    ):
        self.device = device
        # set the default get_batch() function.
        self.get_batch = lambda return_names=False: self._get_batch(return_names)

    def set_loader(
            self,
            loader
    ):
        """
            Set the loader and iterator of the rehearsal-based method

            :param torch.Dataloader loader: the dataloader for the memory of the rehearsal-based method
        """
        self.loader = loader.build_iter(epoch=0, shuffle=True)
        self.iter = iter(self.loader)

    @staticmethod
    def get_unique_key(
            names: typing.List[str],
    ):
        """
            Generates unique key for batch with utterances in names

            :param List[str] names: list of utterance IDs of batch
        """
        return "_".join(sorted(names))

    def _get_batch(
            self,
            return_names: bool = False
    ):
        """
            Sample a batch from the iterator (memory)
        """
        try:
            #names, xs = self.iter.next()
            names, xs = next(self.iter)
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            #names, xs = self.iter.next()
            names, xs = next(self.iter)
        if return_names:
            return _recursive_to(xs, self.device), names
        return  _recursive_to(xs, self.device)

    def get_task_from_batch(
            self,
            names: typing.List[str]
    ):
        """
            Given (the utterance IDs of a batch), return the task number

            :param List[str] names: list containing the utterance IDs of the batch
        """
        T = timer()
        T.time()
        # generate a unique hashable key
        key = self.get_unique_key(names)
        # select task from batch2task if it exists:
        try:
            task = self.batch2task[key]
            return task
        except Exception as e:  # the first time we encounter this batch
            tasks = set([self.utt2task[name] for name in names])
            assert len(tasks) == 1, "batch %s had multiple tasks: %s" % (str(names), str(tasks))
            self.batch2task[key] = list(tasks)[0]
            return list(tasks)[0]

    def _get_batch_and_task(
            self,
            return_names: bool = False
    ):
        """
            Sample a batch from the iterator (memory)
            and add the task to the batch.

            :param bool return_names: return utterance IDs
        """
        T = timer()
        T.time()
        try:
            names, xs = self.iter.next()
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            names, xs = self.iter.next()
        # get the task of the batch
        task = self.get_task_from_batch(names)
        # add it to the batch
        xs['task'] = task
        if return_names:
            return _recursive_to(xs, self.device), names
        return _recursive_to(xs, self.device)

class KnowledgeDistillationBased(LossBasedCL, Rehearsal):
    """
        Knowledge Distillation-based Rehearsal Abstract Class

        :param str device: the device for the model
        :param float alpha: the weight of the regularization
        :param float ctc_weight: the weight of CTC in the KL div loss
        :param float temperature: temperature of the KL div loss
        :param bool eval_mode: if True, old model (which generates logits) is in eval mode
        :param str task_file: see Rehearsal
        :param str task_order: see Rehearsal
    """
    def __init__(
            self,
            device: str,
            alpha: float = 1.0,
            ctc_weight: float = 0.3,
            temperature: float = 1,
            eval_mode: bool = True,
    ):
        super(KnowledgeDistillationBased, self).__init__(device=device)
        self.ctc_weight = ctc_weight
        self.alpha = alpha
        self.T = 1.0 * temperature
        self.eval_mode = eval_mode
        self.KLdiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def distillation_loss(
            self,
            teacher_output: torch.tensor,
            student_output: torch.tensor,
    ):
        """
            Computes the distillation loss
            :param torch.Tensor teacher_output:
            :param torch.Tensor student_output:
        """
        input, target = func.softmax(student_output, dim=2), func.log_softmax(teacher_output, dim=2)
        return self.KLdiv(input, target)

    def set_model(
            self,
            model: torch.nn.Module
    ):
        """
            Sets the old model

            :param torch.nn.Module model: the old model
        """
        self.old_model = copy_model(model, self.device, train=not self.eval_mode)

    def set_loader(
            self,
            loader
    ):
        """
            Set the loader and iterator of the rehearsal-based method

            :param torch.Dataloader loader: the dataloader for the memory of the rehearsal-based method
        """
        self.loader = loader.build_iter(epoch=0, shuffle=True)
        self.iter = iter(self.loader)
        # if self.eval_mode == True, output of model is deterministic
        # therefore, we can already compute the output of the memory by the old model
        #if self.eval_mode:
        #    self.old_outputs = self.collect_outputs()

    def collect_outputs(
            self
    ):
        """
            In case old_model is in eval_mode, outputs are deterministic and they can be computed before training
            That way, they don't have to be computed at each step.
        """
        # to store the outputs
        old_outputs = {}
        # sample over all batches in memory
        while True:
            # sample a batch - we also need the utterance IDs
            batch, names = self.get_batch(return_names=True)
            #logging.info("Collecting outputs for batch from memory: %d (size = %s)" % (len(old_outputs) + 1,
            #                                                                           str(batch['speech'].size())))
            # if already present in old_outputs, we have processed all batches
            if self.get_unique_key(names) in old_outputs.keys():
                break
            # model must return the output
            batch['return_args'] = ["decoder_out", "ctc_out"]
            # compute output from old model
            _, old_dec_output, old_ctc_output = self.old_model.forward_er(**batch)
            old_outputs[self.get_unique_key(names)] = {'ctc': old_ctc_output.detach().cpu(),
                                                       'dec': old_dec_output.detach().cpu()}
            logging.info(f"Processed {len(old_outputs)} old outputs..(just added key={self.get_unique_key(names)}")
        return old_outputs

    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        pass


class PartialCLMethod(CLMethod):
    """
        Partial CL Method abstract class

        Includes all methods which regularize only part of the model (if asked by the user)
        and which therefore must have an "update_exclude(exclude)" method

        :method update_exclude(exclude): sets the include function
    """
    def update_exclude(
            self,
            exclude: typing.List[str] = None
    ):
        """
            Set the include function

            :param List[str] exclude: list of layers to exclude from regularization
        """
        self.include = lambda name: exclude is None or not name in exclude
        logging.info("Exclude = %s" % str(exclude))


class Regularization(PartialCLMethod, LossBasedCL):
    """
        Regularization Abstract Class

        Generic class to be inherited by Regularization-based CL methods (EWC, MAS, etc.)

        :param str device:
        :param str prev_outdir: path for the importance weights
        :param float alpha:
        :param str name: name of the importance weights
        :param List[str] exclude: parameters excluded from regularization
    """
    def __init__(
            self,
            device: str,
            prev_outdir: str,
            alpha: float,
            name: str,
            exclude: typing.List[str] = None,
    ):
        if isinstance(prev_outdir, list) or isinstance(prev_outdir, tuple):
            prev_outdir = prev_outdir[0]
        logging.info("prev_outdir = %s " % prev_outdir)
        prev_outdir = '/'.join(prev_outdir.split("/")[:-1])
        if name:
            assert os.path.exists("%s/%s" % (prev_outdir, name)), "File %s/%s does not exist" % (prev_outdir, name)
        self.alpha = alpha
        self.include = lambda name: exclude is None or not name in exclude
        if name:
            self.iw = torch.load("%s/%s" % (prev_outdir, name), map_location=device)
        else:
            self.iw = None
        self.old_model = None
        self.device = device

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        """
            Sets the old model

            :param torch.nn.Module model: the old model
        """
        self.old_model = {n: 1.0*p for n, p in model.state_dict().items()}

    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the regularization loss

            :param torch.nn.Module model: the current model
        """
        loss = 0
        for name, param in model.named_parameters():
            if self.include(name) and param.requires_grad:
                loss += (self.iw[name] * (param - self.old_model[name]) ** 2).sum()
        return self.alpha * loss


class DataFocused(LossBasedCL):
    """
        Data-Focused CL Abstract class

        Generic class to be inherited by LWF and similar methods

        :param str device:
        :param float alpha:
        :param float ctc_weight:
        :param int temperature:
        :param bool eval_mode:
    """
    def __init__(
          self,
          device: str,
          alpha: float,
          ctc_weight: float = 0.0,
          temperature: float = 1,
          eval_mode: bool = False
    ):
        logging.info("DataFocused with: ctc_weight = %.2f, alpha = %.2f, T = %.2f, eval_mode = %s..." % (ctc_weight, alpha, temperature, str(eval_mode)))
        self.ctc_weight = ctc_weight
        self.alpha = alpha
        self.T = 1.0 * temperature
        self.device = device
        self.eval_mode = eval_mode
        self.old_model = None

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        """
          Sets the old model

          :param torch.nn.Module model: the old model
        """
        self.old_model = copy_model(model, self.device, train=not self.eval_mode)

    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: typing.Dict = None,
            decoder_output: torch.tensor = None,
            ctc_output: torch.tensor = None
    ):
        """
          :param torch.nn.Module model:
          :param torch.tensor inputs: input for the encoder (xs_pad, src_mask, -1, True)
          :param torch.tensor decoder_output: current output of the decoder
          :param torch.tensor ctc_output: current output of CTC
        """
        pass



#####################################################################################################################
########################### REHEARSAL-BASED CL METHODS ##############################################################
#####################################################################################################################

"""
    Rehearsal-Based CL methods: overview
        -- ER: "trains jointly on mini-batch from current task and one from memory"
        -- A_GEM: "gradient-based, avoids interference between gradient from new task and gradient from memory-batch"
        -- GEM: "non-simplified and original version of A-GEM"
        -- KD: "teacher-student method with the old model the teacher and current model the student, using data from memory" 

"""



class ER(LossBasedCL, Rehearsal):
    """
        Experience Replay extension for ASR model.

        :param str device: the device for the model and data
        :param float alpha: the weight of the regularization
    """
    def __init__(
            self,
            device: str,
            alpha: float = 1,
    ):
        super(ER, self).__init__(device=device)
        self.alpha = alpha
        logging.info("CL Method: ER with alpha=%.2f" % (alpha))

    def compute_loss(
            self,
            model: torch.nn.Module
    ):
        """
            Computes the ER loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch
        batch = self.get_batch()
        # compute loss on new model
        loss = model.forward_er(**batch)
        # multiply by regularization weight
        return self.alpha * loss


class A_GEM(PartialCLMethod, GradBasedCL, Rehearsal):
    """
        Average-Gradient Episodic Memory

        :param str device: the device of the model and data
        :param List[str] exclude: list of layers to exclude from regularization
        :param float eps: determines the angle g, g_ref make after correction - if eps == 0, their angle is 90 degrees

    """
    def __init__(
            self,
            device: str,
            exclude: typing.List[str] = None,
            eps: float = 0.0
    ):
        super(A_GEM, self).__init__(device=device)
        self.include = lambda name: exclude is None or not name in exclude
        self.eps = eps
        self.param_sizes = None

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        """
            Sets param sizes, which contains information
            regarding the size and number of element in each layer

            :param torch.nn.Module model: the old model
        """
        self.param_sizes = {n: {'numel': p.numel(), 'size': p.size()}
                            for n, p in model.named_parameters() if self.include(n) and p.requires_grad}

    @staticmethod
    def to_param(
            x: torch.tensor,
            sizes: typing.Dict):
        """
            Turns a vector into a state_dict
            :param torch.tensor x: the tensor as 1D vector
            :param dict sizes: dict with key the layer and value a dict with p.numel() and p.size()

        """
        sd, k = {}, 0
        for n, p in sizes.items():
            sd[n] = x[k:k + p['numel']].view(p['size'])
            k += p['numel']
        return sd

    @staticmethod
    def to_vec(
            param: typing.Union[typing.List, typing.Dict, typing.Tuple]
    ):
        """
            Transforms a dictionary or list of tensors into 1D tensor
            :param dict or list param: dict or list of tensors
        """
        if isinstance(param, dict):
            return torch.cat([p.view(-1) for n, p in param.items()])
        elif isinstance(param, list) or isinstance(param, tuple):
            return torch.cat([p.reshape(-1) for p in param])
        else:
            raise Exception("expected dictionary, list or tuple")

    def grad_fn(
            self,
            model: torch.nn.Module,
    ):
        """
            Returns an updated gradient
            :param nn.Module model: the current model
        """
        # collect model's gradient in vector
        g = self.to_vec({name: 1.0 * param.grad.data.detach()
                         for name, param in model.named_parameters() if self.include(name) and param.requires_grad})
        # reset model's gradient
        model.zero_grad()
        # sample batch from memory
        xs = self.get_batch()
        x = _recursive_to(xs, self.device)
        # compute the loss
        loss = model.forward_er(**x)
        # compute the gradients
        loss.backward()
        # collect memory batch's gradient
        g_ref = self.to_vec({name: 1.0 * param.grad.data.detach()
                             for name, param in model.named_parameters() if self.include(name) and param.requires_grad})
        # check if g and g_ref interfere
        if torch.dot(g, g_ref) < 0:
            # compute new gradient if they interfere: g = g - g'gr / (gr'gr) * gr
            g_tilde = g - (torch.dot(g, g_ref) - self.eps) / torch.dot(g_ref, g_ref) * g_ref
            logging.info("Updating gradient in A-GEM...")
        else:
            # no interference: g_tilde is original gradient
            g_tilde = 1.0 * g
        # turn gradient into state_dict
        grad = self.to_param(g_tilde, self.param_sizes)
        # set the model's gradients
        for name, param in model.named_parameters():
            if self.include(name) and param.requires_grad:
                param.grad = grad[name]


class GEM(PartialCLMethod, GradBasedCL, Rehearsal):
    """
        Gradient Episodic Memory

        :param str device: the device of the model and data
        :param List[str] exclude: list of layers to exclude from regularization
        :param float eps: determines the angle g, g_ref make after correction - if eps == 0, their angle is 90 degrees

    """
    def __init__(
            self,
            device: str,
            exclude: typing.List[str] = None,
            eps: float = 0.0
    ):
        super(GEM, self).__init__(device)
        self.include = lambda name: exclude is None or not name in exclude
        self.eps = eps
        self.param_sizes = None

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        """
            Sets param sizes, which contains information
            regarding the size and number of element in each layer

            :param torch.nn.Module model: the old model
        """
        self.param_sizes = {n: {'numel': p.numel(), 'size': p.size()}
                            for n, p in model.named_parameters() if self.include(n) and p.requires_grad}

    @staticmethod
    def to_param(
            x: torch.tensor,
            sizes: typing.Dict):
        """
            Turns a vector into a state_dict
            :param torch.tensor x: the tensor as 1D vector
            :param dict sizes: dict with key the layer and value a dict with p.numel() and p.size()

        """
        sd, k = {}, 0
        for n, p in sizes.items():
            sd[n] = x[k:k + p['numel']].view(p['size'])
            k += p['numel']
        return sd

    @staticmethod
    def to_vec(
            param: typing.Union[typing.List, typing.Dict, typing.Tuple]
    ):
        """
            Transforms a dictionary or list of tensors into 1D tensor
            :param dict or list param: dict or list of tensors
        """
        if isinstance(param, dict):
            return torch.cat([p.view(-1) for n, p in param.items()])
        elif isinstance(param, list) or isinstance(param, tuple):
            return torch.cat([p.reshape(-1) for p in param])
        else:
            raise Exception("expected dictionary, list or tuple")

    @staticmethod
    def solve_dual_problem(
            g: torch.tensor,
            g_ref: torch.tensor,
    ):
        """
            Solves the dual problem of GEM
                min 0.5*v^2*g_ref'*g_ref + g'*g_ref v
                s.t. v >= 0
            assuming that v is a scalar.
            The solution to this problem is v = max(0, -g'*g_ref / (g_ref'*g_ref))
            The new gradient is then:
                g <- v g_ref + g

            :param torch.tensor g: gradient of the current task
            :param torch.tensor g_ref: gradient of the memory
        """
        # solution to the minimization problem
        v = -torch.dot(g, g_ref) / torch.dot(g_ref, g_ref)
        # constraint: v >= 0
        v = max(0., v)
        logging.info("--> GEM solution: v = %.4f" % v)
        # solution: g <- v g_ref + g
        return v * g_ref + g

    def grad_fn(
            self,
            model: torch.nn.Module,
    ):
        """
            Returns an updated gradient
            :param nn.Module model: the current model
        """
        # collect model's gradient in vector
        g = self.to_vec({name: 1.0 * param.grad.data.detach()
                         for name, param in model.named_parameters() if self.include(name) and param.requires_grad})
        # reset model's gradient
        model.zero_grad()
        # sample batch from memory
        xs = self.get_batch()
        x = _recursive_to(xs, self.device)
        # compute the loss
        loss = model.forward_er(**x)
        # compute the gradients
        loss.backward()
        # collect memory batch's gradient
        g_ref = self.to_vec({name: 1.0 * param.grad.data.detach()
                             for name, param in model.named_parameters() if self.include(name) and param.requires_grad})
        g_tilde = self.solve_dual_problem(g, g_ref)
        # turn gradient into state_dict
        grad = self.to_param(g_tilde, self.param_sizes)
        # set the model's gradients
        for name, param in model.named_parameters():
            if self.include(name) and param.requires_grad:
                param.grad = grad[name]


class KD(KnowledgeDistillationBased):
    """
        Knowledge Distillation with memory for Continual Learning

        :param str device: the device for the model
        :param float alpha: the weight of the regularization
        :param float ctc_weight: the weight of CTC in the KL div loss
        :param float temperature: temperature of the KL div loss
        :param bool eval_mode: if True, old model (which generates logits) is in eval mode
    """

    def __init__(
            self,
            device: str,
            alpha: float = 1.0,
            ctc_weight: float = 0.3,
            temperature: float = 1,
            eval_mode: bool = True,
    ):
        super(KD, self).__init__(device=device, alpha=alpha, ctc_weight=ctc_weight, temperature=temperature, eval_mode=eval_mode)
        logging.info("KD with alpha = %.2f, temperature = %.1f, eval_mode = %s" % (alpha, temperature, str(eval_mode)))

    def compute_loss(
            self,
            model: torch.nn.Module
    ):
        """
            Computes the KD loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch from memory
        x, names = self.get_batch(return_names=True)
        # model should return output, not loss
        x['return_args'] = ['decoder_out', 'ctc_out']
        # obtain output from current and old model
        _, dec_output, ctc_output = model.forward_er(**x)
        # if outputs have already been computed, retrieve them:
        if hasattr(self, 'old_outputs'):
            # return the unique key
            key = self.get_unique_key(names)
            try:
                # return the old dec and ctc outputs, computed previously
                old_dec_output, old_ctc_output = (self.old_outputs[key]['dec'].to(self.device),
                                                  self.old_outputs[key]['ctc'].to(self.device))
            except KeyError as ke:
                logging.info(f"Could not load old outputs: {ke}")
                _, old_dec_output, old_ctc_output = self.old_model.forward_er(**x)
                self.old_outputs[key] = {'ctc': old_ctc_output.detach().cpu(), 'dec': old_dec_output.detach().cpu()}
                logging.info(f"Added to old_outputs.. contains {len(self.old_outputs)} elements..")
        # otherwise compute them now
        else:
            #logging.info("No old_outputs: computing output now")
            _, old_dec_output, old_ctc_output = self.old_model.forward_er(**x)
        # compute KL div loss at decoder-level
        loss = self.distillation_loss(old_dec_output / self.T, dec_output / self.T)
        # optionally, compute KL div loss at CTC-level
        if self.ctc_weight > 0:
            loss = (self.ctc_weight * self.distillation_loss(old_ctc_output / self.T, ctc_output / self.T)
                    + (1 - self.ctc_weight) * loss)
        # multiply KD loss by regularization weight
        return self.alpha * loss


class SVR(KD, SVDBasedCL):
    def __init__(
            self,
            device: str,
            outdir: str,
            old_init_param: str,
            alpha: float = 1.0,
            ctc_weight: float = 0.3,
            temperature: float = 1,
            eval_mode: bool = True,
            transfer_from_old_params: bool = False,

    ):
        KD.__init__(self, device=device, alpha=alpha, ctc_weight=ctc_weight, temperature=temperature,
                                 eval_mode=eval_mode)
        SVDBasedCL.__init__(self, outdir=outdir, init_param=old_init_param, alpha=alpha)
        # SKD must only have KD loss
        self.new_model = None
        self.transfer_from_old_params = transfer_from_old_params

    def set_model(
            self,
            model: torch.nn.Module,
            optimizers: typing.List[torch.optim.Optimizer],
    ):
        """
            Sets the old model

            :param torch.nn.Module model: the old model
        """
        # freeze everything
        for name, param in model.named_parameters():
            param.requires_grad = False
        # set model
        SVDBasedCL.set_model(self, model, optimizers=optimizers)
        self._old_model = copy_model(model, self.device, train=not self.eval_mode)
        #self.new_model = copy_model(model, self.device, train=not self.eval_mode)
        replaced_layers = set()
        # linear layers in old model must have: alpha = 0
        no_svdlinear_layers = True
        for name, module in self._old_model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and hasattr(module, 'alpha'):
                # alpha_init: such that sigmoid(gamma * M  @ alpha) = 0.0
                y = torch.full((module.alpha.size(0),), -10.0  / module.gamma, device=module.alpha.device)  # shape: (d,)
                alpha_init = y
                module.alpha.data.copy_(alpha_init)
                no_svdlinear_layers = False
                # logging: to check if works properly
                actual_alpha = module.sigmoid(module.alpha)
                logging.info(f"OLD MODEL > {name} = {actual_alpha}")
                replaced_layers.add(name)
        assert not no_svdlinear_layers
        # check for current model
        for name, module, in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'alpha'):
                # logging: to check if works properly
                actual_alpha = module.sigmoid(module.alpha)
                logging.info(f"CURRENT MODEL > {name} = {actual_alpha}")
        if self.transfer_from_old_params:
            # Update parameters for layers that were **not** replaced
            for name, param in self._old_model.named_parameters():
                # Extract the module name by removing the parameter suffix
                module_name = ".".join(name.split(".")[:-1])
                # If the layer was not replaced, apply averaging
                if module_name not in replaced_layers:
                    old_param = self.old_model[name]
                    param.data.copy_(old_param.to(param.device))
                    logging.info(f"Using old params {name}...")
                else:
                    logging.info(f"NOT using old params {name}...")
        self.old_model = self._old_model

    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the KD loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch from memory
        x, names = self.get_batch(return_names=True)
        # compute er-loss and kd-loss
        x['return_args'] = ['decoder_out', 'ctc_out']
        # obtain output from current and old model
        er_loss, dec_output, ctc_output = model.forward_er(**x)
        # if outputs have already been computed, retrieve them:
        if hasattr(self, 'old_outputs'):
            # return the unique key
            key = self.get_unique_key(names)
            # return the old dec and ctc outputs, computed previously
            old_dec_output, old_ctc_output = (self.old_outputs[key]['dec'].to(self.device),
                                              self.old_outputs[key]['ctc'].to(self.device))
        # otherwise compute them now
        else:
            _, old_dec_output, old_ctc_output = self.old_model.forward_er(**x)
        # compute KL div loss at decoder-level
        kd_loss = self.distillation_loss(old_dec_output / self.T, dec_output / self.T)
        # optionally, compute KL div loss at CTC-level
        if self.ctc_weight > 0:
            kd_loss = (self.ctc_weight * self.distillation_loss(old_ctc_output / self.T, ctc_output / self.T)
                    + (1 - self.ctc_weight) * kd_loss)
        # add a L2 regularization loss if necessary
        reh_loss = self.alpha * (kd_loss + er_loss) / 2
        stats = {"rehearsal-cl-loss": item(reh_loss)}
        return reh_loss, stats


#####################################################################################################################
########################### REGULARIZATION-BASED CL METHODS #########################################################
#####################################################################################################################


"""
    Regularization-based CL methods: overview
        -- EWC: "adds L2 regularization loss, where loss for each parameters is weighted by importance weight value"
        -- L2: "simple L2 regularization loss, thus same as EWC where importance weights are all equal"

"""

class EWC(Regularization):
    """
        Elastic Weight Consolidation (EWC)

        :param str device
        :param str init_param: output directory of (previous) model (to load importance weights)
        :param float alpha: weight of regularization
        :param str name: name of file with importance weights
        :param List[str] exclude: list of layers which are excluded from regularization
    """
    def __init__(
            self,
            device: str,
            init_param: str,
            alpha: float,
            name: str = 'importance_weight.ewc',
            exclude: typing.List[str] = None
    ):
        super(EWC, self).__init__(device, init_param, alpha, name, exclude)
        logging.info("CL Method: EWC with alpha = %d" % alpha)


class MAS(Regularization):
    """
        Memory-Aware Synapses (MAS)

        :param str device
        :param str init_param: output directory of (previous) model (to load importance weights)
        :param float alpha: weight of regularization
        :param str name: name of file with importance weights
        :param List[str] exclude: list of layers which are excluded from regularization
    """
    def __init__(
            self,
            device: str,
            init_param: str,
            alpha: float,
            name: str = 'importance_weight.mas',
            exclude: typing.List[str] = None
    ):
        super(MAS, self).__init__(device, init_param, alpha, name, exclude)
        logging.info("CL Method: MAS with alpha = %d" % alpha)


#####################################################################################################################
########################### DATA-FOCUSED CL METHODS #################################################################
#####################################################################################################################


"""
    Data-focused CL methods:
        -- LWF: "uses new task's data to transfer knowledge from old model (teacher) to current model (student)"
"""


class LWF(DataFocused):
    """
        Learning without Forgetting for Encoder-Decoder ASR model
        :param float alpha: weight of the regularization
        :param float ctc_weight: weight for CTC
    """
    def __init__(
            self,
            device: str,
            alpha: float,
            ctc_weight: float = 0.0,
            temperature: float = 1,
            eval_mode: bool = False,
            L2_weight: float = 0.0,
    ):
        super(LWF, self).__init__(device, alpha, ctc_weight, temperature, eval_mode)
        self.l2_weight = L2_weight

    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: typing.Dict = None,
            decoder_output: torch.tensor = None,
            ctc_output: torch.tensor = None
    ):
        """
          Compute the loss term of the LWF regularizer
          :param torch.nn.Module model:
          :param dict inputs: contains keys [speech, speech_lengths, text, text_lengths, task]
          :param torch.tensor decoder_output: current output of the decoder
          :param torch.tensor ctc_output: current output of CTC
        """
        inputs["return_args"] = ["ctc_out", "decoder_out"]
        # send inputs through old_model
        _, old_ctc_output, old_decoder_output = self.old_model.forward_er(**inputs)
        # compute Decoder loss term
        loss = distillation_loss(old_decoder_output / self.T, decoder_output / self.T)
        # compute CTC loss term
        if self.ctc_weight > 0.0:
            loss = (self.ctc_weight * distillation_loss(old_ctc_output / self.T, ctc_output / self.T)
                  + (1 - self.ctc_weight) * loss)
        return self.alpha * loss

