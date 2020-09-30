from tqdm import tqdm as tqdm
import numpy as np

import torch
import torch.nn as nn
from GPyOpt.methods import BayesianOptimization

import minebed.mine as mm
import minebed.methods as methods
import minebed.lower_bounds as lower_bounds


class BED:
    """
    Parent BED class that defines the skeleton for child classes.

    Attributes
    ----------
    model: torch.nn.Module (or child class) object
        A parametrised neural network that is trained to compute a lower bound
        on the mutual information between two random variables.
    optimizer: torch.optim object
        The optimiser used to learn the neural network parameters.
    scheduler: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    simulator: function
        Simulator function for the implicit simulator model that is studied.
    prior: np.ndarray
        Samples from the prior distribution used to simulate synthetic data.
    LB_type: str
        The type of mutual information lower bound that is maximised.

    Methods
    -------
    train:
        Runs the main Bayesian experimental design procedure to find the
        optimal design.
    save:
        Saves some essential data.
    """

    def __init__(
            self, model, optimizer, scheduler,
            simulator, prior, LB_type='NWJ'):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.simulator = simulator
        self.prior = prior
        self.LB_type = LB_type

    def train(self):
        pass

    def save(self, filename):
        pass


class GradientFreeBED(BED):
    """
    Performs mutual information neural estimation for Bayesian experimental
    design (MINEBED) for an implicit model where we cannot compute gradients of
    the sampling path with respect to designs. It uses Bayesian optimisation to
    build a GP model of a mutual information lower bound as a function of
    designs in order to find the optimal experimental design.

    Attributes
    ----------
    model: torch.nn.Module (or child class) object
        A parametrised neural network that is trained to compute a lower bound
        on the mutual information between two random variables.
    optimizer: torch.optim object
        The optimiser used to learn the neural network parameters.
    scheduler: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    simulator: function
        Simulator function for the implicit simulator model that is studied.
    prior: np.ndarray
        Samples from the prior distribution used to simulate synthetic data.
    train_curves: list
        List of MINE training curves for each BO evaluation.
    model_states: list
        Final MINE model states after each BO evaluation.
    mine_obj: minebed.mine.MINE object
        The MINE object obtained from the last BO evaluation.
    bo_obj: GPyOpt object
        The GPyOpt object obtained from the last BO evaluation.
    mine_obj_final: minebed.mine.MINE object
        The MINE object obtained by training the model at the optimal design.
    LB_type: str
        The type of mutual information lower bound that is maximised.
    n_epoch: int
        The number of epochs used to train a model during a BO evaluation.
    batch_size: int
        The batch size used to train a model during a BO evaluation.
    ma_window: int
        The moving average window size used to obtain a less noisy estimate of
        the loss function after training a model.
    domain: list of dictionaries
        Specifications of the domain of each design variable.
    constraints: list of dictionaries or None
        Specifications of the constraints of the design domain.
    save_models: boolean
        If True, saves the final state of every model trained during a BO
        evaluation.

    Methods
    -------
    train:
        Runs the main Bayesian experimental design procedure to find the
        optimal design.
    train_final_model:
        Trains a final MINE model at the optimal design.
    save:
        Saves some essential data.
    """

    def __init__(
            self, model, optimizer, scheduler, simulator, prior,
            domain, n_epoch, batch_size, ma_window=100,
            constraints=None, LB_type='NWJ', save_models=False):
        """
        Parameters
        ----------
        model: torch.nn.Module (or child class) object
            A parametrised neural network that is trained to compute a lower
            bound on the mutual information between two random variables.
        optimizer: torch.optim object
            The optimiser used to learn the neural network parameters.
        scheduler: torch.optim.lr_scheduler object
            The learning rate scheduler used in conjunction with the optimizer.
        simulator: function
            Simulator function for the implicit simulator model under study. It
            should take a design d and an array of prior samples as input, i.e.
            data = simulator(d, prior), where the shape of data should be
            (:, dim(y)).
        prior: np.ndarray of size (:, dim(theta))
            Samples from the prior distribution used to simulate data.
        LB_type: str
            The type of mutual information lower bound that is maximised.
            (default is 'NWJ', also known as MINE-f)
        n_epoch: int
            The number of epochs used to train a model during a BO evaluation.
        batch_size: int
            The batch size used to train a model during a BO evaluation.
        ma_window: int
            The moving average window size used to obtain a less noisy estimate
            of the loss function after training a model.
            (default is 100)
        domain: list of dictionaries
            Specifications of the domain of each design variable.
        constraints: list of dictionaries or None
            Specifications of the constraints of the design domain.
            (default is None)
        save_models: boolean
            If True, saves the final state of every model trained during a BO
            evaluation.
            (default is True)
        """

        super(GradientFreeBED, self).__init__(
            model, optimizer, scheduler, simulator, prior, LB_type)

        self.domain = domain
        self.constraints = constraints

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.ma_window = ma_window

        # save initial states, for restarts during BO
        self.init_opt_state = self.optimizer.state_dict()
        self.init_sched_state = self.scheduler.state_dict()

        self.save_models = save_models
        self.model_states = []
        self.train_curves = []

    def _reset_weights(self, m):
        """Resets the parameters of model m."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def _reset_model(self):
        """Reset the parameters of the current model."""
        self.model.apply(self._reset_weights)

    def _reset_optimizer(self):
        """Reset the optimizer to its initial state."""
        self.optimizer.load_state_dict(self.init_opt_state)

    def _reset_scheduler(self):
        """Reset the scheduler to its initial state."""
        self.scheduler.load_state_dict(self.init_sched_state)

    def _store_data(self, lb):
        """Stores some relevant data during training process."""
        if self.save_models:
            self.model_states.append(self.model.state_dict())
        self.train_curves.append(lb)

    def _compute_optimal_design(self, obj):
        """Computes the optimal design after training a GP model."""
        self.d_opt = methods.get_GP_optimum(obj)

    def _objective(self, d):
        """Objective function to be maximised during Bayesian Optimisation."""

        # reset model, optimizer and scheduler
        self._reset_model()
        self._reset_optimizer()
        self._reset_scheduler()

        # simulate data
        Y = self.simulator(d, self.prior)
        train_data = (self.prior, Y)

        # initialize MINE object and optimizer, scheduler
        self.mine_obj = mm.MINE(self.model, train_data, LB_type=self.LB_type)
        self.mine_obj.set_optimizer(self.optimizer)
        self.mine_obj.set_scheduler(self.scheduler)

        # train MINE model
        self.mine_obj.train(self.n_epoch, self.batch_size, bar=False)

        # compute the moving average of all evaluations
        lb_ma = self.mine_obj._ma(
            self.mine_obj.train_lb,
            window=self.ma_window)

        # compute the last moving average
        lb_final_ma = lb_ma[-1]

        # store data for analysis
        self._store_data(lb_ma)

        return lb_final_ma

    def train(
            self, bo_model=None, bo_space=None, bo_acquisition=None,
            X_init=None, Y_init=None, BO_init_num=5, BO_max_num=20,
            verbosity=False):
        """
        Uses Bayesian optimisation to find the optimal design. The objective
        function is the mutual information lower bound at a particular design,
        obtained by training a MINE model.

        Parameters
        ----------
        bo_model:

        bo_space:

        bo_acquisition:

        BO_init_num: int
            The number of initial BO evaluations used to initialise the GP.
            (default is 5)
        BO_max_num: int
            The maximum number of BO evaluations after the initialisation.
            (default is 20)
        verbosity: boolean
            Turn off/on output to the command line.
            (default is False)
        """

        if verbosity:
            print('Initialize Probabilistic Model')

        if bo_model and bo_space and bo_acquisition:
            raise NotImplementedError('Custom BO model not yet implemented.')
        elif all(v is None for v in [bo_model, bo_space, bo_acquisition]):
            pass
        else:
            raise ValueError(
                'Either all BO arguments or none need to be specified.')

        # Define GPyOpt Bayesian Optimization object
        self.bo_obj = BayesianOptimization(
            f=self._objective, domain=self.domain,
            constraints=self.constraints, model_type='GP',
            acquisition_type='EI', normalize_Y=False,
            initial_design_numdata=BO_init_num, acquisition_jitter=0.01,
            maximize=True, X=X_init, Y=Y_init)
        # TODO: Implement a more modular approach with GPy model as input.

        if verbosity:
            print('Start Bayesian Optimisation')

        # run the bayesian optimisation
        self.bo_obj.run_optimization(
            max_iter=BO_max_num, verbosity=verbosity, eps=1e-5)

        # find optimal design from posterior GP model; stored as d_opt
        self._compute_optimal_design(self.bo_obj)

    def train_final_model(self, n_epoch=None, batch_size=None):
        """
        Train a final MINE model at the optimal design.

        Parameters
        ----------
        n_epoch: int
           The number of epochs to be used during training. If nothing is
           specified, the default self.n_epoch is used.
        batch_size: int
           The batch size to be used during training. If nothing is specified,
           the default self.batch_size is used.
        """

        if n_epoch is None:
            n_epoch = self.n_epoch
        if batch_size is None:
            batch_size = self.batch_size

        # reset model, optimizer and scheduler
        self._reset_model()
        self._reset_optimizer()
        self._reset_scheduler()

        # simulate data
        Y = self.simulator(self.d_opt, self.prior)
        train_data = (self.prior, Y)

        # initialize MINE object and optimizer, scheduler
        self.mine_obj_final = mm.MINE(
            self.model, train_data, LB_type=self.LB_type)
        self.mine_obj_final.set_optimizer(self.optimizer)
        self.mine_obj_final.set_scheduler(self.scheduler)

        # train MINE model
        self.mine_obj_final.train(n_epoch, batch_size, bar=False)

        # compute the moving average of all evaluations
        lb_ma = self.mine_obj_final._ma(
            self.mine_obj_final.train_lb,
            window=self.ma_window)

        # store data for analysis
        self._store_data(lb_ma)

    def save(self, filename, extra_data={}):
        """
        Saves some essential data, such as the final model, optimizer and
        scheduler state, the model states and training curves for every BO
        evaluation, the BO evaluations and objective scores, the optimal
        design, and the parameters of the GP model.

        Parameters
        ----------
        filename: str
            Location and name of file to be saved.
        extra_data: dict
            Dictionary of extra files that should be saved.
            (default is an empty dict {})
        """

        # collect internal data for saving
        internal_data = {
            'model_final': self.model.state_dict(),
            'optimizer_final': self.optimizer.state_dict(),
            'scheduler_final': self.scheduler.state_dict(),
            'models_training': self.model_states,
            'training_curves': self.train_curves,
            'BO_X_eval': self.bo_obj.X,
            'BO_Y_eval': self.bo_obj.Y,
            'd_opt': self.d_opt,
            'GP_params': self.bo_obj.model.model.param_array}

        # add external data
        data = dict(internal_data, **extra_data)

        # save as a binary file
        torch.save(data, '{}.pt'.format(filename))

class GradientBasedBED(BED):

    """
    Performs mutual information neural estimation for Bayesian experimental
    design (MINEBED) for an implicit model where we can compute or approximate
    gradients of the sampling path with respect to designs. It uses SGD on the
    neural network parameters and on the experimental design to find the
    optimal design.

    Attributes
    ----------
    model: torch.nn.Module (or child class) object
        A parametrised neural network that is trained to compute a lower bound
        on the mutual information between two random variables.
    optimizer: torch.optim object
        The optimiser used to learn the neural network parameters.
    optimizer_design: torch.optim object
        The optimiser used to learn the experimental designs.
    scheduler: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    scheduler_design: torch.optim.lr_scheduler object
        The learning rate scheduler used in conjunction with the optimizer.
    simulator: function
        Simulator function for the implicit simulator model that is studied.
    prior: np.ndarray
        Samples from the prior distribution used to simulate synthetic data.
    LB_type: str
        The type of mutual information lower bound that is maximised.
    n_epoch: int
        The number of epochs used to train the model.
    batch_size: int
        The batch size used to train the model.
    design_bounds: list
        List of lower and upper bound for all design dimensions.

    Methods
    -------
    train:
        Runs the main Bayesian experimental design procedure to find the
        optimal design.
    save:
        Saves some essential data.
    """

    def __init__(
            self, model, optimizer, optimizer_design, scheduler,
            scheduler_design, simulator, prior, n_epoch, batch_size,
            design_bounds, LB_type='NWJ', device=torch.device('cpu')):
        """
        Parameters
        ----------
        model: torch.nn.Module (or child class) object
            A parametrised neural network that is trained to compute a lower
            bound on the mutual information between two random variables.
        optimizer: torch.optim object
            The optimiser used to learn the neural network parameters.
        optimizer_design: torch.optim object
            The optimiser used to learn the experimental designs.
        scheduler: torch.optim.lr_scheduler object
            The learning rate scheduler used in conjunction with the optimizer.
        scheduler_design: torch.optim.lr_scheduler object
            The learning rate scheduler used in conjunction with the optimizer.
        simulator: function
            Simulator function for the implicit simulator model under study. It
            should take a design d, an array of prior samples and the relevant
            torch.deviceas input, i.e. data = simulator(d, prior, device),
            where the shape of data should be (:, dim(y)).
        prior: np.ndarray of size (:, dim(theta))
            Samples from the prior distribution used to simulate data.
        LB_type: str
            The type of mutual information lower bound that is maximised.
            (default is 'NWJ', also known as MINE-f)
        n_epoch: int
            The number of epochs used to train a model during a BO evaluation.
        batch_size: int
            The batch size used to train a model during a BO evaluation.
        design_bounds: list
            List of lower and upper bound for all design dimensions, of the
            form [lower, upper].
        device: torch.device
            Device on which to run operations.
            (default is on a CPU)

        Note that the simulator function needs to return (data, gradients),
        where the gradients of the data are with respect to experimental
        designs. Ways of approximating these gradients rather than providing
        them are currently not implemented.
        """

        super(GradientBasedBED, self).__init__(
            model, optimizer, scheduler, simulator, prior, LB_type)

        # optimizers
        self.optimizer_design = optimizer_design
        self.scheduler_design = scheduler_design

        # nn parameters
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.data_size = len(self.prior)

        # bounds of the design variable
        self.design_bounds = design_bounds

        # device to run computations on
        self.device = device

        if LB_type == 'NWJ':
            self.loss_function = lower_bounds.minef_loss
            self.grad_function = lower_bounds.minef_gradients
        else:
            raise NotImplementedError

    def _get_batch(self, x, y, ygrads, method='random'):

        # compute indeces
        idx = np.random.choice(
            range(len(x)), size=self.batch_size, replace=False)

        return x[idx], y[idx], ygrads[idx]

    def train(self, d):
        """
        Uses SGD to train a neural network to find the optimal design. The loss
        function is a parametrised lower bound and depends on both the network
        parameters and the designs..

        Parameters
        ----------
        d: torch.tensor
        BO_max_num: int
            The maximum number of BO evaluations after the initialisation.
            (default is 20)
        verbosity: boolean
            Turn off/on output to the command line.
            (default is False)
        """
        
        # put design onto the device
        d.to(self.device)

        # prepare Tensor containers to save data of designs
        self.designs = torch.empty(
            (self.n_epoch + 1, d.shape[0], 1),
            dtype=torch.float, device=self.device)
        self.designs[0] = d.clone().detach()  # save first design to container

        # Prepare Tensor containers to save data of MI + init counter
        self.mutual = torch.empty(
            self.n_epoch,
            dtype=torch.float, device=self.device)
        entry = 0

        for epoch in tqdm(range(self.n_epoch), leave=True, disable=False):

            # simulate data in full
            y_full, ygrads_full = self.simulator(d, self.prior, self.device)

            for b in range(int(self.data_size / self.batch_size)):

                # get batch samples
                xs, ys, ygrads = self._get_batch(
                    self.prior, y_full, ygrads_full)

                # compute loss
                loss = self.loss_function(xs, ys, self.model, self.device)

                # Zero grad the NN optimizer
                self.optimizer.zero_grad()

                # Back-Propagation
                loss.backward()

                # Perform opt steps for NN
                self.optimizer.step()  # NN params

                # save a few things to lists
                self.mutual[entry] = -loss.clone().detach()
                entry += 1

            # Perform opt steps for Design
            self.optimizer_design.zero_grad()
            d.grad = self.grad_function(
                self.prior, y_full, ygrads_full, self.model, self.device)
            self.optimizer_design.step()  # Designs

            # LR scheduler step for psi and designs
            self.scheduler.step()
            self.scheduler_design.step()

            # Clamp design if beyond boundaries
            with torch.no_grad():
                d.clamp_(self.design_bounds[0], self.design_bounds[1])

            # Save designs to list
            self.designs[epoch + 1] = d.clone().detach()

        # Move containers to CPU and convert to Numpy Arrays
        self.mutual = self.mutual.cpu().data.numpy()
        self.designs = self.designs.cpu().data.numpy()

    def save(self, filename, extra_data={}):
        """
        Saves some essential data, such as the final model state, optimizer and
        scheduler state, the training loss and the experimental designs.

        Parameters
        ----------
        filename: str
            Location and name of file to be saved.
        extra_data: dict
            Dictionary of extra files that should be saved.
            (default is an empty dict {})
        """

        # put together internal data
        internal_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_psi_state_dict': self.optimizer.state_dict(),
            'optimizer_design_state_dict': self.optimizer_design.state_dict(),
            'scheduler_psi_state_dict': self.scheduler.state_dict(),
            'scheduler_design_state_dict': self.scheduler_design.state_dict(),
            'batch_size': self.batch_size,
            'prior_samples': self.prior,
            'mi': self.mutual,
            'designs': self.designs}

        # add external data
        data = dict(internal_data, **extra_data)

        # save as a binary file
        torch.save(data, '{}.pt'.format(filename))
