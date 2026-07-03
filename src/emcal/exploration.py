"""ExplorationBias: the exploration-bias parameter (alpha) and its update schedules
(constant, decay, Boyle, Jasrasaria) for the acquisition function.
"""
import numpy as np
import warnings
from enum import Enum
from .enums import EpSchedule


class ExplorationBias:
    """
    Base class for methods of calculating explroation bias at each bo iter

    Methods
    -------
    __init__(*): Constructor method
    __bound_ep(ep_val): Bounds the value of a given exploration parameter between the minimum and maximum value
    update(): Updates value of exploration parameter based on one of the four alpha heuristics
    __set_ep_constant(): Creates a value for the exploration parameter based off of a constant value
    __set_ep_decay(): Creates a value for the exploration parameter based off of a decay heuristic
    __set_ep_boyle(): Creates a value for the exploration parameter based off of a Boyle heuristic
    __set_ep_jasrasaria(): Creates a value for the exploration parameter based off of a Jasrasaria heuristic
    """

    def __init__(
        self,
        ep0,
        ep_curr,
        ep_enum,
        bo_iter,
        bo_iter_max,
        ep_inc,
        ep_f,
        improvement,
        best_error,
        mean_of_var,
    ):
        """
        Parameters
        ----------
        ep0: float
            The original exploration parameter value
        ep_curr: float
            The current exploration parameter value
        ep_enum: Enum
            Whether Boyle, Jasrasaria, Constant, or Decay ep method will be used
        bo_iter: int
            The number of the current BO iteration
        bo_iter_max: int
            The maximum number of BO iterations
        e_inc: float
            The increment for the Boyle's method for calculating exploration parameter: Recommendation is 1.5
        ep_f: float
            The final exploration parameter value: Recommendation is 0
        improvement: bool
            Determines whether last objective was an improvement
        best_error: float
            The lowest error objective value in the training data
        mean_of_var: float
            The value of the average of all posterior variances

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        ------
        For all methods, ep is on domain [0.5, 2] inclusive
        """
        assert all(
            (isinstance(param, (float, int)) or param is None)
            for param in [ep0, ep_curr, ep_inc, ep_f, best_error, mean_of_var]
        ), "ep0, ep_curr, ep_inc, ep_f, best_error, and mean_of_var must be int, float, or None"
        assert (
            isinstance(ep_enum, Enum) == True
        ), "ep_enum must be an Enum instance of Class EpSchedule"
        assert (
            isinstance(improvement, bool) == True or improvement is None
        ), "improvement must be bool or None"
        assert all(
            (isinstance(param, (int)) or param is None)
            for param in [bo_iter, bo_iter_max]
        ), "bo_iter and bo_iter_max must be int or None"
        # Constructor method
        self.ep0 = ep0
        self.ep_curr = ep_curr
        self.ep_enum = ep_enum
        self.bo_iter = bo_iter
        self.bo_iter_max = bo_iter_max
        self.ep_inc = ep_inc
        self.ep_f = ep_f
        self.improvement = improvement
        self.best_error = best_error
        self.mean_of_var = mean_of_var
        # Set ep max and min based off of mathematical bound reasoning
        self.ep_max = 2
        self.ep_min = 0.5

    def __bound_ep(self, ep_val):
        """
        Bounds the value of a given exploration parameter between the minimum and maximum value

        Parameters
        ----------
        ep_val: int or float
            The value of the exploration parameter

        Returns
        --------
        ep_val: int or float
            The value of the exploration parameter within self.ep_min and self.ep_max
        """
        assert isinstance(ep_val, (float, int)), "ep_val must be float or int!"
        if ep_val > self.ep_max:
            warnings.warn("setting ep_val to self.ep_max because it was too large")
            ep_val = self.ep_max
        elif ep_val < self.ep_min:
            warnings.warn("setting ep_val to self.ep_min because it was too small")
            ep_val = self.ep_min
        else:
            assert (
                self.ep_max >= ep_val >= self.ep_min
            ), "Starting exploration bias (ep0) must be greater than or equal to 0.5!"

        return ep_val

    def update(self):
        """
        Updates value of exploration parameter based on one of the four alpha heuristics

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        --------
        Sets the current exploration parameter self.ep_curr, but does not return anything. Use ExplorationBias.ep_curr() to return it

        """
        # Set ep0 and ep_f to the max if they are too large
        if self.ep0 is not None:
            self.ep0 = self.__bound_ep(self.ep0)
        if self.ep_f is not None:
            self.ep_f = self.__bound_ep(self.ep_f)

        if self.ep_enum.value == 1:  # Constant if using constant method
            assert self.ep0 is not None
            ep = self.__set_ep_constant()

        elif self.ep_enum.value == 2:  # Decay
            assert self.ep0 is not None
            assert self.ep_f is not None
            assert self.bo_iter_max is not None
            ep = self.__set_ep_decay()

        elif self.ep_enum.value == 3:  # Boyle
            assert self.ep0 is not None
            assert self.ep_inc is not None
            ep = self.__set_ep_boyle()

        else:  # Jasrasaria
            ep = self.__set_ep_jasrasaria()

        # Set current ep to new ep
        self.ep_curr = ep

    def __set_ep_constant(self):
        """
        Creates a value for the exploration parameter based off of a constant value

        Returns
        --------
        ep: float
            The exploration parameter for the iteration
        """
        ep = self.ep0

        return ep

    def __set_ep_decay(self):
        """
        Creates a value for the exploration parameter based off of a decay heuristic.

        Returns
        --------
        ep: float
            The exploration parameter for the iteration

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Full decay is reached by 1/2 of the maximum number of BO iters
        """
        assert self.bo_iter is not None
        assert self.bo_iter_max - 1 >= self.bo_iter >= 0

        # Set ep_f to max value if it is too big
        # Initialize number of decay steps
        decay_steps = int(self.bo_iter_max / 2)
        # Apply heuristic on 1st iteration and all steps until end of decay steps
        if self.bo_iter < decay_steps or self.bo_iter == 0:
            ep = self.ep0 + (self.ep_f - self.ep0) * (self.bo_iter / self.bo_iter_max)
        else:
            ep = self.ep_f

        return ep

    def __set_ep_boyle(self):
        """
        Creates a value for the exploration parameter based on Boyle's Heuristic for GPO bounds

        Returns
        --------
        ep: float
            The exploration parameter for the iteration

        Notes
        -----
        Based on Heuristic from Boyle, P., Gaussian Processes for regression and Optimisation
        For these parameters, ep gets normalized between 0 and 2 given a neutral value of 1 as the starting point

        References
        ----------
        Boyle, P., Gaussian Processes for regression and Optimisation, Ph.D, Victoria University of Wellington, Wellington, New Zealand, 2007
        """
        # Set ep_curr as ep0 if it is not set
        if self.ep_curr is None:
            ep = self.ep0
        else:
            # Assert that improvement is not None
            assert self.improvement is not None
            # Apply a version of Boyle's heuristic
            # In original Boyle, you want to gradually expand or shrink your bounds
            # We take this concept for ep to increase exploration when improvement is FALSE and increase it when TRUE
            if self.improvement == True:
                # If we improved last time, Decrease exploration
                ep = self.ep_curr / self.ep_inc
            else:
                # If we did not, Increase Exploration
                ep = self.ep_curr * self.ep_inc

        # Ensure that ep stays within the bounds
        ep = self.__bound_ep(ep)

        return ep

    def __set_ep_jasrasaria(self):
        """
        Creates a value for the exploration parameter based off of Jasrasaria's heuristic

        Returns
        --------
        ep: float
            The exploration parameter for the iteration

        References
        ----------
        Heuristic from Jasrasaria, D., & Pyzer-Knapp, E. O. (2018). Dynamic Control of Explore/Exploit Trade-Off In Bayesian Optimization. http://arxiv.org/abs/1807.01279
        """
        assert self.best_error is not None
        assert self.mean_of_var is not None

        # Apply Jasrasaria's Heuristic
        if self.best_error > 0:
            ep = 1 + (self.mean_of_var / self.best_error**2)
        else:
            ep = self.ep_max

        # Ensure that ep stays within the bounds
        ep = self.__bound_ep(ep)

        return ep
