from matplotlib import pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.collections import PatchCollection
import json
from matplotlib import colormaps
from collections.abc import Iterable
import string
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from matplotlib.lines import Line2D
import sklearn

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from .GPBO_Classes_New import Data, MethodName
from .analysis import *
from .analysis import is_job_like, JobContext  # explicit (robust if analysis adds __all__)
from .case_studies import get_cs_class_from_val

import warnings

np.warnings = warnings


class Plotters:
    """
    The base class for Plotting functions

    Methods
    --------------
    __init__(analyzer, save_figs = False): Constructor method
    plot_one_obj_all_methods(z_choice, log_data = False, title = None): Plots SSE, Min SSE, or EI values vs BO iter for all BO Methods at the best runs
    plot_hyperparameters(job, title = None): Plots hyperparameters vs BO Iter for all methods
    plot_parameters(job, z_choice, title = None): Plots parameter sets vs BO Iter for all methods
    __plot_2D_general(data, data_names, data_true, y_label, title, log_data): Plots 2D values of the same data type (ei, sse, min sse) on multiple subplots
    custom_format(x, pos): Custom format for 10x notation
    plot_objs_all_methods(z_choices, log_data = False, title = None): Plots EI, SSE, Min SSE, and EI values vs BO iter for all 7 methods
    __get_z_plot_names_hms(z_choice): Returns the names of the z values for the plot
    plot_hms_all_methods(z_choice, log_data = False, title = None): Plots SSE, Min SSE, or EI values vs BO iter for all BO Methods at the best runs
    plot_gp_fit(z_choice, log_data = False, title = None): Plots comparison of y_sim, GP_mean, GP_stdev, and EI at the best runs
    __scale_z_data(data, z_choice): Scales the z data based on the z choice
    __set_ylab_from_z(z_choice): Returns the y label based on the z choice
    __get_data_to_bo_iter_term(data): Returns the data up to the termination of the BO iteration
    __save_fig(save_path_to): Saves the figure to the save path
    __create_subplots(subplots_needed, sharex = False, sharey = 'none'): Creates subplots based on the number of subplots needed
    __set_subplot_details(ax, x_space, data_df_j, x_label, y_label, title): Sets the details of the subplot
    __set_plot_titles(fig, title, x_label, y_label): Sets the title and labels of the plot
    plot_parity(): Makes parity plots for all methods
    """

    # Class variables and attributes

    def __init__(self, analyzer, save_figs=False):
        """
        Parameters
        ----------
        analyzer: General_Analysis
            An instance of the General_Analysis class
        save_figs: bool, default False
            Save figures to file if True.

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Asserts
        assert isinstance(save_figs, bool), "save_figs must be boolean"
        assert isinstance(
            analyzer, (General_Analysis, All_CS_Analysis, LS_Analysis)
        ), "analyzer must be General_Analysis, LS_Analysis, or All_CS_Analysis"

        # Constructor method
        self.analyzer = analyzer
        self.save_figs = save_figs
        self.cmap = "YlOrRd_r"
        self.xbins = 5
        self.ybins = 5
        self.zbins = 900
        self.title_fntsz = 24
        self.other_fntsz = 24
        self.colors = [
            "red",
            "blue",
            "green",
            "purple",
            "darkorange",
            "deeppink",
            "teal",
        ]
        self.method_names = [
            "Conventional",
            "Log Conventional",
            "Independence",
            "Log Independence",
            "Sparse Grid",
            "Monte Carlo",
            "E[SSE]",
        ]
        self.gpbo_meth_dict = {
            "Conventional": 1,
            "Log Conventional": 2,
            "Independence": 3,
            "Log Independence": 4,
            "Sparse Grid": 5,
            "Monte Carlo": 6,
            "E[SSE]": 7,
        }

    def plot_one_obj_all_methods(self, z_choice, log_data=False, title=None):
        """
        Plots SSE, Min SSE, or acquisition function values vs BO iter for all BO Methods at the best runs

        Parameters
        -----------
        z_choice: str
            One of "min_sse", "sse", or "acq". The values that will be plotted
        log_data: bool, default False
            Plots data on natural log scale if True
        title: str or None, default None
            Title of plot

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        This function plots one objective value for all methods at the best runs. Each method is displayed in a different subplot.

        """
        # Assert Statements
        assert isinstance(z_choice, str), "z_choices must be str"
        assert z_choice in [
            "min_sse",
            "sse",
            "acq",
        ], "z_choices must be one of 'min_sse', 'sse', or 'acq'"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        assert isinstance(log_data, bool), "log_data must be boolean:"

        # Set x and y labels and save path for figure
        x_label = "Loss Evaluations"
        y_label = self.__set_ylab_from_z(z_choice) + "\n"
        save_path = self.analyzer.make_dir_name_from_criteria(
            self.analyzer.criteria_dict
        )

        # Get all jobs
        job_pointer = self.analyzer.get_jobs_from_criteria()
        # Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        # Back out best runs from job_list_best
        emph_runs = df_best["Run Number"].values
        # Initialize list of maximum bo iterations for each method
        meth_bo_max_evals = np.zeros(len(self.method_names))
        # Number of subplots is the length of the best jobs list
        subplots_needed = len(self.method_names) + 1
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=False
        )

        # Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, x_label, y_label)

        # Loop over different jobs
        for i in range(len(job_pointer)):
            # Assert job exists, if it does, great,
            # Get data
            data, data_names, data_true, sp_data, data_true_med = (
                self.analyzer.objective_trajectories(job_pointer[i], z_choice)
            )
            be, be_theta = self.analyzer.best_error(job_pointer[i])
            GPBO_method_val = sp_data["meth_name_val"]

            shrt_name = GPBOMethod(MethodName(GPBO_method_val)).report_name

            # Get Number of runs in the job
            runs_in_job = sp_data["bo_runs_in_job"]
            # Get number of iterations to add at the beginning by grabbing the length of theta_true
            cs_class = get_cs_class_from_val(sp_data["cs_name_val"])
            num_params = len(cs_class.idcs_to_consider)
            num_train_points = sp_data["num_theta_multiplier"] * num_params
            max_iters = sp_data["bo_iter_tot"] + num_train_points

            # Set subplot index to the corresponding method value number
            ax_idx = int(GPBO_method_val - 1)
            # Find the run corresponding to the best row for that job
            emph_run = df_best.loc[
                df_best["BO Method"] == shrt_name, "Run Number"
            ].iloc[0]
            ax_row, ax_col = plot_mapping[ax_idx]

            # Loop over all runs
            for j in range(runs_in_job):
                # Find job Number
                run_number = sp_data["bo_run_num"] + j
                # Create label based on run #
                label = "Run: " + str()
                # Get data until termination
                data_df_j = self.__get_data_to_bo_iter_term(data[j])
                # Define x axis
                bo_len = len(data_df_j) + num_train_points
                bo_space = np.linspace(1, bo_len, bo_len)
                # Set appropriate notation
                if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                    fmt = matplotlib.ticker.FuncFormatter(self.custom_format)
                    ax[ax_row, ax_col].yaxis.set_major_formatter(fmt)

                # Plot data
                if log_data == True:
                    data_df_j = np.log(data_df_j)

                # Add iterations to the beginning
                data_train = be[j] if be is not None else data_df_j[0]
                data_df_j_w_train = np.concatenate(
                    (np.full(num_train_points, data_train), data_df_j)
                )
                if z_choice == "min_sse":
                    # Replace values higher than data_train in data_df_j_w_train with data_train
                    data_df_j_w_train[data_df_j_w_train > data_train] = data_train

                # For result where run num list is the number of runs, print a solid line
                # print(ax_idx, emph_runs[ax_idx], run_number)
                if emph_run == run_number:
                    ax[ax_row, ax_col].plot(
                        bo_space,
                        data_df_j_w_train,
                        alpha=1,
                        color=self.colors[ax_idx],
                        drawstyle="steps",
                    )
                    ax[-1, -1].plot(
                        bo_space,
                        data_df_j_w_train,
                        alpha=1,
                        color=self.colors[ax_idx],
                        drawstyle="steps",
                    )
                else:
                    ax[ax_row, ax_col].plot(
                        bo_space,
                        data_df_j_w_train,
                        alpha=0.2,
                        color=self.colors[ax_idx],
                        linestyle="--",
                        drawstyle="steps",
                    )
                ls_xset = False
                # Plot true value if applicable
                if data_true is not None and j == data.shape[0] - 1:
                    data_ls = data_true[z_choice]
                    if isinstance(data_ls, pd.DataFrame):
                        x = data_ls["Iter"].to_numpy()
                        ls_xset = True
                        if z_choice == "min_sse":
                            y = data_ls["Min Obj Cum."].to_numpy()
                        else:
                            y = data_ls["Min Obj Act"].to_numpy()
                    else:
                        x = [1, data.shape[1]]
                        y = [list(data_true.values())[0], list(data_true.values())[0]]
                    ax[ax_row, ax_col].plot(
                        x, y, color="darkslategrey", linestyle="solid"
                    )
                    ax[-1, -1].plot(x, y, color="darkslategrey", linestyle="solid")

                # Plot median value if applicable
                ls_xset_med = False
                if data_true_med is not None and j == data.shape[0] - 1:
                    if isinstance(data_ls, pd.DataFrame):
                        ls_xset_med = True
                        data_med_ls = data_true_med[z_choice]
                        x_med = data_med_ls["Iter"].to_numpy()
                        if z_choice == "min_sse":
                            y_med = data_med_ls["Min Obj Cum."].to_numpy()
                        else:
                            y_med = data_med_ls["Min Obj Act"].to_numpy()
                        ax[ax_row, ax_col].plot(
                            x_med, y_med, color="k", linestyle="dotted"
                        )
                        ax[-1, -1].plot(x_med, y_med, color="k", linestyle="dotted")

                # Set plot details
                max_x = max_iters
                title = self.method_names[ax_idx]
                if ls_xset and ls_xset_med:
                    max_x_nls = max(max(x), max(x_med))
                    max_x = max(max_x_nls, max_iters)
                elif ls_xset:
                    max_x = max(max(x), max_iters)

                meth_bo_max_evals[ax_idx] = max_x
                x_space = np.linspace(1, max_x, max_x)

                # Concatenate the minimum and maxium value of y to data_df_j_w_train if y exists
                if ls_xset:
                    data_df_j_w_train_bnd = np.concatenate(
                        [data_df_j_w_train, [np.min(y), np.max(y)]]
                    )
                else:
                    data_df_j_w_train_bnd = data_df_j_w_train

                # print(data_df_j_w_train_bnd)

                self.__set_subplot_details(
                    ax[ax_row, ax_col],
                    x_space,
                    data_df_j_w_train_bnd,
                    None,
                    None,
                    title,
                )

            self.__set_subplot_details(
                ax[-1, -1], x_space, data_df_j_w_train_bnd, None, None, "All Methods"
            )

        # Set handles and labels and scale axis if necessary
        # Plot dummy legend
        # define an object that will be used by the legend
        class MulticolorPatch(object):
            def __init__(self, cmap, edgecolor=None, linewidth=None, ncolors=100):
                self.ncolors = ncolors
                self.edgecolor = edgecolor
                self.linewidth = linewidth

                if isinstance(cmap, str):
                    self.cmap = plt.get_cmap(cmap)
                else:
                    self.cmap = cmap

        # define a handler for the MulticolorPatch object
        class MulticolorPatchHandler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                n = orig_handle.ncolors
                width, height = handlebox.width, handlebox.height
                patches = []

                for i, c in enumerate(orig_handle.cmap(i / n) for i in range(n)):
                    patches.append(
                        plt.Rectangle(
                            [width / n * i - handlebox.xdescent, -handlebox.ydescent],
                            width / n,
                            height,
                            facecolor=c,
                            edgecolor=orig_handle.edgecolor,
                            linewidth=orig_handle.linewidth,
                        )
                    )

                patch = PatchCollection(patches, match_original=True)

                handlebox.add_artist(patch)

                if orig_handle.edgecolor is not None:
                    linestyle = (0, (5, 5))  # Dashed line style
                else:
                    linestyle = "solid"

                border_rect = plt.Rectangle(
                    [
                        handlebox.xdescent - 1,
                        -handlebox.ydescent - 1,
                    ],  # Position slightly offset
                    width + 2,  # Width slightly larger to fit around the patches
                    height + 2,  # Height slightly larger to fit around the patches
                    edgecolor="black",  # Border color
                    linestyle=linestyle,  # Dashed border style
                    linewidth=1,  # Border line width
                    fill=False,
                )  # No fill color

                # Add the border rectangle to handlebox
                handlebox.add_artist(border_rect)

                return patch

        # Add a dummy legend
        if ls_xset:
            ax[0, 0].plot(
                [], [], color="darkslategrey", linestyle="solid", label="NLS (Best)"
            )
        if ls_xset_med:
            ax[0, 0].plot([], [], color="k", linestyle="dotted", label="NLS (Median)")
        handles, labels = ax[0, 0].get_legend_handles_labels()
        handles_extra = [
            MulticolorPatch("gist_rainbow"),
            MulticolorPatch("gist_rainbow", edgecolor="white", linewidth=0.25),
        ]
        labels_extra = ["Our Methods (Best)", "Our Methods (Restarts)"]
        handles += handles_extra
        labels += labels_extra

        for k, axs in enumerate(ax.flatten()):
            if log_data == False:
                axs.set_yscale("log")

        # Display the legend
        fig.legend(
            handles,
            labels,
            loc="upper center",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(labels),
            borderaxespad=0,
            handler_map={MulticolorPatch: MulticolorPatchHandler()},
        )
        # Plots legend and title
        plt.tight_layout()

        # save or show figure
        if self.save_figs:
            save_path_dir = os.path.join(save_path, "line_plots", "all_meth_1_obj")
            save_path_to = os.path.join(save_path_dir, z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return

    def plot_hyperparameters(self, job, title=None):
        """
        Plots hyperparameters vs BO Iter for all methods

        Parameters
        -----------
        job: signac.job.Job
            The job to analyze
        title: str or None, default None
            Title of plot
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        data, data_names, data_true, sp_data = self.analyzer.hyperparameter_trajectories(job)
        y_label = "Value"
        title = "Hyperparameter Values"
        fig = self.__plot_2D_general(data, data_names, data_true, y_label, title, False)
        # save or show figure
        if self.save_figs:
            save_path_to = os.path.join(job.fn(""), "line_plots", "hyperparams")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

    def plot_parameters(self, job, z_choice, title=None):
        """
        Plots parameter sets vs BO Iter for all methods for a give z_choice (min_sse, sse, or acq)

        Parameters
        -----------
        job: signac.job.Job
            The job to analyze
        z_choice: str
            One of "min_sse", "sse", or "acq". The values that will be plotted
        title: str or None, default None
            Title of plot
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(z_choice, str), "z_choice must be a string"
        assert z_choice in [
            "min_sse",
            "sse",
            "acq",
        ], "z_choice must be one of 'min_sse', 'sse', or 'acq'"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        data, data_names, data_true, sp_data = self.analyzer.parameter_trajectories(
            job, z_choice
        )
        be, be_theta = self.analyzer.best_error(job)
        GPBO_method_val = sp_data["meth_name_val"]
        # Create label based on method #
        meth_label = self.method_names[GPBO_method_val - 1]
        y_label = "Parameter Values"

        # Get number of iterations to add at the beginning by grabbing the length of theta_true
        cs_class = get_cs_class_from_val(sp_data["cs_name_val"])
        num_params = len(cs_class.idcs_to_consider)
        num_train_points = sp_data["num_theta_multiplier"] * num_params

        if title != None:
            title = title
        else:
            title = meth_label + " Parameter Values"

        fig = self.__plot_2D_general(
            data,
            data_names,
            data_true,
            y_label,
            title,
            False,
            num_train_points,
            be_theta,
        )
        # save or show figure
        if self.save_figs:
            save_path_to = os.path.join(job.fn(""), "line_plots", "params_" + z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

    def __plot_2D_general(
        self,
        data,
        data_names,
        data_true,
        y_label,
        title,
        log_data,
        num_train_points=0,
        be=None,
    ):
        """
        Plots 2D values of the same data type (ei, sse, min sse) on multiple subplots

        Parameters
        -----------
        data:np.ndarray (n_runs x n_iters x n_params)
            Array of data from bo workflow runs
        data_names: list(str)
            List of data names
        data_true: list/ndarray(float/int) or None,
            The true/reference values of each parameter
        y_label: str
            The y label of the plot
        title: str
            The title of the plot
        log_data: bool
            Plots data on natural log scale if True

        Returns
        --------
        fig: plt.figure, The figure object
        """

        # Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
        subplots_needed = data.shape[-1]
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=True
        )
        # Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, None, None)

        # Loop over different hyperparameters (number of subplots)
        for i, ax in enumerate(axes.flatten()):
            # Only plot data if axis is visible
            if i < subplots_needed:
                # The index of the data is i, and one data type is in the last row of the data
                one_data_type = data[:, :, i]

                # Loop over all runs
                for j in range(one_data_type.shape[0]):
                    # Create label based on run #
                    label = "Run: " + str(j + 1)
                    data_df_j = self.__get_data_to_bo_iter_term(one_data_type[j])

                    # Define x axis
                    bo_len = len(data_df_j) + num_train_points
                    bo_space = np.linspace(1, bo_len, bo_len)

                    # Set appropriate notation
                    if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                        fmt = matplotlib.ticker.FuncFormatter(self.custom_format)
                        ax.yaxis.set_major_formatter(fmt)

                    # Plot data
                    if log_data == True:
                        data_df_j = np.log(data_df_j)

                    # duplicate the first value num_train_points times to show the number of training points
                    if num_train_points > 0:
                        data_train = be[j, i] if be is not None else data_df_j[0]
                        data_df_j = np.concatenate(
                            (np.full(num_train_points, data_train), data_df_j)
                        )
                        x_label = "Loss Evaluations"
                    else:
                        x_label = "BO Iterations"

                    ax.step(bo_space, data_df_j, label=label)

                    # Plot true value if applicable
                    if data_true is not None and j == one_data_type.shape[0] - 1:
                        ax.axhline(
                            y=list(data_true.values())[i],
                            color="red",
                            linestyle="--",
                            label="True Value",
                        )

                    # Set plot details
                    title = r"$" + data_names[i] + "$"
                    self.__set_subplot_details(
                        ax, bo_space, data_df_j, None, None, title
                    )

                if not log_data and data_true is None:
                    ax.set_yscale("log")

            # Add legends and handles from last subplot that is visible
            if i == subplots_needed - 1:
                handles, labels = axes[0, -1].get_legend_handles_labels()

        for axs in axes[-1]:
            axs.set_xlabel(x_label, fontsize=self.other_fntsz)

        for axs in axes[:, 0]:
            axs.set_ylabel(y_label, fontsize=self.other_fntsz)

        # Plots legend and title
        plt.tight_layout()
        fig.legend(
            handles,
            labels,
            loc="center left",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(1.0, 0.60),
            borderaxespad=0,
        )

        return fig

    def custom_format(self, x, pos):
        """
        Custom format for 10x notation

        Parameters
        -----------
        x: float
            The value to format
        pos: int
            The position of the value

        Returns
        --------
        str: The formatted value

        Notes
        ------
        Returns 0 if x is 0 and formats the value using scientific notation otherwise
        """
        if x == 0:
            return "0"
        formatted = "{:2.2e}".format(x)  # Format the value using scientific notation
        mantissa, exponent = formatted.split("e")
        return r"${} \times 10^{{{}}}$".format(mantissa, int(exponent))

    # def add_training_iters(self, data):

    def plot_objs_all_methods(self, z_choices, log_data=False, title=None):
        """
        Plots EI, SSE, Min SSE, and EI values vs BO iter for all 7 methods.

        Parameters
        -----------
        z_choices:  list(str)
            list(str)ings "sse_sim", "sse_mean", "sse_var", and/or "acq". The values that will be plotted
        log_data: bool, default False
            Plots data on natural log scale if True
        title: str or None, default None
            Title of plot

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        --------
        Each method is displayed in a different plot and each objective in a different subplot.
        """
        # Break down plot dict and check for correct things
        save_path = self.analyzer.make_dir_name_from_criteria(
            self.analyzer.criteria_dict
        )
        x_label = "Loss Evaluations"

        # Assert Statements
        assert isinstance(title, str) or title is None, "title must be a string or None"
        assert isinstance(log_data, bool), "log_data must be boolean"
        assert isinstance(z_choices, (list, str)), "z_choices must be list or string"
        if isinstance(z_choices, str):
            z_choices = list(z_choices)
        assert all(
            isinstance(item, str) for item in z_choices
        ), "z_choices elements must be str"
        for i in range(len(z_choices)):
            assert z_choices[i] in [
                "min_sse",
                "sse",
                "acq",
            ], "z_choices items must be 'min_sse', 'sse', or 'acq'"

        # Create figure and axes. Number of subplots is 1 for each acq, sse, sse_sim etc.
        subplots_needed = len(z_choices)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=False
        )

        # Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, None, None)
        bo_len_max = 1
        # Get all jobs
        job_pointer = self.analyzer.get_jobs_from_criteria()
        # Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        # Back out best runs from job_list_best
        emph_runs = list(df_best["Run Number"].values)
        methods = df_best["BO Method"].values
        indices_to_insert = [
            i for i, name in enumerate(self.method_names) if name not in methods
        ]
        for index in sorted(indices_to_insert):
            emph_runs.insert(index, 0)
        # Initialize min and max for acq
        miny = 1e-16
        maxy = 0
        # Loop over different methdods (number of subplots)
        for i in range(len(job_pointer)):
            # Get data
            data, data_names, data_true, sp_data, data_true_med = (
                self.analyzer.objective_trajectories(job_pointer[i], z_choices)
            )
            be, be_theta = self.analyzer.best_error(job_pointer[i])
            GPBO_method_val = sp_data["meth_name_val"]
            # Get number of iterations to add at the beginning by grabbing the length of theta_true
            cs_class = get_cs_class_from_val(sp_data["cs_name_val"])
            num_params = len(cs_class.idcs_to_consider)
            num_train_points = sp_data["num_theta_multiplier"] * num_params

            # Max iters is number of BO iters + training points
            max_iters = sp_data["bo_iter_tot"] + num_train_points

            # Create label based on method #
            label = self.method_names[GPBO_method_val - 1]
            # Loop over number of data types
            for k, ax in enumerate(axes.flatten()):
                # Only plot data if axis is visible
                if k < subplots_needed:
                    #             for k in range(data.shape[-1]):
                    # The index of the data type is k, and one data type is in the last row of the data
                    one_data_type = data[:, :, k]

                    # Get Number of runs in the job
                    runs_in_job = sp_data["bo_runs_in_job"]

                    # loop as long as there are runs in the file
                    for j in range(runs_in_job):
                        # Set run number of run in job
                        run_number = sp_data["bo_run_num"] + j
                        # Remove elements that are numerically 0
                        data_df_j = self.__get_data_to_bo_iter_term(one_data_type[j])
                        # Define x axis
                        bo_len = len(data_df_j) + num_train_points
                        bo_space = np.linspace(1, bo_len, bo_len)
                        if bo_len > bo_len_max:
                            bo_len_max = bo_len

                        # Set appropriate notation
                        if (
                            abs(np.max(data_df_j)) >= 1e3
                            or abs(np.min(data_df_j)) <= 1e-3
                        ):
                            fmt = matplotlib.ticker.FuncFormatter(self.custom_format)
                            ax.yaxis.set_major_formatter(fmt)

                        # Plot data
                        if log_data == True:
                            data_df_j = np.log(data_df_j)

                        if z_choices[k] == "acq":
                            miny = float(np.maximum(miny, np.min(data_df_j)))
                        else:
                            if np.min(data_df_j) < 0:
                                miny = np.min(data_df_j)
                            else:
                                miny = float(np.maximum(miny, np.min(data_df_j)))
                        maxy = float(np.maximum(maxy, np.max(data_df_j)))

                        data_train = be[j] if be is not None else data_df_j[0]
                        # duplicate the first value num_train_points times to show the number of training points
                        data_df_j_w_train = np.concatenate(
                            (np.full(num_train_points, data_train), data_df_j)
                        )
                        if z_choices[k] == "min_sse":
                            # Replace values higher than data_train in data_df_j_w_train with data_train
                            data_df_j_w_train[data_df_j_w_train > data_train] = (
                                data_train
                            )

                        # For the best result, print a solid line
                        if emph_runs[GPBO_method_val - 1] == run_number:
                            ax.plot(
                                bo_space,
                                data_df_j_w_train,
                                alpha=1,
                                color=self.colors[GPBO_method_val - 1],
                                label=label,
                                drawstyle="steps",
                            )
                        else:
                            ax.step(
                                bo_space,
                                data_df_j_w_train,
                                alpha=0.2,
                                color=self.colors[GPBO_method_val - 1],
                                linestyle="--",
                                drawstyle="steps",
                            )

                    # Plot true value if applicable
                    ls_xset = False
                    if data_true is not None and i == len(job_pointer) - 1:
                        data_ls = data_true[z_choices[k]]
                        if isinstance(data_ls, pd.DataFrame):
                            x = data_ls["Iter"].to_numpy()
                            ls_xset = True
                            if z_choices[k] == "min_sse":
                                y = data_ls["Min Obj Cum."].to_numpy()
                            else:
                                y = data_ls["Min Obj Act"].to_numpy()
                            ax.plot(
                                x,
                                y,
                                color="darkslategrey",
                                linestyle="solid",
                                label="NLS (Best)",
                            )

                    # Plot median value if applicable
                    ls_xset_med = False
                    if data_true_med is not None and i == len(job_pointer) - 1:
                        data_med_ls = data_true_med[z_choices[k]]
                        if isinstance(data_ls, pd.DataFrame):
                            ls_xset_med = True
                            x_med = data_med_ls["Iter"].to_numpy()
                            if z_choices[k] == "min_sse":
                                y_med = data_med_ls["Min Obj Cum."].to_numpy()
                            else:
                                y_med = data_med_ls["Min Obj Act"].to_numpy()
                            ax.plot(
                                x_med,
                                y_med,
                                color="k",
                                linestyle="dotted",
                                label="NLS (Median)",
                            )

                    # Set plot details
                    if i == 0:
                        max_x = max_iters
                    if ls_xset and ls_xset_med:
                        max_x = max(max(x), max(x_med), max_iters)
                    elif ls_xset:
                        max_x = max(max_iters, max(x))
                    elif ls_xset_med:
                        max_x = max(max_iters, max(x_med))

                    # Set plot details
                    if z_choices[k] == "acq":
                        bo_space_org = np.linspace(1, max_iters, max_iters)
                    else:
                        bo_space_org = np.linspace(1, max_x, max_x)

                    if z_choices[k] == "acq":
                        self.__set_subplot_details(
                            ax,
                            bo_space_org,
                            np.array([miny, maxy]),
                            x_label,
                            rf"${data_names[k]}$",
                            None,
                        )
                    else:
                        use_y = (
                            np.concatenate(
                                [np.array([miny, maxy]), [np.min(y), np.max(y)]]
                            )
                            if ls_xset
                            else np.array([miny, maxy])
                        )
                        self.__set_subplot_details(
                            ax,
                            bo_space_org,
                            use_y,
                            x_label,
                            rf"${data_names[k]}$",
                            None,
                        )

        # Get legend and handles
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if log_data == False:
            for ax in axes.flatten():
                ax.set_yscale("log")

        # Get correct legend order
        desired_order = ["NLS (Best)", "NLS (Median)"] + self.method_names
        order_dict = {label: i for i, label in enumerate(desired_order)}
        handles_labels = zip(handles, labels)
        handles_labels_sorted = sorted(
            handles_labels, key=lambda x: order_dict.get(x[1], float("inf"))
        )
        handles, labels = zip(*handles_labels_sorted)

        # Plots legend and title
        plt.tight_layout()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(1.0, 0.93),
            borderaxespad=0,
        )

        # save or show figure
        if self.save_figs:
            z_choices_sort = sorted(
                z_choices, key=lambda x: ("sse", "min_sse", "acq").index(x)
            )
            save_path_dir = os.path.join(save_path, "line_plots")
            save_path_to = os.path.join(
                save_path_dir, "all_meth_" + "_".join(map(str, z_choices_sort))
            )
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return

    def __get_z_plot_names_hms(self, z_choices, sim_sse_var_ei):
        """
        Returns the z data and title for the heat map plots

        Parameters
        -----------
        z_choices: str
            One of "sse_sim", "sse_mean", "sse_var", or "acq". The values that will be plotted
        sim_sse_var_ei: tuple(np.ndarray, len=4) or tuple(np.ndarray, np.ndarray, np.ndarray, None)
            Tuple of the data from the self.analyzer.gp_heat_map_data() method

        Returns
        --------
        all_z_data: list(np.ndarray)
            List of z data for each objective to plot
        all_z_titles: list(str)
            Mathematical titles for each objective to plot
        all_z_titles_pre: list(str)
            Titles for each objective to plot

        Raises
        ------
        ValueError
            If z_choice is not one of "sse_sim", "sse_mean", "sse_var", or "acq"
        """
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        all_z_data = []
        all_z_titles = []
        all_z_titles_pre = []
        # Find z based on z_choice
        # Fix me: Heat Maps always use just theta: Only the bar labels need to change
        for z_choice in z_choices:
            if "sse_sim" == z_choice:
                title = r"$\mathscr{L}(\mathbf{\theta})$"
                # title = r"$g(\mathbf{\theta})$"
                all_z_data.append(sse_sim)
                all_z_titles.append(title)
                all_z_titles_pre.append("SSE Loss Function, ")
            elif "sse_mean" == z_choice:
                title = r"$\tilde{\mathscr{L}}(\mathbf{\theta})$"
                # title = r"$\tilde{g}(\mathbf{\theta})$"
                all_z_data.append(sse_mean)
                all_z_titles.append(title)
                all_z_titles_pre.append("(Predicted) SSE Loss Function, ")
            elif "sse_var" == z_choice:
                all_z_data.append(sse_var)
                all_z_titles_pre.append("Predicted Variance, ")
                all_z_titles.append(
                    r"$\mathbf{\sigma}^2_{\tilde{\mathscr{L}}(\mathbf{\theta})}$"
                )
                # all_z_titles.append(
                #     r"$\mathbf{\sigma}^2_{\tilde{g}(\mathbf{\theta})}$"
                # )
            elif "acq" == z_choice:
                all_z_data.append(ei)
                all_z_titles.append(r"$\Xi(\mathbf{\theta})$")
                all_z_titles_pre.append("Aquisition Function, ")
            else:
                raise ValueError("choice must contain 'sim', 'mean', 'var', or 'acq'")
        if len(all_z_data) == 1:
            return all_z_data[0], all_z_titles[0], all_z_titles_pre[0]
        else:
            return all_z_data, all_z_titles, all_z_titles_pre

    def plot_local_min_hms(
        self, pair, tot_runs_nls, levels=10, log_data=False, title=None
    ):
        """
        Plots the frequency of local minima given the surface for NLS (true surface) and GPBO (predicted surface) for all GBPO methods

        Parameters
        ----------
        pair: int
            The pair of data parameters. pair 0 is the 1st pair
        levels: int or list(int)
            Number of contour lines to draw for each method
        log_data: bool, default False
            Plots contour data on natural log scale if True
        title: str or None, default None
            Title of plot

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If meshgrids are not the correct shape
            If there are not enough levels for the number of subplots
        Warning
            If log_data is True and minimum values to plot is less than or equal to 0

        Notes
        -------
        For this function, each method is its own subplot. Each plot must be generated separately for each objective function choice.
        """

        assert isinstance(pair, int), "pair must be an integer"
        assert isinstance(levels, (int)), "levels must be an int"
        assert isinstance(log_data, bool), "log_data must be boolean"
        assert isinstance(title, str) or title is None, "title must be a string or None"

        # Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        # Back out best runs from job_list_best
        emph_runs = df_best["Run Number"].values
        emph_iters = df_best["BO Iter"].values
        # emph_iters = (df_best['Max Evals'] / 10).round().astype(int).values

        # Make figures and define number of subplots based on number of different methods + 1 sse_sim map
        subplots_needed = len(job_list_best) + 1
        # meth_to_plt = [key for key, val in self.gpbo_meth_dict.items() if val in meth_list_plt]
        z_choice = "sse_mean"
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=False, sharey=False, threeD=True, x_size=6, y_size=6
        )

        all_z_data = []
        all_sp_data = []
        all_freq_data = []

        # Get all data for subplots needed
        # Loop over number of subplots needed
        for i in range(len(job_list_best)):
            method_list_plt = [df_best["BO Method"].iloc[i]]
            get_ei = False
            # Get data
            analysis_list = self.analyzer.gp_heat_map_data(
                job_list_best[i], emph_runs[i], emph_iters[i], pair, get_ei=get_ei
            )
            sim_sse_var_ei, test_mesh, param_info_dict, sp_data = analysis_list
            # Set correct values based on propagation of errors for gp
            sim_sse_var_ei = self.__scale_z_data(sim_sse_var_ei, sp_data, log_data)

            theta_true = param_info_dict["true"]

            plot_axis_names = param_info_dict["names"]
            idcs_to_plot = param_info_dict["idcs"]
            z, title2, tit2_pre = self.__get_z_plot_names_hms(z_choice, sim_sse_var_ei)

            # Get x and y data from test_mesh
            xx, yy = test_mesh  # NxN, NxN
            # Assert sattements
            assert xx.shape == yy.shape, "Test_mesh must be 2 NxN arrays"
            assert z.shape == xx.shape, "Array z must be NxN"

            all_z_data.append(z)
            all_sp_data.append(sp_data)
            local_min_sets, gpbo_runs = self.analyzer.compare_min(
                tot_runs_nls, method_list_plt
            )
            gpbo_freq = local_min_sets["GPBO Matches"].to_numpy()
            all_freq_data.append(gpbo_freq)

            if (i == len(job_list_best) - 1) and z_choice == "sse_mean":
                z_sim, title3, tit3_pre = self.__get_z_plot_names_hms(
                    "sse_sim", sim_sse_var_ei
                )
                all_z_data.append(z_sim)
                # Get local min data
                local_min_sets = self.analyzer.categ_min(tot_runs_nls, w_noise=False)
                nls_freq = local_min_sets["Num Occurrences"].to_numpy()
                all_freq_data.append(nls_freq)

        # Initialize need_unscale to False
        need_unscale = False

        # Unlog scale the data if vmin is 0 and log_data = True
        if np.amin(all_z_data) == -np.inf or np.isnan(np.amin(all_z_data)):
            need_unscale = True
            if log_data:
                warnings.warn("Cannot plot log scaled data! Reverting to original")
                z = np.exp(all_z_data[i])

        # Find the maximum and minimum values in your data to normalize the color scale
        vmin = min(np.min(arr) for arr in all_z_data)
        vmax = max(np.max(arr) for arr in all_z_data)
        # Check if data scales 2+ orders of magnitude
        mag_diff = (
            int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0
            if vmin > 0
            else False
        )

        # Create a common color normalization for all subplots
        # Do not use log10 scale if natural log scaling data or the difference in min and max values < 1e-3
        if log_data == True or need_unscale or not mag_diff:
            norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False)
            new_ticks = matplotlib.ticker.MaxNLocator(
                nbins=7, min_n_ticks=4
            )  # Set up to 12 ticks
            nticks = new_ticks.tick_values(vmin, vmax)
            cbar_ticks = np.linspace(vmin, vmax, len(nticks))
        else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
            # new_ticks = matplotlib.ticker.LogLocator(numticks=7)
            # nticks = new_ticks.tick_values(vmin, vmax)
            # cbar_ticks = np.linspace(vmin, vmax, len(nticks))
            cbar_ticks = np.logspace(
                np.log10(vmin), np.log10(vmax), levels
            )  # Set 7 equally spaced ticks
            # Get log10 scale bounds
            min_power = np.floor(
                np.log10(vmin)
            )  # Round down the logarithm to get the closest power of 10
            max_power = np.ceil(
                np.log10(vmax)
            )  # Round up the logarithm to get the closest power of 10
            # Create the ticks at powers of 10 within this range
            tick_num = int(max_power - min_power)
            if tick_num - 2 < 3:
                tick_num *= 2
            nticks = np.logspace(min_power, max_power, tick_num + 1)

        # Set theta values to plot
        theta_vals = np.vstack(local_min_sets["Theta Min Obj Cum."])
        thetas = theta_vals[:, idcs_to_plot]
        dx = (np.max(xx) - np.min(xx)) * 0.05
        dy = (np.max(yy) - np.min(yy)) * 0.05

        # Set x and y labels
        if "theta" in plot_axis_names[0] or "tau" in plot_axis_names[0]:
            xlabel = r"$\mathbf{" + "\\" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + "\\" + plot_axis_names[1] + "}$"
        else:
            xlabel = r"$\mathbf{" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + plot_axis_names[1] + "}$"

        # Set plot details
        # Loop over number of subplots
        for i in range(subplots_needed):
            if i != len(job_list_best):
                # Get method value from json file
                GPBO_method_val = all_sp_data[i]["meth_name_val"]
                ax_idx = int(GPBO_method_val - 1)
                ax_row, ax_col = plot_mapping[i]
            else:
                ax_idx = len(job_list_best)
                ax_row, ax_col = plot_mapping[ax_idx]

            z = all_z_data[i]

            use_scientific = np.amin(abs(z)) < 1e-1 or np.amax(abs(z)) > 1000
            if use_scientific:
                fmt = matplotlib.ticker.FuncFormatter(self.custom_format)
            else:
                fmt = "%2.2f"

            if np.all(z == z[0]):
                z = abs(np.random.normal(scale=1e-14, size=z.shape))

            cmap = plt.cm.get_cmap(self.cmap)
            # Create a BoundaryNorm with discrete levels
            discrete_cmap = colors.ListedColormap(
                cmap(np.linspace(0, 1, len(cbar_ticks) - 1))
            )
            boundary_norm = colors.BoundaryNorm(cbar_ticks, discrete_cmap.N)

            # Create a colormap and colorbar for each subplot
            cs_fig = ax[ax_row, ax_col].contourf(
                xx,
                yy,
                z,
                levels=cbar_ticks,  # cbar_ticks, #When doing zoom in activate this
                tick_positions=nticks,
                cmap=cmap,  # discrete_cmap,
                norm=norm,  # boundary_norm ,
                zdir="z",
                offset=0,
            )

            # # Create a line contour for each colormap
            if levels is not None:
                num_levels = len(cbar_ticks)
                indices = np.linspace(0, len(cs_fig.levels) - 1, num_levels, dtype=int)
                indices = np.unique(np.sort(indices))
                selected_levels = cs_fig.levels[indices]
                cs2_fig = ax[ax_row, ax_col].contour(
                    xx,
                    yy,
                    z,
                    levels=selected_levels,  # levels=cs_fig.levels[::tot_lev[i]]
                    colors="k",
                    alpha=0.7,
                    linestyles="dashed",
                    linewidths=3,
                    norm=norm,
                    zdir="z",
                    offset=0,
                )

            # plot theta frequencies
            ax[ax_row, ax_col].bar3d(
                thetas[0, 0] - dx / 2,
                thetas[0, 1] - dy / 2,
                0,
                dx,
                dy,
                all_freq_data[i][0],
                color="green",
                alpha=0.4,
            )
            if len(thetas) > 1:
                ax[ax_row, ax_col].bar3d(
                    thetas[1:, 0] - dx / 2,
                    thetas[1:, 1] - dy / 2,
                    0,
                    dx,
                    dy,
                    all_freq_data[i][1:],
                    color="blue",
                    alpha=0.4,
                )

            # Set plot details
            if i != len(job_list_best):
                label_name = self.method_names[ax_idx]
            else:
                label_name = "NLS"  # tit3_pre + title3
            zlabel = "Frequency"
            self.__set_subplot_details(
                ax[ax_row, ax_col],
                xx,
                yy,
                xlabel,
                ylabel,
                label_name,
                plot_z=all_freq_data[i],
                zlabel=zlabel,
            )
            if i == len(job_list_best) and sp_data["cs_name_val"] != 15:
                ax[ax_row, ax_col].set_zlim(zmax=tot_runs_nls)

            if all_sp_data[0]["cs_name_val"] in [16, 17]:
                ax[ax_row, ax_col].ticklabel_format(
                    style="scientific", axis="y", scilimits=(-2, 2)
                )
                ax[ax_row, ax_col].ticklabel_format(
                    style="scientific", axis="x", scilimits=(-2, 2)
                )
                # ax[ax_row, ax_col].w_zaxis.set_tick_params(rotation=90)
                ax[ax_row, ax_col].xaxis.get_offset_text().set_fontsize(
                    self.other_fntsz
                )  # Adjust size as needed
                ax[ax_row, ax_col].yaxis.get_offset_text().set_fontsize(
                    self.other_fntsz
                )
                ax[ax_row, ax_col].zaxis.get_offset_text().set_fontsize(
                    self.other_fntsz
                )  # For 3D plots

            ax[ax_row, ax_col].tick_params(axis="y", rotation=-30)
            ax[ax_row, ax_col].tick_params(axis="x", rotation=45)
            # ax[ax_row, ax_col].tick_params(axis='x', rotation=90)

        # Get legend information and make colorbar on 1st plot
        handles, labels = ax[0, 0].get_legend_handles_labels()
        if log_data is True and not need_unscale:
            title2 = "log(" + title2 + ")"

        cb_ax = fig.add_axes([1.03, 0, 0.04, 1])
        cbar = fig.colorbar(
            cs_fig,
            orientation="vertical",
            ax=ax,
            cax=cb_ax,
            format=fmt,
            ticks=nticks,  # cbar_ticks, #When doing zoom in activate this
            use_gridspec=True,
        )
        try:
            cbar.ax.locator_params(axis="y", numticks=7)
        except:
            cbar.ax.locator_params(axis="y", nbins=7)
        cbar.ax.tick_params(labelsize=self.other_fntsz)

        cbar.ax.set_ylabel(
            tit2_pre + title2, fontsize=self.other_fntsz, fontweight="bold"
        )

        # Print the title
        if title is not None:
            title = title

        # Print the title and labels as appropriate
        # Define x and y labels
        # For case studies 16 and 17, change the parameter names to be the correct ones from calc_y_fxns
        if all_sp_data[0]["cs_name_val"] in [16, 17]:
            plot_axis_names = tuple(
                "tau_{12}" if name == "theta_1" else "tau_{21}"
                for name in plot_axis_names
            )

        plt.draw()
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.25)

        # save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(save_path, "line_plots", "local_min_3d")
            save_path_to = os.path.join(save_path_dir, z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        # for axs in ax[-1]:
        #     axs.set_xlabel(xlabel, fontsize=self.other_fntsz)

        # for axs in ax[:, 0]:
        #     axs.set_ylabel(ylabel, fontsize=self.other_fntsz)

    def plot_hms_all_methods(
        self, pair, z_choice, levels=10, log_data=False, title=None
    ):
        """
        Plots comparison of y_sim, GP_mean, GP_stdev, and the acquisition function values for all methods
        Parameters
        ----------
        pair: int
            The pair of data parameters. pair 0 is the 1st pair
        z_choice: str
            "sse_sim", "sse_mean", "sse_var", or "acq". The values that will be plotted
        levels: int, default 10
            Number of contour lines to draw.
        log_data: bool, default False
            Plots data on natural log scale if True
        title: str or None, default None
            Title of plot

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If meshgrids are not the correct shape
            If there are not enough levels for the number of subplots
        Warning
            If log_data is True and minimum values to plot is less than or equal to 0

        Notes
        -------
        For this function, each method is its own subplot. Each plot must be generated separately for each objective function choice.
        """
        assert isinstance(pair, int), "pair must be an integer"
        assert isinstance(z_choice, str), "z_choice must be a string"
        assert z_choice in [
            "sse_sim",
            "sse_mean",
            "sse_var",
            "acq",
        ], "z_choice must be one of 'sse_sim', 'sse_mean', 'sse_var', or 'acq'"
        assert isinstance(levels, (int)), "levels must be an int"
        assert isinstance(log_data, bool), "log_data must be boolean"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        print("Z Choice: ", z_choice)
        # Get best data for each method
        df_best, job_list_best = self.analyzer.get_best_data()
        # Back out best runs from job_list_best
        emph_runs = df_best["Run Number"].values
        emph_iters = df_best["BO Iter"].values
        # emph_iters = (df_best['Max Evals'] / 10).round().astype(int).values

        # Make figures and define number of subplots based on number of different methods + 1 sse_sim map
        subplots_needed = len(job_list_best)
        if z_choice == "sse_mean":
            subplots_needed += 1
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=True, sharey=True
        )

        all_z_data = []
        all_sp_data = []
        all_theta_opt = []
        all_theta_next = []
        all_train_theta = []

        # Get all data for subplots needed
        # Loop over number of subplots needed
        for i in range(len(job_list_best)):
            if "acq" in z_choice:
                get_ei = True
            else:
                get_ei = False
            # Get data
            analysis_list = self.analyzer.gp_heat_map_data(
                job_list_best[i], emph_runs[i], emph_iters[i], pair, get_ei=get_ei
            )
            sim_sse_var_ei, test_mesh, param_info_dict, sp_data = analysis_list
            # Set correct values based on propagation of errors for gp
            sim_sse_var_ei = self.__scale_z_data(sim_sse_var_ei, sp_data, log_data)
            MAE = sklearn.metrics.mean_absolute_error(
                sim_sse_var_ei[0].flatten(), sim_sse_var_ei[1].flatten()
            )
            MAPD = mean_absolute_percentage_error(
                sim_sse_var_ei[0].flatten(), sim_sse_var_ei[1].flatten()
            )
            if i == 0:
                print("CS Name: ", df_best["CS Name"].values[0])
            if z_choice == "sse_mean" and self.analyzer.mode == "act":
                print(df_best["BO Method"].values[i])
                print("MAE: ", MAE)
                print("MAPD: ", MAPD)
                print(
                    "Avg Gamma: ",
                    np.mean(sim_sse_var_ei[2].flatten() / sim_sse_var_ei[1].flatten()),
                )
            theta_true = param_info_dict["true"]
            theta_opt = param_info_dict["min_sse"]
            theta_next = param_info_dict["opt_acq"]
            train_theta = param_info_dict["train"]
            plot_axis_names = param_info_dict["names"]
            idcs_to_plot = param_info_dict["idcs"]
            z, title2, tit2_pre = self.__get_z_plot_names_hms(z_choice, sim_sse_var_ei)

            # Get x and y data from test_mesh
            xx, yy = test_mesh  # NxN, NxN
            # Assert sattements
            assert xx.shape == yy.shape, "Test_mesh must be 2 NxN arrays"
            assert z.shape == xx.shape, "Array z must be NxN"

            all_z_data.append(z)
            all_sp_data.append(sp_data)
            all_theta_opt.append(theta_opt)
            all_theta_next.append(theta_next)
            all_train_theta.append(train_theta)

            if (i == len(job_list_best) - 1) and z_choice == "sse_mean":
                z_sim, title3, tit3_pre = self.__get_z_plot_names_hms(
                    "sse_sim", sim_sse_var_ei
                )
                all_z_data.append(z_sim)
                all_theta_opt.append(None)
                all_theta_next.append(None)
                all_train_theta.append(None)

        # Initialize need_unscale to False
        need_unscale = False

        # Unlog scale the data if vmin is 0 and log_data = True
        if np.amin(all_z_data) == -np.inf or np.isnan(np.amin(all_z_data)):
            need_unscale = True
            if log_data:
                warnings.warn("Cannot plot log scaled data! Reverting to original")
                z = np.exp(all_z_data[i])

        if z_choice != "acq":
            # Find the maximum and minimum values in your data to normalize the color scale
            vmin = min(np.min(arr) for arr in all_z_data)
            vmax = max(np.max(arr) for arr in all_z_data)
            # Check if data scales 2+ orders of magnitude
            mag_diff = (
                int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0
                if vmin > 0
                else False
            )

            # Create a common color normalization for all subplots
            # Do not use log10 scale if natural log scaling data or the difference in min and max values < 1e-3
            if log_data == True or need_unscale or not mag_diff:
                norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
                cbar_ticks = np.linspace(
                    vmin, vmax, levels
                )  # Set 7 equally spaced ticks
                nticks = cbar_ticks

            else:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
                # new_ticks = matplotlib.ticker.LogLocator(numticks=7)
                # nticks = new_ticks.tick_values(vmin, vmax)
                # cbar_ticks = nticks #np.logspace(np.log10(vmin), np.log10(vmax), len(nticks))
                # cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), len(nticks))
                cbar_ticks = np.logspace(
                    np.log10(vmin), np.log10(vmax), levels
                )  # Set 7 equally spaced ticks
                # Get log10 scale bounds
                min_power = np.floor(
                    np.log10(vmin)
                )  # Round down the logarithm to get the closest power of 10
                max_power = np.ceil(
                    np.log10(vmax)
                )  # Round up the logarithm to get the closest power of 10
                # print(min_power, max_power)
                # print(cbar_ticks)
                # Create the ticks at powers of 10 within this range
                tick_num = int(max_power - min_power)
                if tick_num - 2 < 3:
                    tick_num *= 2
                nticks = np.logspace(min_power, max_power, tick_num + 1)
                # nticks = np.logspace(min_power, max_power, int(max_power - min_power + 1))
                # print(nticks)

        # Set plot details
        # Loop over number of subplots
        for i in range(subplots_needed):
            if i != len(job_list_best):
                # Get method value from json file
                GPBO_method_val = all_sp_data[i]["meth_name_val"]
                ax_idx = int(GPBO_method_val - 1)
                ax_row, ax_col = plot_mapping[i]
            else:
                ax_idx = len(job_list_best)
                ax_row, ax_col = plot_mapping[ax_idx]

            z = all_z_data[i]
            theta_opt = all_theta_opt[i]
            theta_next = all_theta_next[i]
            train_theta = all_train_theta[i]

            use_scientific = np.amin(abs(z)) < 1e-1 or np.amax(abs(z)) > 1000
            if use_scientific:
                fmt = matplotlib.ticker.FuncFormatter(self.custom_format)
            else:
                fmt = "%2.2f"

            if np.all(z == z[0]):
                z = abs(np.random.normal(scale=1e-14, size=z.shape))

            if z_choice == "acq":
                use_log10 = False
                # Find the maximum and minimum values in your data to normalize the color scale
                vmin = np.amin(z)
                vmax = np.amax(z)
                # Check if data scales 2+ orders of magnitude
                mag_diff = (
                    int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0
                    if vmin > 0
                    else False
                )

                # Create a common color normalization for all subplots
                # Do not use log10 scale if natural log scaling data or the difference in min and max values < 1e-3
                if log_data == True or need_unscale or not mag_diff:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.linspace(
                        vmin, vmax, levels
                    )  # Set 7 equally spaced ticks
                    nticks = cbar_ticks

                else:
                    use_log10 = True

                    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.logspace(
                        np.log10(vmin), np.log10(vmax), levels
                    )  # Set 7 equally spaced ticks
                    # Get log10 scale bounds
                    min_power = np.floor(
                        np.log10(vmin)
                    )  # Round down the logarithm to get the closest power of 10
                    max_power = np.ceil(
                        np.log10(vmax)
                    )  # Round up the logarithm to get the closest power of 10
                    # Create the ticks at powers of 10 within this range
                    tick_num = int(max_power - min_power)
                    if tick_num - 2 < 3:
                        tick_num *= 2
                    nticks = np.logspace(min_power, max_power, tick_num + 1)

            # Create a colormap and colorbar for each subplot
            if log_data == True:
                cs_fig = ax[ax_row, ax_col].contourf(
                    xx,
                    yy,
                    z,
                    levels=self.zbins,  # cbar_ticks
                    cmap=plt.cm.get_cmap(self.cmap),
                    norm=norm,
                )
            else:
                cs_fig = ax[ax_row, ax_col].contourf(
                    xx,
                    yy,
                    z,
                    levels=cbar_ticks,  # self.zbins,
                    tick_positions=nticks,
                    cmap=plt.cm.get_cmap(self.cmap),
                    norm=norm,
                )

            # Create a line contour for each colormap
            if levels is not None:
                num_levels = len(cbar_ticks)
                indices = np.linspace(0, len(cs_fig.levels) - 1, num_levels, dtype=int)
                selected_levels = cs_fig.levels[indices]
                cs2_fig = ax[ax_row, ax_col].contour(
                    cs_fig,
                    levels=selected_levels,  # levels=cs_fig.levels[::tot_lev[i]]
                    colors="k",
                    alpha=0.7,
                    linestyles="dashed",
                    linewidths=3,
                    norm=norm,
                )

            # plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax[ax_row, ax_col].scatter(
                    theta_true[idcs_to_plot[0]],
                    theta_true[idcs_to_plot[1]],
                    color="blue",
                    label="True",
                    s=200,
                    marker=(5, 1),
                    zorder=2,
                )
            if train_theta is not None:
                ax[ax_row, ax_col].scatter(
                    train_theta[:, idcs_to_plot[0]],
                    train_theta[:, idcs_to_plot[1]],
                    color="green",
                    s=100,
                    label="Train",
                    marker="x",
                    zorder=1,
                )
            if theta_next is not None:
                ax[ax_row, ax_col].scatter(
                    theta_next[idcs_to_plot[0]],
                    theta_next[idcs_to_plot[1]],
                    color="black",
                    s=175,
                    label="Opt Acq",
                    marker="^",
                    zorder=3,
                )
            if theta_opt is not None:
                ax[ax_row, ax_col].scatter(
                    theta_opt[idcs_to_plot[0]],
                    theta_opt[idcs_to_plot[1]],
                    color="darkmagenta",
                    s=160,
                    label="Min Obj",
                    marker=".",
                    edgecolor="magenta",
                    linewidth=0.7,
                    zorder=4,
                )

            if z_choice == "acq":
                divider1 = make_axes_locatable(ax[ax_row, ax_col])
                cax1 = divider1.append_axes("right", size="5%", pad="6%")
                cbar = fig.colorbar(
                    cs_fig, ax=ax[ax_row, ax_col], cax=cax1, use_gridspec=True
                )  # format = fmt
                try:
                    cbar.ax.locator_params(axis="y", numticks=7)
                except:
                    cbar.ax.locator_params(axis="y", nbins=7)
                fmt_acq = matplotlib.ticker.FuncFormatter(self.custom_format)
                cbar.ax.yaxis.set_major_formatter(fmt_acq)
                cbar.ax.set_ylabel(
                    tit2_pre + title2, fontsize=self.other_fntsz, fontweight="bold"
                )
                cbar.ax.tick_params(
                    labelsize=int(self.other_fntsz / 2), labelleft=False
                )
                if use_log10:
                    cbar.formatter.set_powerlimits((0, 0))

            # Set plot details
            if i != len(job_list_best):
                label_name = self.method_names[ax_idx]
            else:
                label_name = tit3_pre + title3
            self.__set_subplot_details(
                ax[ax_row, ax_col], xx, yy, None, None, label_name
            )

            if all_sp_data[0]["cs_name_val"] in [16, 17]:
                ax[ax_row, ax_col].ticklabel_format(
                    style="scientific", axis="both", scilimits=(-2, 2)
                )

        # Get legend information and make colorbar on 1st plot
        handles, labels = ax[0, 0].get_legend_handles_labels()
        if log_data is True and not need_unscale:
            title2 = "log(" + title2 + ")"

        if z_choice != "acq":
            cb_ax = fig.add_axes([1.03, 0, 0.04, 1])
            cbar = fig.colorbar(
                cs_fig,
                orientation="vertical",
                ax=ax,
                cax=cb_ax,
                format=fmt,
                use_gridspec=True,
                ticks=nticks,
            )
            try:
                cbar.ax.locator_params(axis="y", numticks=7)
            except:
                cbar.ax.locator_params(axis="y", nbins=7)
            cbar.ax.tick_params(labelsize=self.other_fntsz)
            cbar.ax.set_ylabel(
                tit2_pre + title2, fontsize=self.other_fntsz, fontweight="bold"
            )

        # Print the title
        if title is not None:
            title = title

        # Print the title and labels as appropriate
        # Define x and y labels
        # For case studies 16 and 17, change the parameter names to be the correct ones from calc_y_fxns
        if all_sp_data[0]["cs_name_val"] in [16, 17]:
            plot_axis_names = tuple(
                "tau_{12}" if name == "theta_1" else "tau_{21}"
                for name in plot_axis_names
            )

        if "theta" in plot_axis_names[0] or "tau" in plot_axis_names[0]:
            xlabel = r"$\mathbf{" + "\\" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + "\\" + plot_axis_names[1] + "}$"
        else:
            xlabel = r"$\mathbf{" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + plot_axis_names[1] + "}$"

        for axs in ax[-1]:
            axs.set_xlabel(xlabel, fontsize=self.other_fntsz)

        for axs in ax[:, 0]:
            axs.set_ylabel(ylabel, fontsize=self.other_fntsz)

        self.__set_plot_titles(fig, title, None, None)

        # Plots legend and title
        # if len(labels) > 0:
        #     # fig.legend(handles, labels, loc= "upper left", fontsize = self.other_fntsz,
        #     #         bbox_to_anchor=(0.2, 0.85), borderaxespad=0)
        #     fig.legend(handles, labels, loc= "upper left", fontsize = self.other_fntsz,
        #             bbox_to_anchor=(0.23, 0.87), borderaxespad=0)

        anchory = 0.49 if all_sp_data[0]["cs_name_val"] not in [16, 17] else 0.45

        fig.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(0.98, anchory),
            borderaxespad=0,
        )

        plt.tight_layout()

        # save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(
                save_path,
                "heat_maps",
                "all_methods",
                plot_axis_names[0] + "-" + plot_axis_names[1],
            )
            save_path_to = os.path.join(save_path_dir, z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return plt.show()

    def plot_gp_fit(
        self,
        job,
        run_num,
        bo_iter,
        pair,
        z_choices,
        levels=7,
        log_data=False,
        title=None,
    ):
        """
        Plots comparison of y_sim, GP_mean, and GP_stdev
        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        run_num: int
            The run number to analyze
        bo_iter: int
            The bo iteration to analyze
        pair: int
            The pair of data parameters. pair 0 is the 1st pair
        z_choices: str, list(str),
            One of "sse_sim", "sse_mean", "sse_var", or "acq". The values that will be plotted
        levels: int, list(int), or None
            Number of zbins to skip when drawing contour lines
        log_data: bool, default False
            Plots data on natural log scale if True
        title: str or None, default None
            Title of plot

        Returns
        -------
        plt.show(), A heat map of test_mesh and z

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If meshgrids are not the correct shape
            If there are not enough levels for the number of subplots
        Warning
            If log_data is True and minimum values to plot is less than or equal to 0

        Notes
        -------
        For this function, each objective function value is its own subplot. Each plot must be generated separately for each method.
        """
        # Assert Statements
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(run_num, int), "run_num must be an integer"
        assert isinstance(bo_iter, int), "bo_iter must be an integer"
        assert isinstance(pair, int), "pair must be an integer"
        assert isinstance(log_data, bool), "log_data must be boolean"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        assert isinstance(
            z_choices, (Iterable, str)
        ), "z_choices must be Iterable or str"
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        for z_choice in z_choices:
            assert z_choice in [
                "sse_sim",
                "sse_mean",
                "sse_var",
                "acq",
            ], "z_choices elements must be 'sse_sim', 'sse_mean', 'sse_var', or 'acq'"

        assert isinstance(levels, (int, list)), "levels must be an int or list"
        # Define plot levels
        if isinstance(levels, int):
            levels = [levels] * len(z_choices)
        else:
            levels = levels
        assert len(levels) == len(
            z_choices
        ), "levels must be int or have the same length as z_choices"
        # Get all data for subplots needed
        get_ei = True if "acq" in z_choices else False
        analysis_list = self.analyzer.gp_heat_map_data(
            job, run_num, bo_iter, pair, get_ei=get_ei
        )
        sim_sse_var_ei, test_mesh, param_info_dict, sp_data = analysis_list
        # Get method value from json file
        GPBO_method_val = sp_data["meth_name_val"]
        # Set correct values based on propagation of errors for gp
        sim_sse_var_ei = self.__scale_z_data(sim_sse_var_ei, sp_data, log_data)
        theta_true = param_info_dict["true"]
        theta_opt = param_info_dict["min_sse"]
        theta_next = param_info_dict["opt_acq"]
        train_theta = param_info_dict["train"]
        plot_axis_names = param_info_dict["names"]
        idcs_to_plot = param_info_dict["idcs"]

        # Assert sattements
        # Get x and y data from test_mesh
        xx, yy = test_mesh  # NxN, NxN
        assert xx.shape == yy.shape, "Test_mesh must be 2 NxN arrays"

        # Make figures and define number of subplots based on number of files (different methods)
        subplots_needed = len(z_choices)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=True, sharey=True
        )

        # Find z based on z_choice
        all_z_data, all_z_titles, all_z_titles_pre = self.__get_z_plot_names_hms(
            z_choices, sim_sse_var_ei
        )

        sse_cond = None

        # Loop over number of subplots
        for i, ax in enumerate(axes.flatten()):
            if i < subplots_needed:
                # Get data for z_choice
                z = all_z_data[i]
                need_unscale = False

                # Unlog scale the data if vmin is 0 and log_data = True
                if np.min(z) == -np.inf or np.isnan(np.min(z)) or np.min(z) == 0:
                    need_unscale = True
                    if log_data:
                        warnings.warn(
                            "Cannot plot log scaled data! Reverting to original"
                        )
                        z = np.exp(all_z_data[i])

                # Create normalization
                vmin = np.nanmin(z)
                vmax = np.nanmax(z)
                # If all z data are the same, add a small amount of noise to each to allow for plotting
                if vmin == vmax:
                    vmin -= 1e-14
                    vmax += 1e-14

                # Check if data scales 3 orders of magnitude
                mag_diff = (
                    int(math.log10(abs(vmax)) - math.log10(abs(vmin))) > 2.0
                    if vmin > 0
                    else False
                )

                if need_unscale == False and log_data:
                    title2 = "log(" + all_z_titles[i] + ")"
                else:
                    title2 = all_z_titles[i]

                condition = log_data or vmin < 0 or not mag_diff
                if "sse" in z_choices[i]:
                    if sse_cond is None:
                        sse_cond = condition
                    condition = sse_cond
                # Choose an appropriate colormap and scaling based on vmin, vmax, and log_data
                # If not using log data, vmin > 0, and the data scales 3 orders+ of magnitude use log10 to view plots
                if condition:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.linspace(vmin, vmax, levels[i])
                    nticks = cbar_ticks

                else:
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.logspace(
                        np.log10(vmin), np.log10(vmax), levels[i]
                    )  # Set 5 equally spaced ticks
                    # Get log10 scale bounds
                    min_power = np.floor(
                        np.log10(vmin)
                    )  # Round down the logarithm to get the closest power of 10
                    max_power = np.ceil(
                        np.log10(vmax)
                    )  # Round up the logarithm to get the closest power of 10
                    # Create the ticks at powers of 10 within this range
                    nticks = np.logspace(
                        min_power, max_power, int(max_power - min_power + 1)
                    )

                # Create a colormap and colorbar normalization for each subplot
                cs_fig = ax.contourf(
                    xx,
                    yy,
                    z,
                    levels=cbar_ticks,  # self.zbins,
                    # tick_positions=nticks,
                    cmap=plt.cm.get_cmap(self.cmap),
                    norm=norm,
                )

                # Create a line contour for each colormap
                if levels is not None:
                    num_levels = len(cbar_ticks)
                    indices = np.linspace(
                        0, len(cs_fig.levels) - 1, num_levels, dtype=int
                    )
                    selected_levels = cs_fig.levels[indices]
                    cs2_fig = ax.contour(
                        cs_fig,
                        levels=selected_levels,
                        colors="k",
                        alpha=0.7,
                        linestyles="dashed",
                        linewidths=3,
                        norm=norm,
                    )

                # plot min obj, max ei, true and training param values as appropriate
                if theta_true is not None:
                    ax.scatter(
                        theta_true[idcs_to_plot[0]],
                        theta_true[idcs_to_plot[1]],
                        color="blue",
                        label="True",
                        s=200,
                        marker=(5, 1),
                        zorder=2,
                    )
                if train_theta is not None:
                    ax.scatter(
                        train_theta[:, idcs_to_plot[0]],
                        train_theta[:, idcs_to_plot[1]],
                        color="green",
                        s=100,
                        label="Train",
                        marker="x",
                        zorder=1,
                    )
                if theta_next is not None:
                    ax.scatter(
                        theta_next[idcs_to_plot[0]],
                        theta_next[idcs_to_plot[1]],
                        color="black",
                        s=175,
                        label="Opt Acq",
                        marker="^",
                        zorder=3,
                    )
                if theta_opt is not None:
                    ax.scatter(
                        theta_opt[idcs_to_plot[0]],
                        theta_opt[idcs_to_plot[1]],
                        color="darkmagenta",
                        s=160,
                        label="Min Obj",
                        marker=".",
                        edgecolor="magenta",
                        linewidth=0.7,
                        zorder=4,
                    )

                # Set plot details
                self.__set_subplot_details(ax, xx, yy, None, None, all_z_titles[i])
                if sp_data["cs_name_val"] in [16, 17]:
                    ax.ticklabel_format(
                        style="scientific", axis="both", scilimits=(-2, 2)
                    )

                # Use a custom formatter for the colorbar
                fmt = matplotlib.ticker.FuncFormatter(self.custom_format)

                divider1 = make_axes_locatable(ax)
                cax1 = divider1.append_axes("right", size="5%", pad="6%")
                cbar = fig.colorbar(
                    cs_fig,
                    ax=ax,
                    cax=cax1,
                    use_gridspec=True,
                    ticks=cbar_ticks,
                )
                cbar.ax.yaxis.set_major_formatter(fmt)
                cbar.ax.tick_params(labelsize=int(self.other_fntsz / 2))

        # Get legend information and make colorbar on last plot
        handles, labels = axes[0, 0].get_legend_handles_labels()

        # Print the title
        if title is None:
            title = self.method_names[GPBO_method_val - 1]

        # For case studies 16 and 17, change the parameter names to be the correct ones from calc_y_fxns
        if sp_data["cs_name_val"] in [16, 17]:
            plot_axis_names = tuple(
                "tau_{12}" if name == "theta_1" else "tau_{21}"
                for name in plot_axis_names
            )

        # Print the title and labels as appropriate
        # Define x and y labels
        if "theta" in plot_axis_names[0] or "tau" in plot_axis_names[0]:
            xlabel = r"$\mathbf{" + "\\" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + "\\" + plot_axis_names[1] + "}$"
        else:
            xlabel = r"$\mathbf{" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + plot_axis_names[1] + "}$"

        for axs in axes[-1]:
            axs.set_xlabel(xlabel, fontsize=self.other_fntsz)

        for axs in axes[:, 0]:
            axs.set_ylabel(ylabel, fontsize=self.other_fntsz)

        self.__set_plot_titles(fig, title, None, None)

        # Plots legend and title
        fig.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(-0.02, 1),
            borderaxespad=0,
        )

        plt.tight_layout()

        # save or show figure
        if self.save_figs:
            z_choices_sort = sorted(
                z_choices,
                key=lambda x: ("sse_sim", "sse_mean", "sse_var", "acq").index(x),
            )
            z_choices_str = "_".join(map(str, z_choices_sort))
            title_str = title.replace(" ", "_").lower()
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(
                save_path,
                "heat_maps",
                title_str,
                plot_axis_names[0] + "-" + plot_axis_names[1],
                z_choices_str,
            )
            save_path_to = os.path.join(
                save_path_dir, "run_" + str(run_num) + "_" + "iter_" + str(bo_iter)
            )
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return plt.show()

    def __scale_z_data(self, sim_sse_var_ei, sp_data, log_data):
        """
        Scales the objective (sse_sim, sse_gp, sse_var, or acq_func) data based on the method and log_data

        Parameters
        -----------
        sim_sse_var_ei: tuple(np.ndarray, len=4) or tuple(np.ndarray, np.ndarray, np.ndarray, None)
            Tuple of the data from the self.analyzer.gp_heat_map_data() method
        sp_data: dict
            Dictionary of the data from the json file
        log_data: bool
            Plots data on natural log scale if True

        Returns
        --------
        sim_sse_var_ei: tuple(np.ndarray, len=4) or tuple(np.ndarray, np.ndarray, np.ndarray, None)
            tuple of the data from the analysis with correct scaling for plots
        """
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        # Get log or unlogged data values
        if log_data == False:
            # Change sse sim, mean, and stdev to not log for 1B
            if sp_data["meth_name_val"] in [2]:
                # SSE variance is var*(e^((log(sse)))^2
                sse_mean = np.exp(sse_mean)
                sse_var = sse_var * sse_mean**2
                sse_sim = np.exp(sse_sim)

        # If getting log values
        else:
            # Get log data from 1A, 2A, 2B, 2C, and 2D
            if not sp_data["meth_name_val"] in [2]:
                # SSE Variance is var/sse**2
                sse_var = sse_var / sse_mean**2
                sse_mean = np.log(sse_mean)
                sse_sim = np.log(sse_sim)

        sim_sse_var_ei = [sse_sim, sse_mean, sse_var, ei]
        return sim_sse_var_ei

    def __set_ylab_from_z(self, z_choice):
        """
        Sets the y label based on the z_choice

        Parameters
        -----------
        z_choice: str
            One of "sse", "min_sse", or "acq"

        Returns
        --------
        y_label: str
            The y label for the plot
        """

        if self.analyzer.mode == "gp":
            label_g = "\\tilde{\mathscr{L}}(\mathbf{"
            label_a = "(Predicted) SSE Loss Function, "
        else:
            label_g = "\mathscr{L}(\mathbf{"
            label_a = "SSE Loss Function, "
        if "sse" == z_choice:
            theta = "\\theta}^o" if self.analyzer.mode != "acq" else "\\theta^*}"
            y_label = label_g + theta + ")"
        if "min_sse" == z_choice:
            theta = "\\theta}^{\prime}" + ")"
            y_label = label_g + theta
        if "acq" == z_choice:
            label_a = "Aquisition Function, "
            y_label = "\Xi(\mathbf{\\theta^*})"
        final_label = label_a + r"$" + y_label + "$"
        return final_label

    def __get_data_to_bo_iter_term(self, data_all_iters):
        """
        Gets non-zero data for plotting from data array

        Parameters
        -----------
        data_all_iters: np.ndarray
            Data from all iterations

        Returns
        --------
        data_df_j:np.ndarray
            Data that is not numerically 0
        """
        # Remove elements that are numerically 0
        data_df_run = pd.DataFrame(data=data_all_iters)
        data_df_j = data_df_run.loc[(abs(data_df_run) > 1e-14).any(axis=1), 0]
        data_df_i = data_df_run.loc[:, 0]  # Used to be data_df_i
        # Ensure we have at least 2 elements to plot
        if len(data_df_j) < 2:
            data_df_j = data_df_i[
                0 : int(len(data_df_j) + 2)
            ]  # +2 for stopping criteria + 1 to include last point

        return data_df_j

    def __save_fig(self, save_path, ext="png", close=True):
        """Save a figure from pyplot.
        Parameters
        ----------
        save_path : string
            The path (and filename, without the extension) to save the
            figure to.
        ext : string (default='png')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
        close : boolean (default=True)
            Whether to close the figure after saving.  If you want to save
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.
        """

        # Extract the directory and filename from the given path
        directory = os.path.split(save_path)[0]
        filename = "%s.%s" % (os.path.split(save_path)[1], ext)
        if directory == "":
            directory = "."

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # The final path to save to
        savepath = os.path.join(directory, filename)

        # Actually save the figure
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

        # Close it
        if close:
            plt.close()

    def __create_subplots(
        self,
        num_subplots,
        sharex="row",
        sharey="none",
        threeD=False,
        x_size=6,
        y_size=6,
    ):
        """
        Creates Subplots based on the amount of data

        Parameters
        ----------
        num_subplots: int
            Total number of needed subplots
        sharex: str, default "row"
            sharex values for subplots
        sharey: str, default "none"
            sharey value for subplots

        Returns
        -------
        fig: matplotlib.figure
            The matplotlib figure object
        axes: matplotlib.axes.Axes
            2D array of axes
        total_ax_num: int
            The number of axes generated total
        plot_mapping: dict
            Dictionary mapping plot number to axes

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        """

        assert num_subplots >= 1, "Number of subplots must be at least 1"
        assert isinstance(num_subplots, int), "Num subplots must be int"
        # Make figures and define number of subplots
        # If you are making more than one figure, sharex is always true
        if num_subplots == 1:
            sharex = True

        # Make enough rows and columns and get close to equal number of each
        row_num = int(np.floor(np.sqrt(num_subplots)))
        col_num = int(np.ceil(num_subplots / row_num))
        assert (
            row_num * col_num >= num_subplots
        ), "row * col numbers must be at least equal to number of graphs"
        total_ax_num = row_num * col_num

        # Creat subplots
        gridspec_kw = {"wspace": 0.4, "hspace": 0.2}
        if threeD:
            subplot_kw = {"projection": "3d"}
        else:
            subplot_kw = {}
        fig, axes = plt.subplots(
            row_num,
            col_num,
            figsize=(col_num * x_size, row_num * y_size),
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            subplot_kw=subplot_kw,
        )

        # Turn off unused axes
        for i, axs in enumerate(axes.flatten()):
            if i >= num_subplots:
                axs.axis("off")

        # Make plot mapping to map an axes to an iterable value
        plot_mapping = {}
        for i in range(row_num):
            for j in range(col_num):
                plot_number = i * col_num + j
                plot_mapping[plot_number] = (i, j)

        return fig, axes, total_ax_num, plot_mapping

    def __set_subplot_details(
        self, ax, plot_x, plot_y, xlabel, ylabel, title, plot_z=None, zlabel=None
    ):
        """
        Function for setting plot settings

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to set the plot settings for
        plot_x: np.ndarray
            The x data for plotting
        plot_y: np.ndarray
            The y data for plotting
        xlabel: str or None
            The label for the x axis
        ylabel: str or None
            The label for the y axis
        title: str or None
            The subplot title

        Returns
        -------
        ax: matplotlib.axes.Axes
            The axes with the plot settings set

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Group inputs by type
        none_str_vars = [title, xlabel, ylabel]
        int_vars = [self.xbins, self.ybins, self.other_fntsz]
        arr_vars = [plot_x, plot_y]

        # Assert Statements
        assert all(
            isinstance(var, str) or var is None for var in none_str_vars
        ), "title, xlabel, and ylabel must be string or None"
        assert all(
            isinstance(var, int) for var in int_vars
        ), "xbins, ybins, and fontsize must be int"
        assert all(
            var > 0 or var is None for var in int_vars
        ), "integer variables must be positive"
        assert all(
            isinstance(var, (np.ndarray, pd.core.series.Series)) or var is None
            for var in arr_vars
        ), "plot_x, plot_y must be np.ndarray or pd.core.series.Series or None"

        # Set title, label, and axes
        if title is not None:
            pad = 6 + 4 * title.count("_")
            ax.set_title(title, fontsize=self.other_fntsz, fontweight="bold", pad=pad)
        if xlabel is not None:
            pad = 4 * xlabel.count("_") + self.other_fntsz * 1.1
            ax.set_xlabel(
                xlabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )
        if ylabel is not None:
            pad = 4 * ylabel.count("_") + self.other_fntsz * 1.1
            ax.set_ylabel(
                ylabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )
        if zlabel is not None:
            pad = 5 * zlabel.count("_") + self.other_fntsz
            ax.set_zlabel(
                zlabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )

        # Turn on tick parameters and bin number
        ax.xaxis.set_tick_params(labelsize=self.other_fntsz, direction="in", pad=5)
        ax.yaxis.set_tick_params(labelsize=self.other_fntsz, direction="in", pad=5)
        ax.locator_params(axis="y", nbins=self.ybins)
        ax.locator_params(axis="x", nbins=self.xbins)
        ax.minorticks_on()  # turn on minor ticks
        ax.tick_params(which="minor", direction="in", top=True, right=True)

        # Set a and y bounds and aspect ratio
        if plot_z is None:
            if plot_x is not None and not np.isclose(
                np.min(plot_x), np.max(plot_x), rtol=1e-6
            ):
                ax.set_xlim(left=np.min(plot_x), right=np.max(plot_x))

            if plot_y is not None and abs(np.min(plot_y)) <= 1e-16:
                ax.set_ylim(ymin=1e-16, ymax=np.max(plot_y) * 1.1)

            if plot_y is not None and (np.min(plot_y) == np.max(plot_y) == 0):
                ax.set_ylim(bottom=np.min(plot_y) - 0.05, top=np.max(plot_y) + 0.05)

            ax.set_box_aspect(1)
        else:
            ax.zaxis.set_tick_params(labelsize=self.other_fntsz, direction="in", pad=10)
            ax.locator_params(axis="z", nbins=self.ybins)
            if plot_x is not None and not np.isclose(
                np.min(plot_x), np.max(plot_x), rtol=1e-6
            ):
                ax.set_xlim(left=np.min(plot_x), right=np.max(plot_x))

            if plot_y is not None and not np.isclose(
                np.min(plot_y), np.max(plot_y), rtol=1e-6
            ):
                ax.set_ylim(bottom=np.min(plot_y), top=np.max(plot_y))

            if plot_z is not None:
                # if np.max(plot_z) > 10:
                #     max_value = np.maximum(np.max(plot_z), 1000)
                # else:
                max_value = np.maximum(np.max(plot_z), 5)
                ax.set_zlim(zmin=0, zmax=max_value)
            ax.set_box_aspect([1, 1, 1])

        return ax

    def __set_plot_titles(self, fig, title, x_label, y_label):
        """
        Helper function to set plot titles and labels for figures with subplots

        Parameters
        ----------
        fig: matplotlib.figure
            The figure to set the title and labels for
        title: str or None
            The title of the figure
        x_label: str or None
            The x label of the figure
        y_label: str or None
            The y label of the figure
        """
        if self.title_fntsz is not None:
            fig.suptitle(title, weight="bold", fontsize=self.title_fntsz)
        if x_label is not None:
            fig.supxlabel(x_label, fontsize=self.other_fntsz, fontweight="bold")
        if y_label is not None:
            fig.supylabel(y_label, fontsize=self.other_fntsz, fontweight="bold")
        return

    def plot_parity(self):
        """
        Makes Parity plots of validation and true data for selected methods in best
        """
        # Get Best Data Runs and iters
        df_best, job_list_best = self.analyzer.get_best_data()
        runs = df_best["Run Number"].to_list()
        iters = df_best["BO Iter"].to_list()
        GPBO_methods = df_best["BO Method"].to_list()
        ax_idxs = [self.gpbo_meth_dict[str] - 1 for str in GPBO_methods]
        # Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
        subplots_needed = len(runs)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=False, sharey=False
        )
        # Print the title and labels as appropriate
        self.__set_plot_titles(
            fig,
            None,
            "True Values, " + r"$\mathbf{y}$",
            "Predicted Values, " + r"$\mathbf{\mu}$" + "\n",
        )

        # Loop over different hyperparameters (number of subplots)
        for i, ax in enumerate(axes.flatten()):
            # Only plot data if axis is visible
            if i < subplots_needed:
                # Get the test data associated with the best job
                test_data = self.analyzer.gp_parity_data(
                    job_list_best[i], runs[i], iters[i]
                )

                # Get data from test_data
                sim_data = test_data.y_vals
                gp_mean = test_data.gp_mean
                gp_stdev = np.sqrt(abs(test_data.gp_var))

                # Plot x and y data
                ax.plot(sim_data, sim_data, color="k")
                ax.scatter(sim_data, gp_mean, color="blue", label="GP Mean")
                ax.errorbar(
                    sim_data,
                    gp_mean,
                    yerr=1.96 * gp_stdev,
                    alpha=0.3,
                    fmt="o",
                    color="blue",
                )

                RMSE = mean_squared_error(sim_data, gp_mean, squared=False)
                MAPD = mean_absolute_percentage_error(sim_data, gp_mean)
                MAE = sklearn.metrics.mean_absolute_error(sim_data, gp_mean)
                R2 = r2_score(sim_data, gp_mean)
                # Set plot details
                ax.text(
                    0.95,
                    0.05,
                    # "MAPD: " + "{:.2f}".format(MAPD) + "%",
                    "RMSE: " + "{:.2f}".format(RMSE),
                    horizontalalignment="right",  # Align text to the right
                    verticalalignment="bottom",  # Align text to the bottom
                    transform=ax.transAxes,  # Use axis coordinates (0 to 1 range)
                    fontsize=self.other_fntsz,
                )
                formatted_r2 = "{:.2f}".format(R2)
                ax.text(
                    0.95,
                    0.15,
                    f"$R^2 = {formatted_r2}$",
                    horizontalalignment="right",  # Align text to the right
                    verticalalignment="bottom",  # Align text to the bottom
                    transform=ax.transAxes,  # Use axis coordinates (0 to 1 range)
                    fontsize=self.other_fntsz,
                )
                self.__set_subplot_details(
                    ax, sim_data, gp_mean, None, None, self.method_names[ax_idxs[i]]
                )

            # Add legends and handles from last subplot that is visible
            if i == subplots_needed - 1:
                handles, labels = axes[0, -1].get_legend_handles_labels()

        # Plots legend
        # if labels:
        #     fig.legend(handles, labels, loc= "upper right", fontsize = self.other_fntsz, bbox_to_anchor=(1.0, 0.4),
        #                borderaxespad=0)

        plt.tight_layout()

        # Save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(save_path)
            save_path_to = os.path.join(save_path_dir, "parity_plots")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return

    def hist_categ_min(
        self, tot_runs, w_gpbo=True, w_gpbo_sse=False, meth_list=[3,4,5,6,7], w_noise=False
    ):
        """
        Creates objective and parameter histograms for the minima found by least squares
        """
        w_noise_str = "_w_noise" if w_noise else "_wo_noise"
        meth_list = [self.method_names[val - 1] for val in meth_list]
        if not w_gpbo:
            local_min_sets = self.analyzer.categ_min(tot_runs, w_noise=w_noise)
        else:
            local_min_sets, gpbo_runs = self.analyzer.compare_min(
                tot_runs, meth_list=meth_list
            )
        cs_name_dict = {
            key: self.analyzer.criteria_dict[key] for key in ["cs_name_val"]
        }

        add_gp = "gpbo_" if w_gpbo is True else ""
        ls_hist_fig_path = os.path.join(
            self.analyzer.make_dir_name_from_criteria(cs_name_dict),
            "ls_" + add_gp + "local_min_hist_" + str(tot_runs) + w_noise_str,
        )

        # Get the unique instacnes of theta and the counts of each instance
        unique_theta = np.vstack(local_min_sets["Theta Min Obj Cum."].values)
        theta_counts = local_min_sets["Num Occurrences"].values
        # Find the index in unique_theta closest to simulator.theta_true
        distances = np.linalg.norm(
            unique_theta - self.analyzer.simulator.theta_true, axis=1
        )
        closest_index = np.argmin(distances)

        # Get % local minima found
        percent_local_min = 100 * (local_min_sets["Num Occurrences"].iloc[0] / tot_runs)
        text_str = "NLS true min found " + f"{percent_local_min:.0f}" + " % of the time"

        if w_gpbo is True:
            theta_counts2 = local_min_sets["GPBO Matches"].values
            percent_local_min2 = 100 * (theta_counts2[0] / gpbo_runs)
            text_str2 = (
                "GPBO true min found " + f"{percent_local_min2:.0f}" + " % of the time"
            )

        # Get theta labels, bolding the one closest to theta_true
        theta_labels = np.vectorize(lambda val: f"{val:.2g}")(unique_theta)
        theta_labels = theta_labels.astype(float).tolist()
        theta_labels = [
            r"$\mathbf{" + str(label) + "}$" if i == closest_index else label
            for i, label in enumerate(theta_labels)
        ]

        # Map Theta values to indices for plotting
        theta_indices = np.arange(len(unique_theta))

        # Histogram for Theta using custom x labels
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(19, 10))

        ax[0].bar(
            theta_indices, theta_counts, alpha=0.7, edgecolor="black", label="NLS"
        )
        if w_gpbo is True:
            ax[0].bar(
                theta_indices, theta_counts2, alpha=0.7, edgecolor="black", label="GPBO"
            )
            ax[0].text(
                0.95,
                0.80,
                text_str2,
                transform=ax[0].transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="black",
                    boxstyle="round,pad=0.5",
                ),
            )
        ax[0].set_xticks(theta_indices)
        ax[0].set_xticklabels(
            theta_labels, rotation=45, ha="right"
        )  # Custom labels for x-axis
        ax[0].set_ylabel("Frequency", fontsize=20)
        ax[0].tick_params(axis="y", labelsize=14)
        ax[0].grid(axis="y", linestyle="--", alpha=0.7)
        ax[0].text(
            0.95,
            0.95,
            text_str,
            transform=ax[0].transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="black",
                boxstyle="round,pad=0.5",
            ),
        )

        # Plot for "Min Obj"
        y_str_obj = "SSE Loss Function, " + r"$\mathscr{L}(\mathbf{\theta^{\prime}})$"
        x_str_obj = "Parameter Values, " + r"$\mathbf{\theta^{\prime}}$"
        ax[1].bar(
            theta_indices, local_min_sets["Min Obj Cum."], alpha=0.7, edgecolor="black"
        )
        if w_gpbo is True and w_gpbo_sse is True:
            ax[1].bar(
                theta_indices, local_min_sets["GPBO SSE"], alpha=0.7, edgecolor="black"
            )
        ax[1].set_xticks(theta_indices)
        ax[1].set_xticklabels(
            theta_labels, rotation=45, ha="right"
        )  # Custom labels for x-axis
        ax[1].set_xlabel(x_str_obj, fontsize=20)
        ax[1].set_ylabel(y_str_obj, fontsize=20)
        ax[1].tick_params(axis="y", labelsize=14)
        ax[1].grid(axis="y", linestyle="--", alpha=0.7)
        ax[1].set_yscale("log")

        for n, axs in enumerate(ax):
            axs.text(
                0.05,
                1.05,
                "(" + string.ascii_uppercase[n] + ")",
                transform=axs.transAxes,
                size=20,
                weight="bold",
            )

        handles, labels = ax[0].get_legend_handles_labels()
        if w_gpbo is True:
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2
            )
            fig.suptitle("Local Minima Found by NLS and Emulator GPBO", fontsize=16)
        else:
            fig.suptitle("Local Minima Found by NLS", fontsize=16)
        plt.tight_layout()

        # Save or show figure
        if self.save_figs:
            save_path_to = os.path.join(ls_hist_fig_path)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        # if self.save_csv:
        # plt.savefig(ls_hist_fig_path)
        # else:
        #     plt.show()

        return

    def plot_nlr_heat_maps(
        self,
        test_mesh,
        all_z_data,
        z_titles,
        levels,
        param_info_dict,
        log_data,
        title=None,
    ):
        """
        Plots comparison of y_sim, GP_mean, and GP_stdev
        Parameters
        ----------
            test_mesh: list of ndarray of length 2, Containing all values of the parameters for the heat map x and y. Gen with np.meshgrid()
            theta_true: ndarray or None, Containing the true input parameters in all dimensions
            theta_obj_min: ndarray or None, Containing the optimal input parameters predicted by the GP
            param_names: list of str, Parameter names. Length of 2
            levels: int, list of int or None, Number of levels to skip when drawing contour lines
            idcs_to_plot: list of int, Indecies of parameters to plot
            all_z_data: list of np.ndarrays, The list of values that will be plotted. Ex. SSE, SSE_Var, EI
            z_titles: list of str, The list of the names of the values in z
            xbins: int, Number of bins for x
            ybins: int, Number of bins for y
            zbins: int, Number of bins for z
            title: str or None, Title of graph
            title_fontsize: int, fontisize for title. Default 24
            other_fontsize: int, fontisize for other values. Default 20
            save_path: str or None, Path to save figure to. Default None (do not save figure).
        Returns
        -------
            plt.show(), A heat map of test_mesh and z
        """

        # Assert Statements
        list_vars = [test_mesh, all_z_data, z_titles]
        assert all(
            isinstance(item, np.ndarray) for item in all_z_data
        ), "all_z_data elements must be np.ndarray"
        assert all(
            isinstance(item, np.ndarray) for item in test_mesh
        ), "test_mesh elements must be np.ndarray"

        # Define plot levels
        if levels is None:
            tot_lev = None
        elif len(levels) == 1:
            tot_lev = levels * len(all_z_data)
        else:
            tot_lev = levels

        assert tot_lev is None or len(tot_lev) == len(
            all_z_data
        ), "levels must be length 1, None, or len(all_z_data)"

        # Get info from param dict
        theta_true = param_info_dict["true"]
        theta_opt = param_info_dict["min_sse"]
        param_names = param_info_dict["names"]
        idcs_to_plot = param_info_dict["idcs"]

        # Assert sattements
        # Get x and y data from test_mesh
        xx, yy = test_mesh  # NxN, NxN
        assert xx.shape == yy.shape, "Test_mesh must be 2 NxN arrays"

        # Make figures and define number of subplots
        subplots_needed = len(all_z_data)
        fig, ax, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=True, sharey=True
        )

        # Find the maximum and minimum values in your data to normalize the color scale
        vmin = min(np.min(arr) for arr in all_z_data)
        vmax = max(np.max(arr) for arr in all_z_data)
        mag_diff = (
            int(math.log10(abs(vmax)) - math.log10(abs(vmin))) >= 2.0
            if vmin > 0
            else False
        )

        # Create a common color normalization for all subplots
        if log_data == True or not mag_diff or vmin < 0:
            # print(vmin, vmax)
            norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False)
            cbar_ticks = np.linspace(vmin, vmax, self.zbins)
        else:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
            cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), self.zbins)

        # Set plot details
        # Loop over number of subplots
        for i in range(subplots_needed):
            # Get method value from json file
            ax_row, ax_col = plot_mapping[i]

            z = all_z_data[i]

            # Create a colormap and colorbar for each subplot
            if log_data == True:
                cs_fig = ax[ax_row, ax_col].contourf(
                    xx,
                    yy,
                    z,
                    levels=cbar_ticks,
                    cmap=plt.cm.get_cmap(self.cmap),
                    norm=norm,
                )
            else:
                cs_fig = ax[ax_row, ax_col].contourf(
                    xx,
                    yy,
                    z,
                    levels=cbar_ticks,
                    cmap=plt.cm.get_cmap(self.cmap),
                    norm=norm,
                )

            # Create a line contour for each colormap
            if levels is not None:
                cs2_fig = ax[ax_row, ax_col].contour(
                    cs_fig,
                    levels=cs_fig.levels[:: tot_lev[i]],
                    colors="k",
                    alpha=0.7,
                    linestyles="dashed",
                    linewidths=3,
                    norm=norm,
                )
                # ax[ax_row, ax_col].clabel(cs2_fig,  levels=cs_fig.levels[::tot_lev[i]][1::2], fontsize=other_fontsize, inline=1)

            # plot min obj, max ei, true and training param values as appropriate
            if theta_true is not None:
                ax[ax_row, ax_col].scatter(
                    theta_true[idcs_to_plot[0]],
                    theta_true[idcs_to_plot[1]],
                    color="blue",
                    label="True",
                    s=200,
                    marker=(5, 1),
                    zorder=2,
                )
            if theta_opt is not None:
                for to in range(len(theta_opt)):
                    label = "Min Obj" if to == 0 else None
                    ax[ax_row, ax_col].scatter(
                        theta_opt[to][idcs_to_plot[0]],
                        theta_opt[to][idcs_to_plot[1]],
                        color="white",
                        s=150,
                        label=label,
                        marker=".",
                        edgecolor="k",
                        linewidth=0.3,
                        zorder=4,
                    )

            # Set plot details
            self.__set_subplot_details(
                ax[ax_row, ax_col], xx, yy, None, None, z_titles[i]
            )

        # Get legend information and make colorbar on last plot
        handles, labels = ax[-1, -1].get_legend_handles_labels()

        cb_ax = fig.add_axes([1.03, 0, 0.04, 1])
        if log_data is True or not mag_diff or vmin < 0:
            new_ticks = matplotlib.ticker.MaxNLocator(nbins=7)  # Set up to 7 ticks
        else:
            new_ticks = matplotlib.ticker.LogLocator(numticks=7)

        title2 = z_titles[i]

        if "theta" in param_names[0] or "tau" in param_names[0]:
            xlabel = r"$\mathbf{" + "\\" + param_names[0] + "}$"
            ylabel = r"$\mathbf{" + "\\" + param_names[1] + "}$"
        else:
            xlabel = r"$\mathbf{" + param_names[0] + "}$"
            ylabel = r"$\mathbf{" + param_names[1] + "}$"

        for axs in ax[-1]:
            axs.set_xlabel(xlabel, fontsize=self.other_fntsz)

        for axs in ax[:, 0]:
            axs.set_ylabel(ylabel, fontsize=self.other_fntsz)

        cbar = fig.colorbar(
            cs_fig, orientation="vertical", ax=ax, cax=cb_ax, ticks=new_ticks
        )
        cbar.ax.tick_params(labelsize=self.other_fntsz)
        cbar.ax.set_ylabel(
            "Function Value", fontsize=self.other_fntsz, fontweight="bold"
        )

        # Print the title
        if title is not None:
            title = title + " " + str(param_names)

        # Print the title and labels as appropriate
        # Define x and y labels
        self.__set_plot_titles(fig, title, None, None)

        # Plots legend
        if labels:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=self.other_fntsz,
                bbox_to_anchor=(-0.02, 1),
                borderaxespad=0,
            )

        plt.tight_layout()

        nlr_plot = "func_ls_compare" if theta_true is None else "sse_contour"

        # Save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(
                save_path, "heat_maps", param_names[0] + "-" + param_names[1]
            )
            save_path_to = os.path.join(save_path_dir, "least_squares")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return plt.show()


class All_CS_Plotter(Plotters):
    """
    Plotter for all case study data plots

    Methods
    -------
    __init__(analyzer, save_figs = False): Initializes the All_CS_Plotter class
    __create_subplots(num_subplots, sharex = "row", sharey = 'none'): Creates subplots based on the number of subplots needed
    __set_subplot_details(ax, xlabel, ylabel, title): Sets the subplot details
    __save_fig(save_path, ext='png', close=True): Saves the figure to a file
    make_bar_charts(mode): Makes bar charts of the best runs for each method
    """

    def __init__(self, analyzer, save_figs=False):
        """
        Parameters
        ----------
        analyzer: General_Analysis
            An instance of the General_Analysis class
        save_figs: bool, default False
            Save figures to file if True

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Asserts
        assert isinstance(save_figs, bool), "save_figs must be boolean"
        assert isinstance(
            analyzer, All_CS_Analysis
        ), "analyzer must be an instance of All_CS_Analysis"
        super().__init__(analyzer, save_figs)
        self.hatches = ["*", "\\", "\\", "o", "o", "o", "o", "o"]

        self.color_dict = {
            "Conventional": "red",
            "Log Conventional": "blue",
            "Independence": "green",
            "Log Independence": "purple",
            "Sparse Grid": "darkorange",
            "Monte Carlo": "deeppink",
            "E[SSE]": "teal",
            "NLS": "grey",
            "SHGO-Sob": "skyblue",
            "SHGO-Simp": "skyblue",
            "NM": "darkgoldenrod",
            "GA": "mediumseagreen",
        }

        self.hatch_dict = {
            "Conventional": "\\",
            "Log Conventional": "\\",
            "Independence": "o",
            "Log Independence": "o",
            "Sparse Grid": "o",
            "Monte Carlo": "o",
            "E[SSE]": "o",
            "NLS": "*",
            "SHGO-Sob": None,
            "SHGO-Simp": None,
            "NM": None,
            "GA": None,
        }

    def __create_subplots(
        self, num_subplots, sharex="row", sharey="none", row_num_size=16
    ):
        """
        Creates Subplots based on the amount of data

        Parameters
        ----------
        num_subplots: int
            Total number of needed subplots
        sharex: str, default "row"
            sharex value for subplots
        sharey: str, default "none"
            sharey value for subplots

        Returns
        -------
        fig: matplotlib.figure
            The matplotlib figure object
        axes: matplotlib.axes.Axes
            2D array of axes
        total_ax_num: int
            The number of axes generated total
        plot_mapping: dict
            Dictionary which maps plot numbers to axes

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """

        assert num_subplots >= 1, "Number of subplots must be at least 1"
        assert isinstance(num_subplots, int), "Num subplots must be int"
        # Make figures and define number of subplots
        # If you are making more than one figure, sharex is always true
        if num_subplots == 1:
            sharex = True

        # Make enough rows and columns and get close to equal number of each
        row_num = 1
        col_num = num_subplots
        assert (
            row_num * col_num >= num_subplots
        ), "row * col numbers must be at least equal to number of graphs"
        total_ax_num = row_num * col_num

        # Creat subplots
        gridspec_kw = {"wspace": 0.4, "hspace": 0.2}
        fig, axes = plt.subplots(
            row_num,
            col_num,
            figsize=(col_num * 6, row_num * row_num_size),
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
        )

        # Turn off unused axes
        for i, axs in enumerate(axes.flatten()):
            if i >= num_subplots:
                axs.axis("off")

        # Make plot mapping to map an axes to an iterable value
        plot_mapping = {}
        for i in range(row_num):
            for j in range(col_num):
                plot_number = i * col_num + j
                plot_mapping[plot_number] = (i, j)

        return fig, axes, total_ax_num, plot_mapping

    def __set_subplot_details(self, ax, xlabel, ylabel, title):
        """
        Function for setting plot settings

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axis to set the plot settings for
        plot_x:np.ndarray
            The x data for plotting
        plot_y:np.ndarray
            The y data for plotting
        xlabel: str or None
            The label for the x axis
        ylabel: str or None
            The label for the y axis
        title: str or None
            The subplot title

        Returns
        -------
        ax: matplotlib.axes.Axes
            The axis with the plot settings set

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Group inputs by type
        none_str_vars = [title, xlabel, ylabel]
        int_vars = [self.xbins, self.ybins, self.other_fntsz]

        # Assert Statements
        assert all(
            isinstance(var, str) or var is None for var in none_str_vars
        ), "title, xlabel, and ylabel must be string or None"
        assert all(
            isinstance(var, int) for var in int_vars
        ), "xbins, ybins, and fontsize must be int"
        assert all(
            var > 0 or var is None for var in int_vars
        ), "integer variables must be positive"

        # Set title, label, and axes
        if title is not None:
            pad = 6 + 4 * title.count("_")
            ax.set_title(title, fontsize=self.other_fntsz, fontweight="bold", pad=pad)
        if xlabel is not None:
            pad = 6 + 4 * xlabel.count("_")
            ax.set_xlabel(
                xlabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )
        if ylabel is not None:
            pad = 6 + 2 * ylabel.count("_")
            ax.set_ylabel(
                ylabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )

        # Turn on tick parameters and bin number
        ax.xaxis.set_tick_params(labelsize=self.other_fntsz, direction="in")
        ax.yaxis.set_tick_params(labelsize=self.other_fntsz, direction="in")
        ax.minorticks_on()  # turn on minor ticks
        ax.tick_params(which="minor", direction="in", top=True, right=True)
        ax.locator_params(axis="x", nbins=self.xbins)

        return ax

    def __save_fig(self, save_path, ext="png", close=True):
        """Save a figure from pyplot.
        Parameters
        ----------
        save_path : string
            The path (and filename, without the extension) to save the
            figure to.
        ext : string (default='png')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
        close : boolean (default=True)
            Whether to close the figure after saving.  If you want to save
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.
        """

        # Extract the directory and filename from the given path
        directory = os.path.split(save_path)[0]
        filename = "%s.%s" % (os.path.split(save_path)[1], ext)
        if directory == "":
            directory = "."

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # The final path to save to
        savepath = os.path.join(directory, filename)

        # Actually save the figure
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

        # Close it
        if close:
            plt.close()

    def make_bar_charts(self, mode):
        """
        Makes a bar chart of relevant data for each method and case study. Produces Figures 2 and 7 in the paper

        Parameters:
        -----------
        mode: str
            Whether to make a bar chart for the objective or time information for the case studies

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Options for mode are "objs" or "time"
        """
        assert mode in [
            "time",
            "objs",
            "si_time",
        ], "mode must be 'time', 'objs', or 'si_time'"
        # Make figures and define number of subplots (3. One for comp time, one for fxn evals, one for g(\theta))
        t_label_lst = [
            get_cs_class_from_val(cs_num).name for cs_num in self.analyzer.cs_list
        ]

        if mode == "time":
            add_value = 1
            fig, axes, num_subplots, plot_mapping = self.__create_subplots(
                2, sharex=False, sharey=True
            )
            # fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            #     3, sharex=False, sharey=True
            # )
        elif mode == "si_time":
            add_value = 1
            # fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            #     2, sharex=False, sharey=True
            # )
            fig, axes, num_subplots, plot_mapping = self.__create_subplots(
                1, sharex=False, sharey=True
            )
        else:
            add_value = 1
            fig, axes, num_subplots, plot_mapping = self.__create_subplots(
                4, sharex=False, sharey=True
            )

        # bar_size = 1 / (len(self.analyzer.cs_list) + add_value)
        # padding = 1 / (len(self.analyzer.cs_list))
        height_per_group = 1 / len(self.analyzer.cs_list)
        bars_per_group = len(self.method_names) + add_value
        bar_size = height_per_group / bars_per_group
        padding = bar_size

        # Get jobs associated with the case studies given
        df_averages = self.analyzer.get_averages_best(other_meths=["NLS", "GA"])

        desired_order = [
            "Simple Linear",
            "Large Linear",
            "Yield-Loss",
            "Log Logistic",
            "2D Log Logistic",
            "Simple Multimodal",
            "Muller y0",
            "Muller x0",
            "ACN-Water",
            "BOD Curve",
        ]
        # Convert the 'Department' column to a categorical type with the specified order
        df_averages["CS Name"] = pd.Categorical(
            df_averages["CS Name"], categories=desired_order, ordered=True
        )
        df_averages["BO Method"] = pd.Categorical(
            df_averages["BO Method"],
            categories=["NLS", "GA"] + self.method_names[::-1],
            ordered=True,
        )

        # Sort the DataFrame by the 'Department' column
        df_averages = df_averages.sort_values(["CS Name", "BO Method"])

        # print(df_averages.head())
        def calculate_new_column(group, meth_comp_time="NLS"):
            # Calculate Avg Evals for NLS in the current group
            nls_avg_evals = group.loc[
                group["BO Method"] == meth_comp_time, "Avg F Evals Tot"
            ].values

            nls_std_evals = group.loc[
                group["BO Method"] == meth_comp_time, "Std F Evals Tot"
            ].values

            nls_avg_L_evals = group.loc[
                group["BO Method"] == meth_comp_time, "Avg Evals Tot"
            ].values

            nls_std_L_evals = group.loc[
                group["BO Method"] == meth_comp_time, "Std Evals Tot"
            ].values

            nls_avg_time = group.loc[
                group["BO Method"] == meth_comp_time, "Avg Time"
            ].values

            nls_std_time = group.loc[
                group["BO Method"] == meth_comp_time, "Std Time"
            ].values

            # Calculate the new column
            group["D"] = nls_avg_evals - group["Avg F Evals Tot"]
            group["Std D"] = np.sqrt(nls_std_evals**2 + group["Std F Evals Tot"] ** 2)

            group["A"] = (group["Avg Time"] - nls_avg_time)
            group["Std A"] = np.sqrt(nls_std_time**2 + group["Std Time"] ** 2)

            # Calculate the new column (Time Parity in minutes)
            group["F_Time_Parity"] = (group["A"] / 60) / group["D"]

            # Calculate the uncertainty in the new column
            group["F_Par_std"] = group["F_Time_Parity"] * np.sqrt(
                (group["Std A"]/ group["A"]) ** 2
                + (group["Std D"] / group["D"]) ** 2
            )
            group = group.drop(columns=["Std D", "D", "Std A", "A"])

            # group["L_deficit"] = group["Avg Evals Tot"] - nls_avg_L_evals
            # group["L_deficit_std"] = np.sqrt(
            #     nls_std_L_evals**2 + group["Std Evals Tot"] ** 2
            # )

            return group
        
        def calc_deficit_col(group, meth_comp_time="NLS"):
            # Calculate Avg Evals for NLS in the current group

            nls_avg_L_evals = group.loc[
                group["BO Method"] == meth_comp_time, "Avg Evals Tot"
            ].values

            nls_std_L_evals = group.loc[
                group["BO Method"] == meth_comp_time, "Std Evals Tot"
            ].values


            group["L_deficit_NLS"] = group["Avg Evals Tot"] - nls_avg_L_evals
            group["L_deficit_NLS_std"] = np.sqrt(
                nls_std_L_evals**2 + group["Std Evals Tot"] ** 2
            )

            return group

        # Apply the calculation for each group
        df_averages = df_averages.groupby("CS Name", group_keys=False).apply(
            calculate_new_column, ("GA")
        )
        df_averages = df_averages.groupby("CS Name", group_keys=False).apply(
            calc_deficit_col, ("NLS")
        )

        df_averages.loc[df_averages["L_deficit_NLS"] <= 0, "L_deficit_NLS_std"] = 0

        if mode == "objs":
            names = ["Median Loss", "Avg Evals", "Avg Evals Tot", "Avg Opt Acq"]
            std_names = ["IQR Loss", "Std Evals", "Std Evals Tot", "Std Opt Acq"]
            titles = [
                "Median "
                + r"$\mathscr{L}(\mathbf{\theta}^{\prime})$"
                + " \n at Termination",
                "Avg. "
                + r"$\mathscr{L}(\cdot)$"
                + " Evaluations \n to Reach "
                + r"$\mathscr{L}(\mathbf{\theta}^{\prime})$",
                "Total " + r"$\mathscr{L}(\cdot)$" + " Evalulations",
                "Avg. " + r"$\Xi(\mathbf{\theta}^*)$" + "\n Last 10 Iterartions",
            ]
        elif mode == "si_time":
            names = ["L_deficit_NLS"]
            std_names = ["L_deficit_NLS_std"]
            titles = [
                "Estimated "
                + r"$\mathscr{L}(\cdot)$"
                + " Evaluation Deficit \n for Parity",
            ]
        else:
            names = ["Avg Time", "F_Time_Parity"]
            std_names = ["Std Time", "F_Par_std"]
            titles = [
                "Avg. Run Time (min)",
                "Estimated " + r"$f(\cdot)$" + " Cost \n for Parity (min)",
            ]
            # names = ["Avg Time", "F_Time_Parity", "L_deficit"]
            # std_names = ["Std Time", "F_Par_std", "L_deficit_std"]
            # titles = [
            #     "Avg. Run Time (min)",
            #     "Estimated " + r"$f(\cdot)$" + " Cost \n for Parity (min)",
            #     "Estimated " + r"$\mathscr{L}(\cdot)$" + " Deficit \n for Parity",
            # ]

        t_label_lst = list(df_averages["CS Name"].unique())
        t_label_lst = [item.replace("Muller", "Müller") for item in t_label_lst]
        t_label_lst = [
            item.replace("ACN-Water", "ACN-Water VLE") for item in t_label_lst
        ]

        # y_locs = np.arange(len(self.analyzer.cs_list)) * (
        #     bar_size * (len(self.analyzer.cs_list)) + padding
        # )
        y_locs = np.arange(len(self.analyzer.cs_list)) * (height_per_group + padding)

        axes = axes.flatten()
        for i in range(len(self.analyzer.meth_val_list) + add_value):
            if i < len(self.analyzer.meth_val_list):
                # loop over methods n reverse order
                meth_val = self.analyzer.meth_val_list[-1 - i]
                meth_averages = df_averages.loc[
                    df_averages["BO Method"] == self.method_names[meth_val - 1]
                ]
                label = self.method_names[meth_val - 1]
                color = self.colors[meth_val - 1]
            else:
                if mode in ["objs", "si_time"]:
                    label = "NLS"
                else:
                    label = "GA"
                meth_averages = df_averages.loc[df_averages["BO Method"] == label]
                color = self.color_dict[label]

            for j in range(len(names)):
                scl_value = 60 if names[j] == "Avg Time" else 1
                avg_val = meth_averages[names[j]] / scl_value
                std_val = meth_averages[std_names[j]] / scl_value
                avg_val = np.maximum(avg_val, 0)
                std_val = np.maximum(std_val, 0)
                hatch = self.hatches[-1 - i] if label != "GA" else None
                color = "mediumseagreen" if label == "GA" else color
                rects = axes[j].barh(
                    y_locs + i * bar_size,
                    avg_val,
                    xerr=std_val,
                    align="center",
                    height=bar_size,
                    color=color,
                    label=label,
                    hatch=hatch,
                )

                if i == 0:
                    # Set plot details on last iter
                    height_per_group = (
                        bar_size * (len(self.method_names) + add_value) + padding
                    )  # Total height of one group, including padding
                    tick_positions = y_locs + (
                        height_per_group / 2
                    )  # y_locs + padding * len(self.analyzer.cs_list) / 2
                    # Set plot details on last iter
                    self.__set_subplot_details(axes[j], None, None, titles[j])
                    axes[j].set(
                        yticks=tick_positions,
                        yticklabels=t_label_lst,
                        ylim=[0 - padding, len(y_locs)],
                    )
                    axes[j].set_ylim([0 - padding, len(tick_positions)])
                    axes[j].set_ylim(
                        axes[j].get_ylim()[0],
                        max(tick_positions) + height_per_group / 2,
                    )

        if mode == "time":
            axes[0].set_xscale("log")
            axes[1].set_xscale("log")
        elif mode == "si_time":
            axes[0].set_xlim(left=0)
        else:
            axes[0].set_xscale("log")
            axes[3].set_xscale("log")

        for n, ax in enumerate(axes):
            ax.grid()
            if len(axes) > 1:
                ax.text(
                    -0.1,
                    1.05,
                    "(" + string.ascii_uppercase[n] + ")",
                    transform=ax.transAxes,
                    size=20,
                    weight="bold",
                )

        # Add legends and handles from last subplot that is visible
        handles, labels = axes[-1].get_legend_handles_labels()

        # Plots legend
        if labels:
            if "time" in mode:
                fig.legend(
                    reversed(handles),
                    reversed(labels),
                    loc="upper left",
                    ncol=1,
                    fontsize=self.other_fntsz,
                    bbox_to_anchor=(1.05, 0.95),
                    borderaxespad=0,
                )
            else:
                fig.legend(
                    reversed(handles),
                    reversed(labels),
                    loc="upper center",
                    ncol=4,
                    fontsize=self.other_fntsz,
                    bbox_to_anchor=(0.55, 1.10),
                    borderaxespad=0,
                )

        plt.tight_layout()

        # Save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(save_path)
            save_path_to = os.path.join(save_path_dir, str(mode) + "_bar")
            df_averages.to_csv(save_path_to + ".csv", index=False)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        # return df_averages
        return df_averages

    def make_derivfree_bar(self, s_meths=["NLS", "SHGO-Sob", "NM", "GA"], ver="med"):
        """
        Makes a bar chart of relevant data for each method and case study. Produces Figures 2 and 7 in the paper

        Parameters:
        -----------
        mode: str
            Whether to make a bar chart for the objective or time information for the case studies

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        Options for mode are "objs" or "time"
        """
        # Make figures and define number of subplots (3. One for SSE, one for L2 Norm, and one for (total) fxn evals)
        t_label_lst = [
            get_cs_class_from_val(cs_num).name for cs_num in self.analyzer.cs_list
        ]

        add_value = 1
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            3, sharex=False, sharey=True, row_num_size=16
        )

        # Bar sizing = number of case studies + padding
        height_per_group = 1 / len(self.analyzer.cs_list)
        bars_per_group = len(s_meths) + add_value
        bar_size = height_per_group / bars_per_group
        padding = bar_size

        # Get jobs associated with the case studies given
        df_averages = self.analyzer.get_averages_best(s_meths)
        df_bests = self.analyzer.get_all_meths_best(s_meths)
        desired_order = [
            "Simple Linear",
            "Large Linear",
            "Yield-Loss",
            "Log Logistic",
            "2D Log Logistic",
            "Simple Multimodal",
            "Muller y0",
            "Muller x0",
            "ACN-Water",
            "BOD Curve",
        ]
        for df in [df_averages, df_bests]:
            # Convert the 'CS Name' column to a categorical type with the specified order
            df["CS Name"] = pd.Categorical(
                df["CS Name"], categories=desired_order, ordered=True
            )
            df["BO Method"] = pd.Categorical(
                df["BO Method"],
                categories=s_meths + self.method_names[::-1],
                ordered=True,
            )

            # Sort the DataFrame by the 'CS Name' column
            df = df.sort_values(["CS Name", "BO Method"], inplace=True)

        names = ["Median Loss", "Avg Evals Tot", "Median L2 Norm"]
        std_names = ["IQR Loss", "Std Evals Tot", "IQR L2 Norm"]
        best_names = ["Best Loss", "Max Evals", "Best L2 Norm"]
        titles = [
            "Median "
            + r"$\mathscr{L}(\mathbf{\theta}^{\prime})$"
            + " \n at Termination",
            "Total " + r"$\mathscr{L}(\cdot)$" + " Evalulations",
            "Median " + r"$L_2-Norm$" + " \n at Termination",
        ]

        t_label_lst = list(df_averages["CS Name"].unique())
        t_label_lst = [item.replace("Muller", "Müller") for item in t_label_lst]
        t_label_lst = [
            item.replace("ACN-Water", "ACN-Water VLE") for item in t_label_lst
        ]

        y_locs = np.arange(len(self.analyzer.cs_list)) * (height_per_group + padding)
        axes = axes.flatten()
        added_labels = set()

        cs_name_to_constant = {
            "Simple Linear": 20,
            "Simple Multimodal": 20,
            "ACN-Water": 20,
            "Large Linear": 50,
            "Muller x0": 40,
            "Muller y0": 20,
            "Log Logistic": 40,
            "2D Log Logistic": 40,
            "Yield-Loss": 30,
            "BOD Curve": 20,
        }

        for i in range(len(s_meths) + 1):
            if i == 0:
                # Make a DF including just GPBO info
                filtered_df = df_averages[
                    df_averages["BO Method"].isin(self.method_names)
                ]
                # Get only the values of the BO Method where Median Loss is the minimum
                #     meth_averages = (
                #     filtered_df.loc[filtered_df.groupby("CS Name")["Median Loss"].idxmin()]
                # )
                meth_averages = filtered_df.loc[
                    filtered_df.groupby("CS Name")["Median Loss"]
                    .apply(lambda x: x.idxmin() if not x.isna().all() else None)
                    .dropna()
                ]

                # Find index of the method in the method names list
                meth_best = df_bests.loc[~df_bests["BO Method"].isin(s_meths)]

                # Add number of fxn evals to each method
                # meth_best.loc[:,"Max Evals"] = meth_best["CS Name"].map(cs_name_to_constant) + meth_best["Max Evals"]
                # meth_averages.loc[:,"Avg Evals Tot"] = meth_averages["CS Name"].map(cs_name_to_constant) + meth_averages["Avg Evals Tot"]
            else:
                meth_averages = df_averages.loc[
                    df_averages["BO Method"] == s_meths[i - 1]
                ]
                meth_best = df_bests.loc[df_bests["BO Method"] == s_meths[i - 1]]
                label = s_meths[i - 1]

            for j in range(len(names)):
                scl_value = 60 if names[j] == "Avg Time" else 1
                best_val = meth_best[best_names[j]] / scl_value
                alpha = 1 if ver == "med" else 0.5
                for idx in range(len(meth_averages["BO Method"])):
                    label_med = meth_averages["BO Method"].iloc[idx]
                    avg_val = meth_averages[names[j]].iloc[idx] / scl_value
                    std_val = meth_averages[std_names[j]].iloc[idx] / scl_value
                    # avg_val = meth_averages[names[j]] / scl_value
                    # std_val = meth_averages[std_names[j]] / scl_value
                    avg_val = np.maximum(avg_val, 0)
                    std_val = np.maximum(std_val, 0)
                    if label_med not in self.method_names and idx == 0:
                        label_m_use = label_med + " (Median)"
                    else:
                        label_m_use = None
                        added_labels.add(label_med)

                    axes[j].barh(
                        y_locs[idx] + i * bar_size,
                        avg_val,
                        xerr=std_val,
                        align="center",
                        height=bar_size,
                        label=label_m_use,
                        color=self.color_dict[label_med],
                        hatch=self.hatch_dict[label_med],
                        alpha=alpha,
                    )
                if ver == "best-med":
                    for idx, val in enumerate(best_val):
                        label_best = meth_best["BO Method"].iloc[idx]
                        if label_best not in self.method_names and idx == 0:
                            label_b_use = label_best + " (Best)"
                        else:
                            label_b_use = None
                            added_labels.add(label_best)
                        axes[j].barh(
                            y_locs[idx] + i * bar_size,
                            val,
                            align="center",
                            height=bar_size,
                            label=label_b_use,
                            color=self.color_dict[label_best],
                            hatch=self.hatch_dict[label_best],
                        )

                # Add in the best performace with an alpha value of 1. Change median/avg values to alpha value of 0.5

                if i == 0:
                    # Set plot details on last iter
                    height_per_group = (
                        bar_size * (len(s_meths) + add_value) + padding
                    )  # Total height of one group, including padding
                    tick_positions = y_locs + (
                        height_per_group / 2
                    )  # Center of each group
                    self.__set_subplot_details(axes[j], None, None, titles[j])
                    axes[j].set(
                        yticks=tick_positions,
                        yticklabels=t_label_lst,
                        ylim=[0 - padding, len(y_locs)],
                    )
                    axes[j].set_ylim([0 - padding, len(tick_positions)])
                    axes[j].set_ylim(
                        axes[j].get_ylim()[0],
                        max(tick_positions) + height_per_group / 2,
                    )

        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[2].set_xscale("log")

        for n, ax in enumerate(axes):
            ax.grid()
            if len(axes) > 1:
                ax.text(
                    -0.1,
                    1.05,
                    "(" + string.ascii_uppercase[n] + ")",
                    transform=ax.transAxes,
                    size=20,
                    weight="bold",
                )

        # Add legends and handles from last subplot that is visible
        class MulticolorPatch(object):
            def __init__(self, cmap, edgecolor=None, linewidth=None, ncolors=100):
                self.ncolors = ncolors
                self.edgecolor = edgecolor
                self.linewidth = linewidth

                if isinstance(cmap, str):
                    self.cmap = plt.get_cmap(cmap)
                else:
                    self.cmap = cmap

        # define a handler for the MulticolorPatch object
        class MulticolorPatchHandler(object):
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                n = orig_handle.ncolors
                width, height = handlebox.width, handlebox.height
                patches = []

                for i, c in enumerate(orig_handle.cmap(i / n) for i in range(n)):
                    patches.append(
                        plt.Rectangle(
                            [width / n * i - handlebox.xdescent, -handlebox.ydescent],
                            width / n,
                            height,
                            facecolor=c,
                            edgecolor=orig_handle.edgecolor,
                            linewidth=orig_handle.linewidth,
                        )
                    )

                patch = PatchCollection(patches, match_original=True)

                handlebox.add_artist(patch)

                if orig_handle.edgecolor is not None:
                    linestyle = (0, (5, 5))  # Dashed line style
                else:
                    linestyle = "solid"

                border_rect = plt.Rectangle(
                    [
                        handlebox.xdescent - 1,
                        -handlebox.ydescent - 1,
                    ],  # Position slightly offset
                    width + 2,  # Width slightly larger to fit around the patches
                    height + 2,  # Height slightly larger to fit around the patches
                    edgecolor="black",  # Border color
                    linestyle=linestyle,  # Dashed border style
                    linewidth=1,  # Border line width
                    fill=False,
                )  # No fill color

                # Add the border rectangle to handlebox
                handlebox.add_artist(border_rect)

                return patch

        # Add a dummy legend
        handles, labels = axes[0].get_legend_handles_labels()
        if ver == "med":
            handles_extra = [
                MulticolorPatch("gist_rainbow"),
            ]
            labels_extra = ["GPBO (Median)"]
        else:
            handles_extra = [
                MulticolorPatch("gist_rainbow", edgecolor="white", linewidth=0.25),
                MulticolorPatch("gist_rainbow"),
            ]
            labels_extra = ["GPBO (Median)", "GPBO (Best)"]
        handles += handles_extra
        labels += labels_extra

        # Plots legend
        if labels:
            fig.legend(
                reversed(handles),
                reversed(labels),
                loc="upper center",
                ncol=3,
                fontsize=self.other_fntsz,
                bbox_to_anchor=(0.55, 1.10),
                borderaxespad=0,
                handler_map={MulticolorPatch: MulticolorPatchHandler()},
            )
        plt.tight_layout()

        # Save or show figure
        if self.save_figs:
            save_path = self.analyzer.make_dir_name_from_criteria(
                self.analyzer.criteria_dict
            )
            save_path_dir = os.path.join(save_path)
            combined_string = "_".join(s_meths)
            save_path_to = os.path.join(save_path_dir, combined_string + "_bar-" + ver)
            df_averages.to_csv(save_path_to + ".csv", index=False)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        # return df_averages
        return df_averages, df_bests
