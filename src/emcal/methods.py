"""GPBOMethod: descriptor of a GPBO method's properties (emulator vs objective GP,
log-scaling, sparse-grid/Monte-Carlo EI), all derived from its MethodName.
"""
from .enums import MethodName


class GPBOMethod:
    """
    Describes a GPBO method's properties, all derived from its MethodName.

    Attributes
    ----------
    method_name : MethodName
        The method identifier.
    log_scaled : bool
        True only for LOG_CONVENTIONAL (B1); see the note in __init__.
    is_emulator : bool (property)
        Whether the GP emulates the model output directly (Type-2 methods).
    report_name : str (property)
        The manuscript's shorthand name for this method.
    uses_sparse_grid, uses_monte_carlo : bool (property)
        How the expected-improvement integral is evaluated, if applicable.
    """

    def __init__(self, method_name):
        """
        Parameters
        ----------
        method_name: MethodName Class instance, The name associated with the method being tested. Enum type

        Raises
        ------
        AssertionError
            If method_name is not an instance of MethodName
        """
        assert isinstance(
            method_name, MethodName
        ), "method_name must be an instance of MethodName"
        self.method_name = method_name
        # log_scaled marks the conventional method whose objective is ln(SSE)
        # (LOG_CONVENTIONAL / B1). The emulator log method (LOG_INDEPENDENCE / B2)
        # applies log scaling inside the acquisition, not via this flag, so it is
        # intentionally False here.
        self.log_scaled = self.method_name == MethodName.LOG_CONVENTIONAL

    @property
    def is_emulator(self):
        """
        Whether the GP emulates the model output directly (Type-2 methods).

        Returns
        -------
        bool
        """
        # The conventional methods fit the GP to the SSE objective; all others emulate.
        return self.method_name not in (
            MethodName.CONVENTIONAL,
            MethodName.LOG_CONVENTIONAL,
        )

    @property
    def report_name(self):
        """
        The manuscript's shorthand name for this method.

        Returns
        -------
        str
        """
        report_names = {
            MethodName.CONVENTIONAL: "Conventional",
            MethodName.LOG_CONVENTIONAL: "Log Conventional",
            MethodName.INDEPENDENCE: "Independence",
            MethodName.LOG_INDEPENDENCE: "Log Independence",
            MethodName.SPARSE_GRID: "Sparse Grid",
            MethodName.MONTE_CARLO: "Monte Carlo",
            MethodName.EXPECTED_SSE: "E[SSE]",
        }
        return report_names[self.method_name]

    @property
    def uses_sparse_grid(self):
        """
        Whether EI is evaluated with a sparse-grid integral (SPARSE_GRID / C2).

        Returns
        -------
        bool
        """
        return self.method_name == MethodName.SPARSE_GRID

    @property
    def uses_monte_carlo(self):
        """
        Whether EI is evaluated with a Monte-Carlo integral (MONTE_CARLO / D2).

        Returns
        -------
        bool
        """
        return self.method_name == MethodName.MONTE_CARLO
