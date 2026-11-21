
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


class VisualiseMetrics:
    def __init__(self, display="inline"):
        plt.ion()
        self._figures = {}  # metric name -> (fig, ax), returns all at once
        self._metric_to_values = {}  # metric name -> values
        self.display = display # Whether to print figures inline or to file.

    def plot_metric_over_time(self, metric_values, title, ylabel=None):

        """Plot a metric (e.g., accuracy) across chunks

        Parameters:
            metric_values: metric values provided by the chunk.
            title: the metric to plot
            ylabel: Y-axis label

        Returns:
            A plotted graph of the desired metric

        Raises:
            ValueError: if no metrics are given to plot
            TypeError: an incorrect metric type is provided
                        or an incorrect ylabel.

        Time Complexity:

        Space Complexity:


        """

        if title not in self._figures or not plt.fignum_exists(self._figures[title][0].number):
            self._figures[title] = plt.subplots()

        fig, ax = self._figures[title]
        ax.clear()

        if title not in self._metric_to_values:
            self._metric_to_values[title] = []
        self._metric_to_values[title].append(float(metric_values))

        y = np.asarray(self._metric_to_values[title])

        ax.plot(np.arange(y.size), y, c="green", linewidth=3, label=title, linestyle="solid", marker="o",
                                 markerfacecolor="orange")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # make the x-axis in integers ONLY
        ax.set_xlabel("Chunk")
        ax.set_ylabel(ylabel or title)
        ax.set_title(f"{title} over time")
        ax.legend()
        return fig


    def compare_models(self, metric1, metric2, labels):

        """Compare two models on streaming metrics (e.g., trees with different `buffer_size`
        values)

        Parameters:
            metric1 & 2: The two models to compare
            labels: Selected y-axis labels.
        """
        if "compare_models" not in self._figures or not plt.fignum_exists(self._figures["compare_models"][0].number):
            self._figures["compare_models"] = plt.subplots()

        fig, ax = self._figures["compare_models"]
        ax.clear()



    def plot_predictions_vs_ground_truth(y_true, y_pred):
        """Visualise predictions vs. actuals on latest chunk

        Parameters:
            y_true -- _description_
            y_pred -- _description_
        """
        pass

    def finish(self):
        """Finish up plotting and display the plot
            as either inline or printed to file

        Returns:
            a plot displayed inline or printed to file
        Raises:
            TypeError / ValueError: if an incorrect display option is given
        """
        plt.ioff()
        if self.display == "inline":
            plt.show()
        elif self.display == "print":
            plt.savefig("fig.png")
        else:
         raise ValueError("Invalid display option entered")
