
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
            O(N)
        Space Complexity:
            O(N)

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


    def compare_models(self, metric1, metric2, labels, metric_name):

        """Compare two models on streaming metrics (e.g., trees with different `buffer_size`
        values)

        Parameters:
            metric1 & 2: The two models to compare
            labels: Selected y-axis labels.
            metric_name: The particular metric the two models are tested against

        Returns:
        A plotted line graph with two lines representing each model, and measuring
        the given metric against each other

        Raises:
            ValueError: if either metric1, metric2, or metric_name are not given.
            TypeError: if metric_name is given as a string

        Time Complexity:
            O(N)
        Space Complexity:
            O(N)

        """
        if metric1 is None or metric2 is None or metric_name is None:
            raise TypeError("metric1, metric2, and metric_name are empty!")

        if "compare_models" not in self._figures or not plt.fignum_exists(self._figures["compare_models"][0].number):
            self._figures["compare_models"] = plt.subplots()

        fig, ax = self._figures["compare_models"]
        ax.clear()
        m1 = np.asarray(metric1)
        m2 = np.asarray(metric2)

        ax.plot(np.arange(m1.size), m1, c="green", linewidth=3, label=labels[0], linestyle="solid", marker="o",)
        ax.plot(np.arange(m2.size), m2, c="red", linewidth=3, label=labels[1], linestyle="solid", marker="o",)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # make the x-axis in integers ONLY
        ax.set_xlabel("Chunk")
        ax.set_ylabel("Metric")
        ax.set_title(f"{metric_name} over time for  {labels[0]} and {labels[1]}")
        ax.legend()

        return fig




    def plot_predictions_vs_ground_truth(self, y_true, y_pred):
        """Visualise predictions vs. actuals on latest chunk, perfect predictions lie on the line
        whereas less accurate ones deviate from the dotted line

        Parameters:
            y_true -- _description_
            y_pred -- _description_

         Returns:
            A scatterplot line comparing Predicted vs Actual values.

        Raises:
            TypeError: if y_true or y_pred are meaningless inputs
            ValueError: if y_true or y_pred are not the same shape.

        Time Complexity:
            O(N)
        Space Complexity:
            O(N)

        """

        if y_true is None or y_pred is None:
            raise TypeError("y_true or y_pred are empty!")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape!")

        # Initialize axes and figure
        if "pred_vs_true" not in self._figures or not plt.fignum_exists(self._figures["pred_vs_true"][0].number):
            self._figures["pred_vs_true"] = plt.subplots()

        fig, ax = self._figures["pred_vs_true"]
        ax.clear()

        # Assign predicted values and true values to an array to be plotted.
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        # Creating a trend line
        z = np.polyfit(y_true, y_pred, 1)
        ax.scatter(y_true, y_pred, c="red", edgecolors="k")

        # ax.plot(y_true, c="green", linestyle="none", label="Ground truth", marker="o", alpha=0.5)
        # ax.plot(y_pred, c="red", linestyle="none", label="Prediction", marker="o", alpha=0.5)
        low = min(y_true.min(), y_pred.min())
        high = max(y_true.max(), y_pred.max())
        ax.plot([low, high], [low, high], color="green", linestyle="--", label="Perfect Fit")
        ax.set_xlabel("Actual values")
        ax.set_ylabel("Predicted values")
        ax.set_title("Predicted vs Actual values on latest chunk of data")
        ax.legend()

        return fig



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
