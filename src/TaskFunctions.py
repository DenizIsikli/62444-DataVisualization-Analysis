import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TaskFunctions_1:
    def placeholder(self):
        pass


class TaskFunctions_2:
    def placeholder(self):
        pass


if __name__ == '__main__':
    tf_1 = TaskFunctions_1()
    tf_2 = TaskFunctions_2()

# Task 4
# Process given data into a time based format from a sample group
def process_data_temporal(df, date_column):
    # note how many samples you want to take
    df_sample = df.sample(30000)  # note how many samples you want to take
    # convert date data into hour, days of the week and months.
    df_sample["hour"] = df_sample[date_column].dt.hour
    df_sample["day"] = df_sample[date_column].dt.weekday
    df_sample["month"] = df_sample[date_column].dt.month

    return df_sample


# Create a bar plot based on an x and y value input.
def plot_histogram_2_values(ax, x, y, color, title, x_label, y_label, month_names=None):
    if month_names:
        x_labels = [month_names[val] for val in x]
    else:
        x_labels = x

    ax.bar(x, y, color=color, alpha=0.7)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, ha='right', horizontalalignment="center", verticalalignment="top")

    # Remove top and right border for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Simplify the y-axis to show fewer tick marks
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


# Create a line plot based on two different value inputs.
def plot_line_comparison(x_yellow, y_yellow, x_green, y_green, x_label, y_label, title, month_names=None):
    plt.figure(figsize=(14, 4))
    plt.plot(x_yellow, y_yellow, label="Yellow Taxi", marker='o', color="yellow")
    plt.plot(x_green, y_green, label="Green Taxi", marker='o', color="green")

    if month_names:
        plt.xticks(x_yellow, [month_names[val] for val in x_yellow])  # Set the x-axis ticks to be month names

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)