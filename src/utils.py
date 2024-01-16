import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px # Used for interactive visualizations


def create_scatterplot(df, x_col, y_col, title, xlabel, ylabel):
    """
    This function creates a scatter plot with a linear regression line from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_col (str): The column in the DataFrame to use for the x-axis.
    y_col (str): The column in the DataFrame to use for the y-axis.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """

    # Create the plot
    plt.figure(figsize=(7, 7))
    sns.regplot(x=df[x_col], y=df[y_col], scatter_kws={"alpha": 0.3})

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()


def get_a_random_chunk_property(data):
    """
    This function only serves an example of fetching some of the properties
    from the data.
    Indeed, all the content in "data" may be useful for your project!
    """

    chunk_index = np.random.choice(len(data))

    date_list = list(data[chunk_index]["near_earth_objects"].keys())

    date = np.random.choice(date_list)

    objects_data = data[chunk_index]["near_earth_objects"][date]

    object_index = np.random.choice(len(objects_data))

    object = objects_data[object_index]

    properties = list(object.keys())
    property = np.random.choice(properties)

    print("date:", date)
    print("NEO name:", object["name"])
    print(f"{property}:", object[property])


def load_data_from_google_drive(url):
    url_processed='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url_processed)
    return df


# Function to remove outliers based on the  Interquartile Range (IQR) method for a given column
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.75 * IQR #3.75 was found to fit best for the entire dataset
    upper_bound = Q3 + 3.75 * IQR #3.75 was found to fit best for the entire dataset
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Function to display histograms for a given column
def plot_histogram(df, column, color, title, ax, x_label, y_label, bins=None):
    # Determine the number of bins based on the data's range and a reasonable bin width
    if bins is None: #If bins parameter isn't specified, calculate bin width
        bin_width = (df[column].max() - df[column].min()) / 100  # adjust int as necessary for bin size
        bins = np.arange(df[column].min(), df[column].max() + bin_width, bin_width)
    # If bins is an integer, it will define the number of bins directly
    # If bins is a sequence, it will define the bin edges directly

    ax.hist(df[column], bins=bins, color=color, alpha=0.7)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Remove top and right border for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Simplify the y-axis to show fewer tick marks
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


# Function to create a scatter plot for two given columns
def plot_scatter(df, x_column, y_column, color, title, ax, x_label, y_label):
    ax.scatter(df[x_column], df[y_column], alpha=0.5, color=color, s=1)

    # Trend line
    z = np.polyfit(df[x_column], df[y_column], 1)
    p = np.poly1d(z)
    ax.plot(df[x_column], p(df[x_column]), color="red", linewidth=2, alpha=0.5, label=f'Linear trend line (y={z[0]:.2f}x+{z[1]:.2f})')

     # Enhance readability
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Remove gridlines and box border for a cleaner look
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')

    # Include a legend that explains the trend line
    ax.legend()


def setUpDataToKepler(df_clean, df_zones):
    # Add the lat and lng from PULocationID
    df_taxi = pd.merge(df_clean, df_zones[['LocationID', 'lat', 'lng']], left_on='PULocationID', right_on='LocationID',
                       how='left')

    # Rename lat and lng so we can add then again for DOLocationID
    df_taxi.rename(columns={'lat': 'lat_PU', 'lng': 'lng_PU'}, inplace=True)

    # Add the lat and lng from DOLocationID
    df_taxi = pd.merge(df_taxi, df_zones[['LocationID', 'lat', 'lng']], left_on='DOLocationID', right_on='LocationID',
                       how='left')

    # Get remove duplicate rows and get an count on how many duplicates there was
    df_taxi_series = df_taxi.value_counts()
    df_taxi = df_taxi_series.reset_index()
    df_taxi.columns = [*df_taxi.columns[:-1], 'count']

    return df_taxi

def getKeplerconfig():
    config = {'version': 'v1',
              'config': {'visState': {'filters': [],
                                      'layers': [{'id': 'sr4d85f',
                                                  'type': 'line',
                                                  'config': {'dataId': 'yellow_taxi',
                                                             'label': 'Yellow taxi',
                                                             'color': [137, 218, 193],
                                                             'highlightColor': [252, 242, 26, 255],
                                                             'columns': {'lat0': 'lat_PU',
                                                                         'lng0': 'lng_PU',
                                                                         'lat1': 'lat',
                                                                         'lng1': 'lng',
                                                                         'alt0': None,
                                                                         'alt1': None},
                                                             'isVisible': True,
                                                             'visConfig': {'opacity': 0.8,
                                                                           'thickness': 1,
                                                                           'colorRange': {'name': 'Global Warming',
                                                                                          'type': 'sequential',
                                                                                          'category': 'Uber',
                                                                                          'colors': ['#5A1846',
                                                                                                     '#900C3F',
                                                                                                     '#C70039',
                                                                                                     '#E3611C',
                                                                                                     '#F1920E',
                                                                                                     '#FFC300']},
                                                                           'sizeRange': [0, 10],
                                                                           'targetColor': None,
                                                                           'elevationScale': 1},
                                                             'hidden': False,
                                                             'textLabel': [{'field': None,
                                                                            'color': [255, 255, 255],
                                                                            'size': 18,
                                                                            'offset': [0, 0],
                                                                            'anchor': 'start',
                                                                            'alignment': 'center'}]},
                                                  'visualChannels': {'colorField': {'name': 'count', 'type': 'integer'},
                                                                     'colorScale': 'quantile',
                                                                     'sizeField': None,
                                                                     'sizeScale': 'linear'}},
                                                 {'id': '7jknc0w',
                                                  'type': 'line',
                                                  'config': {'dataId': 'green_taxi',
                                                             'label': 'Green taxi',
                                                             'color': [179, 173, 158],
                                                             'highlightColor': [252, 242, 26, 255],
                                                             'columns': {'lat0': 'lat_PU',
                                                                         'lng0': 'lng_PU',
                                                                         'lat1': 'lat',
                                                                         'lng1': 'lng',
                                                                         'alt0': None,
                                                                         'alt1': None},
                                                             'isVisible': True,
                                                             'visConfig': {'opacity': 0.8,
                                                                           'thickness': 1,
                                                                           'colorRange': {'name': 'Global Warming',
                                                                                          'type': 'sequential',
                                                                                          'category': 'Uber',
                                                                                          'colors': ['#5A1846',
                                                                                                     '#900C3F',
                                                                                                     '#C70039',
                                                                                                     '#E3611C',
                                                                                                     '#F1920E',
                                                                                                     '#FFC300']},
                                                                           'sizeRange': [0, 10],
                                                                           'targetColor': None,
                                                                           'elevationScale': 1},
                                                             'hidden': False,
                                                             'textLabel': [{'field': None,
                                                                            'color': [255, 255, 255],
                                                                            'size': 18,
                                                                            'offset': [0, 0],
                                                                            'anchor': 'start',
                                                                            'alignment': 'center'}]},
                                                  'visualChannels': {'colorField': {'name': 'count', 'type': 'integer'},
                                                                     'colorScale': 'quantile',
                                                                     'sizeField': None,
                                                                     'sizeScale': 'linear'}}],
                                      'interactionConfig': {
                                          'tooltip': {'fieldsToShow': {'yellow_taxi': [{'name': 'PULocationID',
                                                                                        'format': None},
                                                                                       {'name': 'DOLocationID',
                                                                                        'format': None},
                                                                                       {'name': 'LocationID_x',
                                                                                        'format': None},
                                                                                       {'name': 'LocationID_y',
                                                                                        'format': None},
                                                                                       {'name': 'count',
                                                                                        'format': None}],
                                                                       'green_taxi': [
                                                                           {'name': 'PULocationID', 'format': None},
                                                                           {'name': 'DOLocationID', 'format': None},
                                                                           {'name': 'LocationID_x', 'format': None},
                                                                           {'name': 'LocationID_y', 'format': None},
                                                                           {'name': 'count', 'format': None}]},
                                                      'compareMode': False,
                                                      'compareType': 'absolute',
                                                      'enabled': True},
                                          'brush': {'size': 0.5, 'enabled': False},
                                          'geocoder': {'enabled': False},
                                          'coordinate': {'enabled': False}},
                                      'layerBlending': 'normal',
                                      'splitMaps': [{'layers': {'sr4d85f': False, '7jknc0w': True}},
                                                    {'layers': {'sr4d85f': True, '7jknc0w': False}}],
                                      'animationConfig': {'currentTime': None, 'speed': 1}},
                         'mapState': {'bearing': 0,
                                      'dragRotate': False,
                                      'latitude': 40.70939597499655,
                                      'longitude': -73.9480385959474,
                                      'pitch': 0,
                                      'zoom': 9.765323739323783,
                                      'isSplit': True},
                         'mapStyle': {'styleType': 'dark',
                                      'topLayerGroups': {},
                                      'visibleLayerGroups': {'label': True,
                                                             'road': True,
                                                             'border': False,
                                                             'building': True,
                                                             'water': True,
                                                             'land': True,
                                                             '3d building': False},
                                      'threeDBuildingColor': [9.665468314072013,
                                                              17.18305478057247,
                                                              31.1442867897876],
                                      'mapStyles': {}}}}
    return config


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


#Function to prepare data
def prepare_data(df, date_column, day_names, month_names):
    df_sample = process_data_temporal(df, date_column)
    df_sample["day_name"] = df_sample["day"].map(dict(enumerate(day_names)))
    df_sample["day_name"] = pd.Categorical(df_sample["day_name"], categories=day_names, ordered=True)
    return df_sample


#Function for grouping and aggregating data
def group_and_aggregate(df, group_column, agg_column=None, operation='count'):
    if operation == 'mean':
        return df.groupby(group_column)[agg_column].mean().reset_index(name=agg_column + '_mean')
    else:
        return df.groupby(group_column)[group_column].count().reset_index(name='count')


def plot_neo_lineplot(df, period, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    neos_per_week = df.groupby(period)['id'].nunique()
    neos_per_week.plot(kind='line', marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_neo_histogram(df, distribution, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = df[distribution]
    ax.hist(sizes, bins=20, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_neo_bar(df, period, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    average_size_per_week = df.groupby(period)['estimated_diameter_min (m)'].mean()
    average_size_per_week.plot(kind='bar', color='coral', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ticks = [0] + list(range(4, max(df['week']) + 1, 5))
    ax.set_xticks(ticks)


def plot_neo_boxplot(df, x, y, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ticks = [0] + list(range(4, max(df['week']) + 1, 5))
    ax.set_xticks(ticks)


def plot_pie_chart(data, labels_column, values_column, title, colors, explode=None):
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        data[values_column],
        explode=explode,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )

    # Change the color and increase the size of the percentage texts
    for text in autotexts:
        text.set_color('white')  # Change to a color that contrasts well with your pie colors
        text.set_size(10)  # You can adjust the size as needed
        text.set_weight('bold')  # Optional: set to 'bold' for better readability

    # We'll use the 'labels_column' to add the legend to the pie chart.
    ax.legend(
        wedges,
        data[labels_column],
        title="",
        bbox_to_anchor=(1.4, 0.5),
        loc='center right'
    )
    ax.set_aspect('equal')  # Equal aspect ratio ensures that the pie chart is circular
    plt.title(title, fontsize=16, weight='bold')


def plot_scatter_ass2(df, x_column, y_column, color, title, ax, x_label, y_label, trendline=False):
    # Scatter plot
    scatter = ax.scatter(df[x_column], df[y_column], alpha=0.5, color=color, s=10)

    # Optional trend line
    if trendline:
        z = np.polyfit(df[x_column].dropna(), df[y_column].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[x_column], p(df[x_column]), color="red", linewidth=1, alpha=0.5, label=f'Linear trend (y={z[0]:.2f}x+{z[1]:.2f})')

    # Logarithmic scale for the x-axis
    ax.set_xscale('log')
    # Enhance readability
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Clean up the plot - remove gridlines and box border for a cleaner look
    ax.grid(True, which="both", ls="--", lw=0.5, color='gray', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')

    # Include a legend if a trend line is plotted
    if trendline:
        legend = ax.legend(frameon=False, loc='upper left')
        for text in legend.get_texts():
            text.set_color('gray')


def plot_interactive_scatter(df, x_column, y_column, title, x_label, y_label):
    df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
    df = df.dropna(subset=[x_column, y_column])

    # Create the interactive scatter plot with Plotly Express
    fig = px.scatter(df, x=x_column, y=y_column, log_x=True, color_discrete_sequence=['green'],
                     title=title, labels={x_column: x_label, y_column: y_label})

    # Improve layout for better readability
    fig.update_layout(
        legend_title_text='Legend',
        plot_bgcolor='white',
        xaxis=dict(
            title=x_label,
            linecolor='black',
            showgrid=False,
            gridcolor='LightGrey',
            gridwidth=0.5,
        ),
        yaxis=dict(
            title=y_label,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGrey',
            gridwidth=0.5,
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        )
    )

    # Update marker size and opacity for better visibility
    fig.update_traces(marker=dict(size=8, opacity=0.7))

    # Add interactive note
    fig.add_annotation(
        text="Interactive: Hover over data points for more information.",
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.07,
        bordercolor='black',
        borderwidth=1
    )


def plot_interactive_histogram(df, column, title, x_label):
    # Convert the column to numeric and drop NaN values
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])

    # Calculate the 95th percentile value
    percentile_95 = df[column].quantile(0.95)

    # Cap values at the 95th percentile
    df[column] = np.where(df[column] > percentile_95, percentile_95, df[column])

    # Create the histogram with Plotly Express
    fig = px.histogram(df, x=column, nbins=50, title=title)

    # Custom layout adjustments
    fig.update_layout(
        plot_bgcolor='white',  # Set the background color to white
        xaxis=dict(
            title=x_label,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGrey',
            gridwidth=0.5,
            fixedrange=True,
        ),
        yaxis=dict(
            title='Count',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGrey',
            gridwidth=0.5,
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        # Add annotation for the last bin
        annotations=[
            dict(
                x=percentile_95,
                y=0,
                xref="x",
                yref="paper",
                text="All data above 95th percentile, max size 5000 (m)",
                showarrow=True,
                arrowhead=5,
                ax=-25,
                ay=30
            )
        ]
    )

    # Update the marker color and the line color around the bars
    fig.update_traces(marker_color='RoyalBlue', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)

    # Add interactive note
    fig.add_annotation(
        text="Interactive: Hover over bars for more information.",
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.07,
        bordercolor='black',
        borderwidth=1
    )


def plot_interactive_box(df, column, title, y_label):
    # Ensure the data type of the column is numeric for Plotly
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop NaN values that could have been introduced during conversion
    df = df.dropna(subset=[column])

    # Filter out non-positive values for log scale
    df = df[df[column] > 0]

    # Create the interactive box plot with a customized outlier definition
    fig = px.box(
        df,
        y=column,
        title=title,
        labels={column: y_label},
        notched=True,  # Shows the confidence interval around the median
        log_y=True,  # Apply a log scale to the y-axis
        points='suspectedoutliers',  # Show only suspected outliers
    )

    # Improve layout
    fig.update_layout(
        plot_bgcolor='white',  # Set the background color to white
        yaxis=dict(
            title=y_label,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGrey',
            gridwidth=0.5,
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        )
    )

    # Customize the box for better emphasis
    fig.update_traces(boxmean=True,  # Show the mean of the distribution
                      marker=dict(size=2, opacity=0.5),  # Reduce marker size for outliers
                      line=dict(width=2))  # Increase linewidth to emphasize the box

    # Add interactive note
    fig.add_annotation(
        text="Interactive: Hover over the plot for more information.",
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0,
        y=1.07,
        bordercolor='black',
        borderwidth=1
    )