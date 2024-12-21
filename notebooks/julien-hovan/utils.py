import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_feature_interactive(df, label_name='label', excluded_features=None):
    """
    Creates an interactive dropdown to select features and plot their distribution.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_name (str, optional): The name of the label column. Defaults to 'label'.
        excluded_features (list, optional): List of features to exclude. Defaults to None.
    """
    if excluded_features is None:
        excluded_features = []

    features = [col for col in df.columns if col != label_name and col != 'filename' and col not in excluded_features]

    feature_dropdown = widgets.SelectMultiple(
        options=features,
        description='Features:',
        disabled=False
    )
    
    plot_type_dropdown = widgets.Dropdown(
        options=['histogram', 'violinplot', 'boxplot'],
        value='histogram',
        description='Plot Type:',
        disabled=False,
    )

    output = widgets.Output()

    def update_plot(*args):
        selected_features = feature_dropdown.value
        plot_type = plot_type_dropdown.value
        with output:
            clear_output(wait=True)
            if not selected_features:
                print("Please select at least one feature.")
                return
            for feature_name in selected_features:
                plt.figure(figsize=(10, 6))
                if plot_type == 'histogram':
                    sns.histplot(df[feature_name], kde=True)
                    plt.title(f'Histogram of {feature_name}')
                elif plot_type == 'violinplot':
                    sns.violinplot(x=df[feature_name])
                    plt.title(f'Violin Plot of {feature_name}')
                elif plot_type == 'boxplot':
                    sns.boxplot(x=df[feature_name])
                    plt.title(f'Box Plot of {feature_name}')
                plt.xlabel(feature_name)
                plt.show()

    feature_dropdown.observe(update_plot, names='value')
    plot_type_dropdown.observe(update_plot, names='value')

    display(feature_dropdown, plot_type_dropdown, output)

# create interactive plot for aggregated features by label
def plot_feature_by_label_interactive(df, label_name='label', excluded_features=None):
    """
    Creates an interactive dropdown to select features and plot their distribution by label.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_name (str, optional): The name of the label column. Defaults to 'label'.
        excluded_features (list, optional): List of features to exclude. Defaults to None.
    """
    if excluded_features is None:
        excluded_features = []

    features = [col for col in df.columns if col != label_name and col != 'filename' and col not in excluded_features]
    
    feature_dropdown = widgets.SelectMultiple(
        options=features,
        description='Features:',
        disabled=False
    )
    
    output = widgets.Output()

    def update_plot(*args):
        selected_features = feature_dropdown.value
        with output:
            clear_output(wait=True)
            if not selected_features:
                print("Please select at least one feature.")
                return
            for feature_name in selected_features:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=label_name, y=feature_name, data=df)
                plt.title(f'{feature_name} by {label_name}')
                plt.xticks(rotation=45)
                plt.show()

    feature_dropdown.observe(update_plot, names='value')
    
    display(feature_dropdown, output)
