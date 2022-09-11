from preprocess import Handling_Data
from preprocess import Merge_Datasets
import gradio as gr


def start():
    """
    Start Gradio interface with two checkboxes
    """

    start = gr.Interface(fn=Merge_Datasets.merge_datasets, height=600, width=800,
                         title="Merge Datasets",
                         description="Merge datasets",
                         parameters=[
                             gr.inputs.Checkbox(label="Merged Variables"),

                             gr.inputs.Checkbox(label="Separate Variables "),
                         ])

    start.launch(enable_queue=False, server_port=8000)

    # if the first checkbox is checked , go to the next interface
    if start.parameters[0].value:  # if the first checkbox is checked
        start.close()
        handling_data()  # go to the next interface
    if start.parameters[1].value:
        start.close()
        merge_datasets()  # go to the next interface


def handling_data():
    pre_d = gr.Interface(fn=Handling_Data.inspect_data,
                         inputs=[gr.inputs.Timeseries(label="Select a timeseries")],
                         outputs=[gr.outputs.Label(label="Recommended data preprocessing steps"),
                                  gr.outputs.HighlightedText(label="Data preprocessing steps")],
                         title="Data Preprocessing Recommendations",
                         description="This is a data preprocessing recommendation tool.\n"
                                     "It will recommend data preprocessing steps for any data set.\n",
                         author="Daniel Tiboah-Addo",
                         thumbnail="C:/Users/dtibo/OneDrive/Documents/Masters_Final_Project/rec.jpg")

    pred_algo = gr.Interface(fn=Handling_Data.take_data,
                             inputs=[gr.inputs.Timeseries(label="Select a timeseries")],
                             outputs=[gr.outputs.Label(label="Recommended algorithms"),
                                      gr.outputs.HighlightedText(label="Algorithms")],
                             title="Algorithm Recommendations",
                             description="This is an algorithm recommendation tool.\n"
                                         "It will recommend algorithms for any data set.\n",
                             author="Daniel Tiboah-Addo")

    demo = gr.TabbedInterface([pre_d, pred_algo], tab_names=["Recommend Preprocessing", " Recommend Algorithms"])

    demo.launch(enable_queue=False, server_port=8000)

    demo.close()


def ml_modelling():
    """
    Gradio interface that performs ML modelling
    """
    ml_modelling = gr.TabbedInterface([gr.Interface(fn=Handling_Data.model_data,
                                                    inputs=[gr.inputs.Timeseries(label="Select a timeseries")],
                                                    outputs=[gr.outputs.Label(label="Recommended algorithms"),
                                                             gr.outputs.HighlightedText(label="Algorithms")],
                                                    title="Algorithm Recommendations",
                                                    description="This is an algorithm recommendation tool.\n"
                                                    ,
                                                    author="Daniel Tiboah-Addo")], tab_names=["Recommend Algorithms"])

    ml_modelling.launch(enable_queue=False, server_port=8000)

    ml_modelling.close()


if __name__ == "__main__":
    handling_data()
