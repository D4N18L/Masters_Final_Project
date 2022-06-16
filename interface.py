from preprocess import Handling_Data
import gradio as gr


def main():
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


if __name__ == "__main__":
    main()
