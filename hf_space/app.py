"""
Gradio app for calibration, ECE, decision curve visualization
"""
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from src.metrics.calibration import reliability_curve
from src.metrics.decision_curve import decision_curve
from src.metrics.ece import compute_ece

DEMO_PATH = 'data/synthetic/first24h_minimal.csv'

# Helper to plot and save to temp file
def plot_calibration(y_true, y_prob):
    prob_true, prob_pred = reliability_curve(y_true, y_prob)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title('Reliability Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name)
    plt.close()
    return tmp.name

def plot_ece(y_true, y_prob):
    ece = compute_ece(y_true, y_prob)
    plt.figure()
    plt.bar(['ECE'], [ece])
    plt.title('Expected Calibration Error')
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name)
    plt.close()
    return tmp.name

def plot_decision_curve(y_true, y_prob):
    curve = decision_curve(y_true, y_prob)
    thresholds, net_benefits = zip(*curve)
    plt.figure()
    plt.plot(thresholds, net_benefits)
    plt.title('Decision Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name)
    plt.close()
    return tmp.name

def demo():
    df = pd.read_csv(DEMO_PATH)
    y_true = df['label'].values
    y_prob = df['age'] / 100  # Demo: use age/100 as fake probability
    return y_true, y_prob

def process_file(file):
    df = pd.read_csv(file.name)
    y_true = df['y_true'].values
    y_prob = df['y_prob'].values
    return y_true, y_prob

def app_interface(file=None):
    if file is None:
        y_true, y_prob = demo()
    else:
        y_true, y_prob = process_file(file)
    cal_fig = plot_calibration(y_true, y_prob)
    ece_fig = plot_ece(y_true, y_prob)
    dec_fig = plot_decision_curve(y_true, y_prob)
    return cal_fig, ece_fig, dec_fig

demo_btn = gr.Button("Use synthetic demo")
file_input = gr.File(label="Upload CSV (y_true, y_prob)")
cal_img = gr.Image()
ece_img = gr.Image()
dec_img = gr.Image()

def run_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Calibration & Decision Curve Explorer")
        with gr.Row():
            file_input.render()
            demo_btn.render()
        with gr.Row():
            cal_img.render()
            ece_img.render()
            dec_img.render()
        def update(file):
            return app_interface(file)
        file_input.change(update, inputs=[file_input], outputs=[cal_img, ece_img, dec_img])
        demo_btn.click(lambda: app_interface(None), outputs=[cal_img, ece_img, dec_img])
    demo.launch()

if __name__ == '__main__':
    run_app()
