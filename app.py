from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import gradio as gr

ROOT = Path(__file__).resolve().parent


def _load_hero_image() -> str | None:
    hero = ROOT / "images" / "tts_075b_hf_page.png"
    if hero.exists():
        return str(hero)
    return None


def fake_streaming_tts(
    text: str,
    speaker_hint: str,
    speaking_rate: float,
    prosody_strength: float,
    temperature: float,
    chunk_ms: int,
) -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not text.strip():
        explanation = "(demo) Please provide a short sentence to simulate TTS on public datasets."
    else:
        explanation = (
            "[Demo Output] This WebUI describes how a streaming TTS model trained on public datasets would behave.\n\n"
            f"- Input text snippet:\n{text.strip()}\n\n"
            "- In practice, the model would have been trained on open TTS corpora and evaluated on standardized benchmarks."
        )
    config = (
        f"timestamp: {ts}\n"
        f"speaker hint: {speaker_hint or 'N/A'}\n"
        f"speaking rate: {speaking_rate:.2f}\n"
        f"prosody strength: {prosody_strength:.2f}\n"
        f"temperature: {temperature:.2f}\n"
        f"chunk size: {chunk_ms} ms\n"
    )
    return explanation, config


def build_app() -> gr.Blocks:
    hero_path = _load_hero_image()
    with gr.Blocks(title="Public Dataset Streaming TTS WebUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Public Dataset Streaming TTS WebUI (Demo)")
        if hero_path is not None:
            with gr.Row():
                gr.Image(value=hero_path, label="Model card screenshot", type="filepath")
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="English text to synthesize", lines=6)
                speaker_hint = gr.Textbox(label="Speaker hint (optional)")
                with gr.Accordion("Advanced parameters", open=False):
                    speaking_rate = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="speaking rate")
                    prosody_strength = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="prosody strength")
                    temperature = gr.Slider(0.3, 1.5, value=0.9, step=0.05, label="temperature")
                    chunk_ms = gr.Slider(80, 320, value=160, step=10, label="chunk size (ms)")
                run_btn = gr.Button("Run streaming TTS (demo)", variant="primary")
            with gr.Column(scale=3):
                explanation = gr.Textbox(label="Generation process (textual description)", lines=12)
                config = gr.Textbox(label="Configuration summary", lines=8)
        run_btn.click(
            fn=fake_streaming_tts,
            inputs=[text, speaker_hint, speaking_rate, prosody_strength, temperature, chunk_ms],
            outputs=[explanation, config],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7873, share=False)
